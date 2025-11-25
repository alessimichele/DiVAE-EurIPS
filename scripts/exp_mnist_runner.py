#!/usr/bin/env python3
"""
Runner for one MNIST experiment configured from CLI args.
This mirrors the `flow_run.py` config but is fully parameterized for SLURM-array execution.
"""
import argparse
import shlex
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

# repo bootstrap
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from drvae.model import Model
from drvae.datamodules import VAEDataModule
from drvae.callbacks import *


def build_module_config(n_latent: int = 10,  prior_kind: str = "standard", gmm_k: int = 10):
    enc_cfg = {"class": "linear_encoder", "args": {"input_shape": (1,28,28), "h_dims": [300, 300], "n_latent": n_latent}}
    dec_cfg = {"class": "linear_decoder", "args": {"n_latent": n_latent, "h_dims": [300, 300], "output_shape": (1,28,28)}}

    if prior_kind == "standard":
        prior_cfg = None
    elif prior_kind == "gmm":
        prior_cfg = {"prior": "gmm", "args": {"n_components": gmm_k}}
    elif prior_kind == "vamp":
        prior_cfg = {"prior": "vamp", "args": {"n_pseudo_inputs": 10}}
    else:
        raise ValueError("Unknown prior kind")

    module_config = {
        "module": "vae",
        "args": {
            "input_shape": (1,28,28),
            "n_latent": n_latent,
            "encoder_config": enc_cfg,
            "decoder_config": dec_cfg,
            "likelihood": "bernoulli",
            "gaussian_std": 0.1,
            "prior_config": prior_cfg,
            "aligner_config": None,
        }
    }
    return module_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gmm-k", type=int, default=10)
    p.add_argument("--prior", choices=["standard", "gmm", "vamp"], required=True)
    p.add_argument("--density-reg", action="store_true", help="Enable density regularization (reg weight warmup)")
    p.add_argument("--aligner", choices=["none", "direct", "flow"], default="none")
    p.add_argument("--external-density", choices=["dpa", "knn"], required=False)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-root", type=str, default="logs_mnist")
    args = p.parse_args()

    seed_everything(args.seed)

    # Build module config
    module_config = build_module_config(n_latent=args.n_latent, prior_kind=args.prior, gmm_k=args.gmm_k)

    # aligner config
    if args.aligner == "none":
        module_config['args']['aligner_config'] = None
    elif args.aligner == "direct":
        module_config['args']['aligner_config'] = {"aligner": "direct", "args": {"detach_encoder": False}}
    elif args.aligner == "flow":
        module_config['args']['aligner_config'] = {"aligner": "flow", "args": {
            "hidden": 128,
            "K": 8,
            "detach_encoder": False,
            "huber_delta": 1.0,
            "penalty": 1e-3,
            "add_consistency": False,
            "consistency_weight": 1e-1,
        }}

    # external density config
    if args.external_density == "dpa":
        external_density = {"method": "dpa", "pca_components": args.n_latent, "dpa_zpar": 1.96}
    elif args.external_density == "knn":
        external_density = {"method": "knn", "pca_components":  args.n_latent, "knn_k": 50}
    else:
        # default to dpa if not specified in
        args.external_density = "dpa"
        external_density = {"method": "dpa", "pca_components":  args.n_latent, "dpa_zpar": 1.96}

    if args.density_reg:
        density_reg = True
    else:
        density_reg = False
        
    kl_weight_config = {"mode": "epoch_warmup", "epochs": args.epochs//2, "min": 0.0, "max": 1.0}
    reg_weight_config = {"mode": "epoch_warmup", "epochs": args.epochs//2, "min": 0.0, "max": 1.0}
    # Build Model
    model = Model(module_config=module_config,
                  density_reg=density_reg,    
                  density_eval=True,
                  external_density_config=external_density or {"method": "dpa", "pca_components":  args.n_latent, "dpa_zpar": 1.96},
                    kl_weight_config=kl_weight_config,
                    reg_weight_config=reg_weight_config,
                )

    # DataModule
    dm = VAEDataModule(dataset='mnist', data_path=os.path.expanduser("~/datasets/"), batch_size=128,)

    # Callbacks: build default set, add OracleDensityOverride when requested, Jacobs only for flow
    callbacks = [ImageLoggingCallback(), EndOfTrainingEvalCallback(), MetricsPlotCallback(), UMAPPlotCallback(), 
                 DensityHistogramCallback(), TrainingTimeCallback()]
    
    if args.aligner == "flow":
        callbacks.insert(0, JacobianLoggingCallback())

    # Logger: create a structured log name
    base = f"{args.prior}"
    base += f"_reg{int(args.density_reg)}_{args.aligner}_ext_{args.external_density}_seed{args.seed}"
    logger = CSVLogger(save_dir=args.log_root, name=base)

    trainer = Trainer(max_epochs=args.epochs,
                      accelerator="auto",
                      devices="auto",
                      logger=logger,
                      enable_progress_bar=True,
                      callbacks=callbacks)

    # Fit
    dm.prepare_data()
    dm.setup()
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
