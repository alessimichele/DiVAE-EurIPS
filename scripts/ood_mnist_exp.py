#!/usr/bin/env python3
"""
Usage examples:
  python scripts/ood_elbo_experiment.py --a logs_synthetic_2031965 --b logs_synthetic_6032000 --n-ood 10000 --dry-run

This script assumes the project `Model` class can be loaded via
  from drvae.model import Model
and that synthetic datasets can be generated with
  from drvae.datamodules.gmm_synthetic import GMMSynth

"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional

import torch
import numpy as np
# repo bootstrap
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# local imports
from drvae.model import Model
from drvae.datamodules.gmm_synthetic import GMMSynth



def find_checkpoints(run_dir: str) -> List[str]:
    """Search for checkpoint files under run_dir (endswith .ckpt) or in checkpoints/"""
    cks = []
    if not os.path.isdir(run_dir):
        return cks
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith('.ckpt'):
                cks.append(os.path.join(root, f))
    return sorted(cks)


def find_hparams_yaml(run_dir: str) -> Optional[str]:
    p = os.path.join(run_dir, 'hparams.yaml')
    if os.path.isfile(p):
        return p
    # sometimes in version_0
    p2 = os.path.join(run_dir, 'version_0', 'hparams.yaml')
    if os.path.isfile(p2):
        return p2
    return None


@torch.no_grad()
def compute_elbo_batch(model: Model, x: torch.Tensor, device: torch.device) -> np.ndarray:
    x = x.to(device)
    # model.module is the VAE module inside Model
    model.eval()
    with torch.no_grad():
        elbo_t = model.module.elbo(x)  # (B,)
    return elbo_t.cpu().numpy()

from drvae.modules.prior import GMMPrior, VampPrior
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent
@torch.no_grad()
def compute_prior_logprob_batch_and_kl_and_q_ent(model: Model, x: torch.Tensor, device: torch.device) -> np.ndarray:
    x = x.to(device)
    
    model.eval()
   
    
    with torch.no_grad():
        qz = model.module.encode(x)
        z_mean = qz.loc  # (B, D)
        lp = model.module.prior.log_prob(z_mean) # logprob under prior of encoder mean
        # also compute KL(q||p) and entropy of q
        if isinstance(model.module.prior, GMMPrior):
            kl_q_p = kl_divergence_normal_mixture(qz, model.module.prior._mixture()).mean(-1)  # (B,)
        elif isinstance(model.module.prior, VampPrior):
            kl_q_p = kl_divergence_normal_mixture(qz, model.module.prior._mixture()).mean(-1)  # (B,)
        else:
            kl_q_p = kl_divergence_normal_mixture(qz, model.module.prior.dist).mean(-1)  # (B,)
        q_entropy = entropy_normal(qz)  # (B,)
    return lp.cpu().numpy(), kl_q_p.cpu().numpy(), q_entropy.cpu().numpy()


def gather_runs_for_dataset(logs_dir: str) -> List[str]:
    base = os.path.join(logs_dir)
    if not os.path.isdir(base):
        return []
    runs = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full):
            runs.append(full)
    runs.sort()
    return runs



from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent
def kl_divergence_normal_mixture(q: Normal,
                                p: MixtureSameFamily,
                                num_samples: int=1000) -> torch.Tensor:
    """
    Compute KL divergence between q(z) ~ Normal(loc, scale)
    and p(z) ~ MixtureSameFamily(Categorical, Independent(Normal, 1)).
    
    Args:
        q: Normal distribution with batch shape [batch_size, L]
        p: MixtureSameFamily with batch shape [K, L]
        num_samples: Number of samples to approximate the expectation.
        
    Returns:
        kl: Tensor of shape [batch_size], the KL divergence for each batch element.
    """
    z = q.rsample((num_samples,))  # [num_samples, batch_size, L]
    log_qz = Independent(q, 1).log_prob(z)  # [num_samples, batch_size]
    log_pz = p.log_prob(z)  # [num_samples, batch_size]
    kl = log_qz - log_pz  # [num_samples, batch_size]
    return kl.permute(1, 0)  # [batch_size, num_samples]

def entropy_normal(q: Normal, num_samples: int=1000) -> torch.Tensor:
    """
    Compute entropy of a Normal distribution q(z) ~ Normal(loc, scale).
    
    Args:
        q: Normal distribution with batch shape [batch_size, L]
        num_samples: Number of samples to approximate the expectation.
        
    Returns:
        entropy: Tensor of shape [batch_size], the entropy for each batch element.
    """
    # Entropy of a multivariate normal is known in closed form
    return Independent(q, 1).entropy()  # [batch_size]


def run_experiment(logs_dir: str, out_dir: str, n_ref: int = 1000, n_ood: int = 10000,
                   batch_size: int = 256, device: str = 'cuda'):
    os.makedirs(out_dir, exist_ok=True)
    # find run dirs for dataset a
    runs = gather_runs_for_dataset(logs_dir)
    
    # prepare reference samples from A and OOD samples from B
    from drvae.datamodules import VAEDataModule
    ood_dm = VAEDataModule(dataset='fashionmnist')
    ood_dm.prepare_data()
    ood_dm.setup()
    ood_X_dataloader = ood_dm.train_dataloader()

    true_dm = VAEDataModule(dataset='mnist')
    true_dm.prepare_data()
    true_dm.setup()
    true_X_dataloader = true_dm.train_dataloader()

    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')

    out_path = os.path.join(out_dir, f'ood_elbo_results_mnist_vs_fashionmnist.csv')
    keys = ['prior_type', 'density_reg', 'aligner_type', 'seed',
            'true_elbo_mean', 'true_elbo_std', 'ood_elbo_mean', 'ood_elbo_std', 'elbo_diff_mean', 
            'true_logprior_mean', 'true_logprior_std', 'ood_logprior_mean', 'ood_logprior_std', 'logprior_diff_mean',
            'kl_q_prior_mean', 'kl_q_prior_std', 'kl_qtrue_prior_mean', 'kl_qtrue_prior_std', 'kl_qprior_diff_mean',
            'q_entropy_mean', 'q_entropy_std', 'qtrue_entropy_mean', 'qtrue_entropy_std', 'qentropy_diff_mean',
            ]

    with open(out_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()

        for rd in runs:
            print(f"Processing run: {rd}")
            seed = rd.split('/')[-1].split('_')[-1].strip()[-1]
            cks = find_checkpoints(rd)
            if not cks:
                print(f"  No checkpoints found under {rd} — skipping")
                continue
            # pick latest checkpoint
            ckpt = cks[-1]
            hparams = find_hparams_yaml(rd)
            # hparams is a path of a yaml. Read it as a dict
            import yaml
            with open(hparams, 'r') as f:
                hparams_dict = yaml.safe_load(f) if hparams else {}
            if hparams_dict['module_config']['args']['prior_config'] is None:
                prior_type = 'standard'
            else:
                prior_type = hparams_dict['module_config']['args']['prior_config']['prior']
            density_reg = hparams_dict['density_reg']
            aligner_type = 'none'
            if density_reg:
                aligner_type = hparams_dict['module_config']['args']['aligner_config']['aligner']
           
    
            # load model
            try:
                if hparams:
                    model = Model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=False)
                else:
                    model = Model.load_from_checkpoint(ckpt, strict=False)
            except Exception as e:
                print(f"  Failed to load checkpoint {ckpt}: {e}")
                continue

            model.to(device_t)
            model.eval()
            import json
            true_metrics_path = os.path.join(rd, 'version_0', 'figures', 'metrics_eval.json')
            true_metrics_dict = json.load(open(true_metrics_path, 'r'))
            true_elbo_mean = true_metrics_dict['elbo_mean']
            true_elbo_std = true_metrics_dict['elbo_std']
            true_logprior_mean = true_metrics_dict['prior_logpdf_mean']
            true_logprior_std = true_metrics_dict['prior_logpdf_std']
               
            
            # compute metrics on OOD set
            ood_loader = ood_X_dataloader
            ELBO = []
            LOGPRIOR = []
            KL_Q_P = []
            Q_ENTROPY = []
            for batch in ood_loader:
                xy, indexes = batch
                xb, _ = xy
                assert xb.dtype == torch.float32
                assert xb.min() >= 0.0 and xb.max() <= 1.0
                elbos = compute_elbo_batch(model, xb, device_t)
                prs, kl_q_p, q_entropy = compute_prior_logprob_batch_and_kl_and_q_ent(model, xb, device_t)
                ELBO.append(elbos)
                LOGPRIOR.append(prs)
                KL_Q_P.append(kl_q_p)
                Q_ENTROPY.append(q_entropy)
            ELBO = np.concatenate(ELBO)  # (n_ood,)
            LOGPRIOR = np.concatenate(LOGPRIOR)  # (n_ood,)
            KL_Q_P = np.concatenate(KL_Q_P)  # (n_ood,)
            Q_ENTROPY = np.concatenate(Q_ENTROPY)  # (n_ood,)
            data_mean = float(np.mean(ELBO))
            data_std = float(np.std(ELBO))
            prior_mean = float(np.mean(LOGPRIOR))
            prior_std = float(np.std(LOGPRIOR))
            kl_q_p_mean = float(np.mean(KL_Q_P))
            kl_q_p_std = float(np.std(KL_Q_P))
            q_entropy_mean = float(np.mean(Q_ENTROPY))
            q_entropy_std = float(np.std(Q_ENTROPY))

            KL_QTRUE_P = []
            QTRUE_ENTROPY = []
            for batch in true_X_dataloader:
                xy, indexes = batch
                xb, _ = xy
                assert xb.dtype == torch.float32
                assert xb.min() >= 0.0 and xb.max() <= 1.0
                _, kl_qtrue_p, qtrue_entropy = compute_prior_logprob_batch_and_kl_and_q_ent(model, xb, device_t)
                KL_QTRUE_P.append(kl_qtrue_p)
                QTRUE_ENTROPY.append(qtrue_entropy)
            KL_QTRUE_P = np.concatenate(KL_QTRUE_P)  
            QTRUE_ENTROPY = np.concatenate(QTRUE_ENTROPY) 
            kl_qtrue_p_mean = float(np.mean(KL_QTRUE_P))
            kl_qtrue_p_std = float(np.std(KL_QTRUE_P))
            qtrue_entropy_mean = float(np.mean(QTRUE_ENTROPY))
            qtrue_entropy_std = float(np.std(QTRUE_ENTROPY))

            writer.writerow({
                'prior_type': prior_type,
                'density_reg': density_reg,
                'aligner_type': aligner_type,
                'seed': seed,
                'true_elbo_mean': true_elbo_mean,
                'true_elbo_std': true_elbo_std,
                'ood_elbo_mean': data_mean,
                'ood_elbo_std': data_std,
                'elbo_diff_mean': data_mean - true_elbo_mean,
                'true_logprior_mean': true_logprior_mean,
                'true_logprior_std': true_logprior_std,
                'ood_logprior_mean': prior_mean,
                'ood_logprior_std': prior_std,
                'logprior_diff_mean': prior_mean - true_logprior_mean,
                'kl_q_prior_mean': kl_q_p_mean,
                'kl_q_prior_std': kl_q_p_std,
                'kl_qtrue_prior_mean': kl_qtrue_p_mean,
                'kl_qtrue_prior_std': kl_qtrue_p_std,
                'kl_qprior_diff_mean': kl_q_p_mean - kl_qtrue_p_mean,
                'q_entropy_mean': q_entropy_mean,
                'q_entropy_std': q_entropy_std,
                'qtrue_entropy_mean': qtrue_entropy_mean,
                'qtrue_entropy_std': qtrue_entropy_std,
                'qentropy_diff_mean': q_entropy_mean - qtrue_entropy_mean
            })
            print(f"  Wrote metrics for {rd}")

    print(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs-dir', default=os.path.join(os.path.dirname(__file__), '..', 'logs_mnist'),
                        help='Path to logs_synthetic folder')
   
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), '..'), help='Output dir')
    parser.add_argument('--n-ref', type=int, default=1000, help='Number of reference samples from A')
    parser.add_argument('--n-ood', type=int, default=10000, help='Number of OOD samples from B')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='Do not load models; only enumerate checkpoints')
    args = parser.parse_args()

    run_experiment(args.logs_dir, args.out_dir, n_ref=args.n_ref, n_ood=args.n_ood,
                   batch_size=args.batch_size, device=args.device)


if __name__ == '__main__':
    main()
