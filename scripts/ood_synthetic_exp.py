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

from drvae.modules.prior import GMMPrior
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent
@torch.no_grad()
def compute_prior_logprob_batch_and_kl_and_q_ent(model: Model, x: torch.Tensor, device: torch.device, true_prior: MixtureSameFamily) -> np.ndarray:
    x = x.to(device)
    
    model.eval()
    # true_prior is a MixtureSameFamily, build as MixtureSameFamily(Categorical(logits=self.logits), Independent(Normal(self.loc, self.std()), 1))
    # move it to device if needed
    true_prior = MixtureSameFamily(Categorical(logits=true_prior.mixture_distribution.logits.to(device)), 
                                  Independent(Normal(true_prior.component_distribution.base_dist.loc.to(device),
                                                                         true_prior.component_distribution.base_dist.scale.to(device)), 1))
    
    with torch.no_grad():
        qz = model.module.encode(x)
        z_mean = qz.loc  # (B, D)
        lp = model.module.prior.log_prob(z_mean) # logprob under prior of encoder mean
        # also compute KL(q||p) and entropy of q
        if isinstance(model.module.prior, GMMPrior):
            kl_q_p = kl_divergence_normal_mixture(qz, model.module.prior._mixture()).mean(-1)  # (B,)
        else:
            kl_q_p = kl_divergence_normal_mixture(qz, model.module.prior.dist).mean(-1)  # (B,)
        kl_q_true_p = kl_divergence_normal_mixture(qz, true_prior).mean(-1)  # (B,)
        q_entropy = entropy_normal(qz)  # (B,)
    return lp.cpu().numpy(), kl_q_p.cpu().numpy(), kl_q_true_p.cpu().numpy(), q_entropy.cpu().numpy()


def gather_runs_for_dataset(logs_dir: str, dataset: str) -> List[str]:
    base = os.path.join(logs_dir, dataset)
    if not os.path.isdir(base):
        return []
    runs = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full):
            runs.append(full)
    runs.sort()
    return runs


def make_synth_dataset_for_dsname(ds_name: str, n_samples: int):
    # map known dataset folders to k/dim -- reuse the mapping from synthetic_results
    ds_map = {
        'logs_synthetic_2031965': {'k': 4, 'dim': 50, 'seed': 2031965 },
        'logs_synthetic_6032000': {'k': 8, 'dim': 50, 'seed': 6032000 },
    }
    if ds_name not in ds_map:
        raise ValueError(f"Unknown dataset name {ds_name}; available: {list(ds_map.keys())}")
    info = ds_map[ds_name]
    # GMMSynth requires n_val > 0 (it samples train and val). Use a small positive n_val.
    synth = GMMSynth(n_train=60000, n_val=10000, dim=info['dim'], k=info['k'], seed=info['seed'])
    # reuse X_train as samples
    X = synth.X_train[:n_samples]  # (n_samples, dim)
    true_prior = synth.gmm2d.mixture  # a torch.distributions.MixtureSameFamily
    return X, true_prior

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


def run_experiment(logs_dir: str, a: str, b: str, out_dir: str, n_ref: int = 1000, n_ood: int = 10000,
                   batch_size: int = 256, device: str = 'cuda', dry_run: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    # find run dirs for dataset a
    runs = gather_runs_for_dataset(logs_dir, a)
    print(f"Found {len(runs)} runs in {a}")

    # prepare reference samples from A and OOD samples from B
    print(f"Generating {n_ood} OOD samples from {b}")
    ood_X_full, true_prior = make_synth_dataset_for_dsname(b, n_ood)

    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')

    out_path = os.path.join(out_dir, f'ood_elbo_results_{a}_vs_{b}.csv')
    keys = ['run_dir', 'ckpt', 'hparams', 'n_ref', 'n_ood',
            'data_metric_mean', 'data_metric_std', 'prior_metric_mean', 'prior_metric_std', 'kl_q_p_mean', 'kl_q_p_std', 'kl_q_true_p_mean', 'kl_q_true_p_std', 'q_entropy_mean', 'q_entropy_std']

    with open(out_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()

        for rd in runs:
            print(f"Processing run: {rd}")
            cks = find_checkpoints(rd)
            if not cks:
                print(f"  No checkpoints found under {rd} — skipping")
                continue
            # pick latest checkpoint
            ckpt = cks[-1]
            hparams = find_hparams_yaml(rd)

            if dry_run:
                print(f"  Dry run: would load ckpt {ckpt} with hparams {hparams}")
                writer.writerow({'run_dir': rd, 'ckpt': ckpt, 'hparams': hparams, 'n_ref': n_ref, 'n_ood': n_ood,
                                 'data_metric_mean': '', 'data_metric_std': '', 'prior_metric_mean': '', 'prior_metric_std': ''})
                continue

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



            # compute metrics on OOD set
            ood_loader = torch.utils.data.DataLoader(ood_X_full, batch_size=batch_size, shuffle=False)
            ELBO = []
            LOGPRIOR = []
            KL_Q_P = []
            KL_Q_TRUE_P = []
            Q_ENTROPY = []
            for xb in ood_loader:
                xb = xb.float()
                elbos = compute_elbo_batch(model, xb, device_t)
                prs, kl_q_p, kl_q_true_p, q_entropy = compute_prior_logprob_batch_and_kl_and_q_ent(model, xb, device_t, true_prior)
                ELBO.append(elbos)
                LOGPRIOR.append(prs)
                KL_Q_P.append(kl_q_p)
                KL_Q_TRUE_P.append(kl_q_true_p)
                Q_ENTROPY.append(q_entropy)
            ELBO = np.concatenate(ELBO)  # (n_ood,)
            LOGPRIOR = np.concatenate(LOGPRIOR)  # (n_ood,)
            KL_Q_P = np.concatenate(KL_Q_P)  # (n_ood,)
            KL_Q_TRUE_P = np.concatenate(KL_Q_TRUE_P)  # (n_ood,)
            Q_ENTROPY = np.concatenate(Q_ENTROPY)  # (n_ood,)
            data_mean = float(np.mean(ELBO))
            data_std = float(np.std(ELBO))
            prior_mean = float(np.mean(LOGPRIOR))
            prior_std = float(np.std(LOGPRIOR))
            kl_q_p_mean = float(np.mean(KL_Q_P))
            kl_q_p_std = float(np.std(KL_Q_P))
            kl_q_true_p_mean = float(np.mean(KL_Q_TRUE_P))
            kl_q_true_p_std = float(np.std(KL_Q_TRUE_P))
            q_entropy_mean = float(np.mean(Q_ENTROPY))
            q_entropy_std = float(np.std(Q_ENTROPY))

            writer.writerow({'run_dir': rd, 'ckpt': ckpt, 'hparams': hparams, 'n_ref': n_ref, 'n_ood': n_ood,
                             'data_metric_mean': data_mean, 'data_metric_std': data_std,
                             'prior_metric_mean': prior_mean, 'prior_metric_std': prior_std,
                             'kl_q_p_mean': kl_q_p_mean, 'kl_q_p_std': kl_q_p_std,
                             'kl_q_true_p_mean': kl_q_true_p_mean, 'kl_q_true_p_std': kl_q_true_p_std,
                             'q_entropy_mean': q_entropy_mean, 'q_entropy_std': q_entropy_std})
            print(f"  Wrote metrics for {rd}: data_mean={data_mean:.4g}, prior_mean={prior_mean:.4g}")

    print(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs-dir', default=os.path.join(os.path.dirname(__file__), '..', 'logs_synthetic'),
                        help='Path to logs_synthetic folder')
    parser.add_argument('--a', required=True, help='Dataset folder name for models (e.g. logs_synthetic_2031965)')
    parser.add_argument('--b', required=True, help='Dataset folder name to sample OOD points from (e.g. logs_synthetic_6032000)')
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'results'), help='Output dir')
    parser.add_argument('--n-ref', type=int, default=1000, help='Number of reference samples from A')
    parser.add_argument('--n-ood', type=int, default=10000, help='Number of OOD samples from B')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='Do not load models; only enumerate checkpoints')
    args = parser.parse_args()

    run_experiment(args.logs_dir, args.a, args.b, args.out_dir, n_ref=args.n_ref, n_ood=args.n_ood,
                   batch_size=args.batch_size, device=args.device, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
