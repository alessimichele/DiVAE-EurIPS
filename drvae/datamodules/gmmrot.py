import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RotGMMCfg:
    n_train: int = 60000
    n_val: int = 10000
    dim: int = 50
    k: int = 8
    seed: int = 123
    pad_sigma: float = 0.02  # small isotropic noise in the 48 padded dims (before rotation)
    mix_temp: float = 1.0    # soften/sharpen class weights when sampling


class RotatedGMMDataset(Dataset):
    """
    Synthetic dataset:
      1) draw z2 ~ GMM_2D (known params)
      2) pad to 50D with N(0, pad_sigma^2) in the last 48 dims
      3) apply a random orthogonal rotation R in R^{50x50}

    Provides oracle log-pdf and posteriors p(k|x) for evaluation.
    """
    def __init__(self, split: str, cfg: RotGMMCfg, R: Optional[np.ndarray] = None, gmm_params: Optional[dict] = None):
        assert split in {"train", "val"}
        self.split = split
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)
        self.N = cfg.n_train if split == "train" else cfg.n_val
        self.D = cfg.dim
        self.K = cfg.k
        # --- choose (or create) rotation ---
        if R is None:
            A = self.rng.randn(self.D, self.D)
            Q, _ = np.linalg.qr(A)
            # enforce det=+1
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            R = Q
        self.R = R.astype(np.float64)
        self.R_inv = self.R.T  # orthogonal

        # --- make a 2D GMM ---
        if gmm_params is None:
            theta = np.linspace(0, 2*np.pi, self.K, endpoint=False)
            rad = 2.5
            mus2 = np.stack([rad*np.cos(theta), rad*np.sin(theta)], axis=1)  # (K,2)
            covs2 = np.stack([np.diag([0.25, 0.25]) for _ in range(self.K)], axis=0)  # (K,2,2)
            logits = np.zeros(self.K)  # uniform
        else:
            mus2 = gmm_params["mus2"]
            covs2 = gmm_params["covs2"]
            logits = gmm_params["logits"]
        pis = np.exp(logits - logits.max())
        pis = pis / pis.sum()
        self.gmm_params = {"mus2": mus2, "covs2": covs2, "pi": pis, "logits": logits}

        # precompute Cholesky for 2D components
        self.L2 = np.stack([np.linalg.cholesky(c) for c in covs2], axis=0)  # (K,2,2)

        # sample data
        self.X, self.y, self.k_true = self._sample()

    def _sample(self):
        rng = self.rng
        cfg = self.cfg
        N, D, K = self.N, self.D, self.K
        pis = self.gmm_params["pi"] ** (1.0 / cfg.mix_temp)
        pis = pis / pis.sum()
        ks = rng.choice(K, size=N, p=pis)
        Z2 = np.zeros((N, 2), dtype=np.float64)
        for k in range(K):
            idx = np.where(ks == k)[0]
            if idx.size == 0:
                continue
            eps = rng.randn(idx.size, 2).dot(self.L2[k].T)
            Z2[idx] = self.gmm_params["mus2"][k] + eps
        # pad to D with small Gaussian noise
        pad = rng.randn(N, D-2) * self.cfg.pad_sigma
        Z = np.concatenate([Z2, pad], axis=1)
        # rotate
        X = Z.dot(self.R.T)  # (N,D)
        # data is 1x1xD for the current linear encoder/decoder
        X_t = torch.from_numpy(X.astype(np.float32)).view(N, 1, 1, D)
        y = torch.from_numpy(ks.astype(np.int64))
        return X_t, y, ks

    # ---------- oracle utils ----------
    def oracle_logpdf(self, x: np.ndarray) -> np.ndarray:
        """x: (N,D) in rotated space"""
        # back-rotate
        z = x.dot(self.R_inv.T)
        z2 = z[:, :2]
        logps = []
        for k in range(self.K):
            mu = self.gmm_params["mus2"][k]
            cov = self.gmm_params["covs2"][k]
            d = z2 - mu
            inv = np.linalg.inv(cov)
            logdet = np.log(np.linalg.det(cov))
            lp = -0.5*(np.einsum("ni,ij,nj->n", d, inv, d) + logdet + 2*np.log(2*np.pi))
            logps.append(np.log(self.gmm_params["pi"][k] + 1e-12) + lp)
        logps = np.stack(logps, axis=1)  # (N,K)
        return np.logaddexp.reduce(logps, axis=1)  # (N,)

    def oracle_posteriors(self, x: np.ndarray) -> np.ndarray:
        z = x.dot(self.R_inv.T)
        z2 = z[:, :2]
        logs = []
        for k in range(self.K):
            mu = self.gmm_params["mus2"][k]
            cov = self.gmm_params["covs2"][k]
            d = z2 - mu
            inv = np.linalg.inv(cov)
            logdet = np.log(np.linalg.det(cov))
            lp = -0.5*(np.einsum("ni,ij,nj->n", d, inv, d) + logdet + 2*np.log(2*np.pi))
            logs.append(np.log(self.gmm_params["pi"][k] + 1e-12) + lp)
        logs = np.stack(logs, axis=1)
        logs = logs - logs.max(axis=1, keepdims=True)
        p = np.exp(logs)
        p = p / p.sum(axis=1, keepdims=True)
        return p  # (N,K)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

