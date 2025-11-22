# drvae/datamodules/gmm_synthetic.py

import torch
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily, Distribution
import torch.nn.functional as F
from torch.utils.data import Dataset

def sample_normal(size, mean=0.0, std=1.0, *, generator=None, device=None, dtype=None):
    """
    Draw N(mean, std^2) samples with an optional torch.Generator for reproducibility.
    mean/std can be scalars or tensors broadcastable to `size`.
    """
    z = torch.randn(size, generator=generator, device=device, dtype=dtype)
    return z * std + mean

class GMM2D:
    def __init__(self, k, seed, symmetric_dataset=False):
        self.k = k
        self.g = torch.Generator().manual_seed(seed)

        self.mixture = self._get_mixture(symmetric_dataset)

    def _get_mixture(self, symmetric_dataset):
        logits = torch.ones(self.k, dtype=torch.float32)
        mixture_distribution = Categorical(logits=logits)
        if not symmetric_dataset:
            loc = sample_normal((self.k, 2), mean=0.0, std=4.0, generator=self.g).clamp(min=-5.0, max=5.0)
            scale = sample_normal((self.k, 2), mean=0.5, std=0.02, generator=self.g).clamp(min=0.4, max=0.6)

            comp = Independent(Normal(loc=loc, scale=scale), 1)
            return MixtureSameFamily(mixture_distribution=mixture_distribution,
                                            component_distribution=comp)
        else:
            theta = torch.linspace(0, 2 * torch.pi, self.k + 1)[:-1]
            rad = 2.5
            loc = torch.stack([rad * torch.cos(theta), rad * torch.sin(theta)], dim=1)  # (K,2)
            scale = torch.ones((self.k, 2), dtype=torch.float32) * 0.25

            comp = Independent(Normal(loc=loc, scale=scale), 1)
            return MixtureSameFamily(mixture_distribution=mixture_distribution,
                                            component_distribution=comp)

    def sample(self, n: int) -> torch.Tensor:
        # manually sample using self.g for reproducibility and returning points and labels
        mixture_distribution = self.mixture.mixture_distribution
        component_distribution = self.mixture.component_distribution

        assignments = torch.multinomial(
            F.softmax(mixture_distribution.logits, dim=0), n, replacement=True, generator=self.g
        )  

        # gather component means and stds
        means = component_distribution.base_dist.loc[assignments]      # (n,2)
        stds = component_distribution.base_dist.scale[assignments]     # (n,2)

        # sample from N(means, stds^2)
        samples = sample_normal((n, 2), mean=means, std=stds, generator=self.g)
        return samples, assignments.to(dtype=torch.int64)


class RandomRotation:
    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.g = torch.Generator().manual_seed(seed)

        # reproducible rotation matrix using self.g
        A = torch.randn((dim, dim), generator=self.g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        # enforce det=+1
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        self.R = Q.float()  # use float32 for data
        self.R_inv = self.R.T  # orthogonal

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R.T

    def unrotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R_inv.T

class GMMSynth:
    def __init__(self, n_train: int, 
                 n_val: int, 
                 dim: int, 
                 k: int, 
                 seed: int, 
                 pad_sigma: float = 0.02, 
                 symmetric_dataset=False):
        self.n_train = n_train
        self.n_val = n_val
        self.dim = dim
        self.k = k
        assert pad_sigma > 0, "pad_sigma must be positive"
        self.pad_sigma = pad_sigma
        self.seed = seed

        self.rot = RandomRotation(dim, seed)
        self.gmm2d = GMM2D(k, seed, symmetric_dataset)

        self.X_train_2d, self.y_train = self.gmm2d.sample(n_train)
        self.X_val_2d, self.y_val = self.gmm2d.sample(n_val)

        X_train_padded = self._pad_dims(self.X_train_2d)
        X_val_padded = self._pad_dims(self.X_val_2d)

        self.X_train = self.rot.rotate(X_train_padded)
        self.X_val = self.rot.rotate(X_val_padded)

    def _pad_dims(self, X_2d: torch.Tensor) -> torch.Tensor:
        n = X_2d.shape[0]
        if self.dim > 2:
            # Use the same generator for reproducibility
            pad = sample_normal(
                (n, self.dim - 2),
                mean=0.0,
                std=self.pad_sigma,
                generator=self.gmm2d.g,
                device=X_2d.device,
                dtype=X_2d.dtype,
            )
            X = torch.cat([X_2d, pad], dim=1)
        else:
            X = X_2d
        return X

    def get_dataset(self, split: str):
        assert split in {"train", "val"}
        X = self.X_train if split == "train" else self.X_val
        y = self.y_train if split == "train" else self.y_val
        # Pass the oracle bits the dataset needs
        return GMMDataset(
            X=X,
            y=y,
            dim=self.dim,
            rot=self.rot,
            mixture=self.gmm2d.mixture,
            pad_sigma=self.pad_sigma
        )


class GMMDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, dim: int, rot: RandomRotation,
                 mixture: MixtureSameFamily, pad_sigma: float):
        assert X.ndim == 2 and X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.dim = dim
        self.rot = rot
        self.mixture = mixture
        self.pad_sigma = float(pad_sigma)

        # Cache component params for posterior computations
        base = self.mixture.component_distribution.base_dist   # Normal with shape (K,2)
        self._loc = base.loc                                  # (K,2)
        self._scale = base.scale                              # (K,2)
        self._logits = self.mixture.mixture_distribution.logits  # (K,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    # --- helpers ---
    def _unrotate_split(self, x: torch.Tensor):
        """
        Unrotate x (N,D) back to the construction frame and split into
        the first two 'signal' dims and the (D-2) 'pad' dims.
        """
        z = self.rot.unrotate(x)                # (N,D)
        z2 = z[:, :2]                           # (N,2)
        zpad = z[:, 2:] if self.dim > 2 else z.new_zeros((z.shape[0], 0))
        return z2, zpad

    # --- oracle APIs ---
    def oracle_logpdf_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        log p_2d(x2) where x is (N,D): unrotate then keep first two dims and
        evaluate the 2D GMM density.
        """
        x2, _ = self._unrotate_split(x)
        # MixtureSameFamily.log_prob expects (...,2) → returns (N,)
        return self.mixture.log_prob(x2)

    def oracle_logpdf_pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        log p_pad(x_pad) where x is (N,D): unrotate then keep last (D-2) dims
        and evaluate the iid Gaussian density with std=pad_sigma.
        """
        _, xpad = self._unrotate_split(x)
        if xpad.numel() == 0:
            return torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        pad_dist = Normal(loc=0.0, scale=self.pad_sigma)
        # Independent over the (D-2) dims → sum log probs
        return pad_dist.log_prob(xpad).sum(dim=1)           # (N,)

    def oracle_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        log p_D(x) under the *true* construction distribution:
        - unrotate to z
        - z[:,:2] ~ 2D GMM
        - z[:,2:] ~ iid N(0, pad_sigma^2)
        Because rotation is orthogonal, |det R| = 1, so no Jacobian term.
        """
    
        logp_2d = self.oracle_logpdf_2d(x)         # (N,)

        logp_pad = self.oracle_logpdf_pad(x)       # (N,)
        return logp_2d + logp_pad
        

    def oracle_posteriors_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        p(k | x) using the 2D GMM in the unrotated frame.
        Returns (N, K) posteriors that sum to 1 across K.
        """
        x2, _ = self._unrotate_split(x)                         # (N,2)

        # Compute log p(x2 | k) for each component k, vectorized:
        # Normal with diagonal covariance → sum over dims
        # Shape tricks: (N,1,2) vs (1,K,2) → broadcast → (N,K,2)
        x2_b = x2[:, None, :]                                   # (N,1,2)
        loc = self._loc[None, :, :]                             # (1,K,2)
        scale = self._scale[None, :, :]                         # (1,K,2)
        comp_logp = Normal(loc=loc, scale=scale).log_prob(x2_b).sum(dim=2)  # (N,K)

        log_w = F.log_softmax(self._logits, dim=0)              # (K,)
        log_post_unnorm = comp_logp + log_w                     # (N,K)
        logZ = torch.logsumexp(log_post_unnorm, dim=1, keepdim=True)
        post = torch.exp(log_post_unnorm - logZ)                # (N,K)
        return post
    
def plot_gt_mixture(mixture, out_dir: str = 'outputs/figures', grid_res: int = 200, levels: int = 50, title=''):
    """Plot ground-truth 2D GMM.

    Accepts either a params dict with keys 'mus2','covs2','pi' OR a
    torch.distributions.MixtureSameFamily object. Saves figure to
    out_dir/gt_mixture.png. Backwards compatible with previous dict API.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.distributions import MixtureSameFamily

    # normalize input: accept dict or MixtureSameFamily
    if not isinstance(mixture, MixtureSameFamily):
        raise ValueError("Expected mixture to be a MixtureSameFamily instance.")
    mix = mixture
    loc_t = mix.component_distribution.base_dist.loc.detach().cpu()
    scale_t = mix.component_distribution.base_dist.scale.detach().cpu()
    mus2 = loc_t.numpy()
    covs2 = np.stack([np.diag((scale_t[k].numpy() ** 2)) for k in range(loc_t.shape[0])], axis=0)
    try:
        probs = mix.mixture_distribution.probs.detach().cpu().numpy()
    except Exception:
        probs = torch.softmax(mix.mixture_distribution.logits.detach().cpu(), dim=0).numpy()
    pi = probs
    

    # grid covering ~4 std around all centers
    all_mu = np.vstack(mus2)
    lo = all_mu.min(axis=0) - 4.0
    hi = all_mu.max(axis=0) + 4.0
    x1 = np.linspace(lo[0], hi[0], grid_res)
    x2 = np.linspace(lo[1], hi[1], grid_res)
    X1, X2 = np.meshgrid(x1, x2)
    G = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # mixture density
    dens = np.zeros(G.shape[0], dtype=float)
    for k in range(len(pi)):
        mu = mus2[k]
        cov = covs2[k]
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        diff = G - mu
        quad = np.einsum('ni,ij,nj->n', diff, inv, diff)
        dens += pi[k] * np.exp(-0.5*quad) / (2*np.pi*np.sqrt(det))
    dens = dens.reshape(grid_res, grid_res)

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X1, X2, dens, levels=levels, cmap='viridis')
    plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label='density')
    ax.scatter(all_mu[:,0], all_mu[:,1], c='k', s=40, marker='x', label='true means')
    ax.set_title('Ground-truth 2D mixture (pre-rotation)')
    ax.set_xlabel('u₁'); ax.set_ylabel('u₂')
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'gt_mixture{title}.png')
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved ground-truth mixture plot to {out_path}")

import math
import torch
from torch.distributions import Normal

@torch.no_grad()
def check_padding_scaling(dim=5, k=7, pad_sigma=0.2, seed=42, n=4096, symmetric=False, tol=1e-5):
    """
    Verifies that for x built as in your GMMSynth:
      log p_D(x) == log p_2(u) + sum_j log N(w_j; 0, pad_sigma^2)
    where (u, w) = R^T x are the unrotated coords (u in R^2, w in R^{D-2}).
    Also checks the w=0 case: log p_D(x|w=0) == log p_2(u) - (D-2)/2 * log(2πσ^2).
    """
    # Build synthetic data
    synth = GMMSynth(n_train=n, n_val=n, dim=dim, k=k, seed=seed,
                     pad_sigma=pad_sigma, symmetric_dataset=symmetric)
    ds = synth.get_dataset("train")
    X = ds.X  # rotated observations

    # Unrotate to (u, w)
    u, w = ds._unrotate_split(X)  # shapes: (n,2), (n, D-2)

    # --- Check 1: arbitrary w (the sampled pads) ---
    logp_D = ds.oracle_logpdf(X)          # (n,)
    logp_2 = ds.oracle_logpdf_2d(X)       # (n,)
    pad_dist = Normal(0.0, pad_sigma)
    logp_pad_expected = pad_dist.log_prob(w).sum(dim=1) if w.numel() > 0 else torch.zeros_like(logp_2)
    err_arbitrary = (logp_D - (logp_2 + logp_pad_expected)).abs().max().item()
    
    if dim > 2:
        zeros = torch.zeros((u.shape[0], dim - 2), dtype=u.dtype, device=u.device)
        z_zero = torch.cat([u, zeros], dim=1)
        x_zero = ds.rot.rotate(z_zero)
        logp_D_zero = ds.oracle_logpdf(x_zero)

        log_scale_factor = -0.5 * (dim - 2) * math.log(2 * math.pi * (pad_sigma ** 2))
        err_w0 = (logp_D_zero - (logp_2 + log_scale_factor)).abs().max().item()
    else:
        err_w0 = 0.0


  

    ok = (err_arbitrary <= tol) 
    print(f"[dim={dim}, k={k}, sigma={pad_sigma}, symmetric={symmetric}] "
          f"max|err_arbitrary|={err_arbitrary:.3e}, max|err_scale|={err_w0:.3e}  -> "
          f"{'OK' if ok else 'FAIL'}")
    return err_arbitrary

if __name__ == "__main__":
    # Quick sweep
    for dim in [3, 5, 8]:
        for sigma in [0.02, 0.1, 0.5, 1.0]:
            for symmetric in [False, True]:
                check_padding_scaling(dim=dim, k=8 if symmetric else 7,
                                      pad_sigma=sigma, seed=123, n=4096,
                                      symmetric=symmetric, tol=3e-6)

    
