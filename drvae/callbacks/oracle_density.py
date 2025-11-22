# drvae/callbacks/oracle_density.py

import numpy as np
import torch
import pytorch_lightning as pl
from typing import Tuple, Union
from drvae.modules.prior import GMMPrior, StandardNormalPrior
import os
import matplotlib.pyplot as plt
import warnings
import logging
from ..model import Model
from ..modules.vae import VAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# drvae/callbacks/_ds_utils.py
from torch.utils.data import ConcatDataset
from ..datamodules.gmm_synthetic import GMMDataset

def _find_rotgmm(dataset) -> Union[GMMDataset, None]:
    """Return the underlying GMMDataset if present, else None."""
    try:
        from drvae.datamodules.gmm_synthetic import GMMDataset
    except Exception:
        class GMMDataset:  # dummy fallback to avoid import errors in non-synth runs
            pass

    visited = set()

    def rec(ds):
        oid = id(ds)
        if oid in visited:
            return None
        visited.add(oid)

        if isinstance(ds, GMMDataset):
            return ds

        if isinstance(ds, ConcatDataset):
            for child in ds.datasets:
                got = rec(child)
                if got is not None:
                    return got
            return None

        # common single-child wrappers
        for attr in ("dataset", "base_dataset"):
            if hasattr(ds, attr):
                return rec(getattr(ds, attr))

        return None

    return rec(dataset)


def get_oracle_funcs(dataset):
    """
    """
    root = _find_rotgmm(dataset)
    if root is None:
        return None, None
    oracle_logpdf = getattr(root, "oracle_logpdf", None)
    oracle_logpdf_2d = getattr(root, "oracle_logpdf_2d", None)
    oracle_logpdf_pad  = getattr(root, "oracle_logpdf_pad", None)
    oracle_posteriors_2d = getattr(root, "oracle_posteriors_2d", None)
    return {"oracle_logpdf": oracle_logpdf,
            "oracle_posteriors_2d": oracle_posteriors_2d,
            "oracle_logpdf_2d": oracle_logpdf_2d,
            "oracle_logpdf_pad": oracle_logpdf_pad}, root

from torch.distributions import MixtureSameFamily, Independent, Normal

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
    # Sample z ~ q
    z = q.rsample((num_samples,))  # [num_samples, batch_size, L]

    # Compute log q(z)
    log_qz = Independent(q, 1).log_prob(z)  # [num_samples, batch_size]
    
    # Compute log p(z)
    log_pz = p.log_prob(z)  # [num_samples, batch_size]

    # Compute KL as expectation over q
    kl = log_qz - log_pz  # [num_samples, batch_size]
    #kl = kl.mean(dim=0)  # Average over samples, shape [batch_size]
    return kl.permute(1, 0)  # [batch_size, num_samples]

def kl_divergence_mixture_mixture(q: MixtureSameFamily, 
                                p: MixtureSameFamily,
                                num_samples: int = 1000,) -> torch.Tensor:
    """
    Compute KL divergence between q(z) ~ MixtureSameFamily(Categorical, Independent(Normal, 1))
    and p(z) ~ MixtureSameFamily(Categorical, Independent(Normal, 1)).

    """
    # Sample z ~ q
    z = q.sample((num_samples,)) # [num_samples, L]

    # Compute log q(z)
    log_qz = q.log_prob(z)  # [num_samples, ]

    # Compute log p(z)
    log_pz = p.log_prob(z)  # [num_samples, ]

    # Compute KL as expectation over q
    kl = log_qz - log_pz  # [num_samples, ]

    return kl.mean(-1)  # [, ]

# =====================================
# Override Computed Density with DPA/KNN/KDE with Ground-Truth Densities
# =====================================
class OracleDensityOverride(pl.Callback):
    """
    Override model log-density estimates with true oracle log-density for synthetic GMMDataset.
    """
    def _collect(self, dl) -> Tuple[np.ndarray, np.ndarray]:
        X, I = [], []
        for (x, _y), idx in dl:
            X.append(x.view(x.size(0), -1))
            I.append(idx)
        X = torch.cat(X, dim=0).cpu()
        I = torch.cat(I, dim=0)
        return X, I

    @torch.no_grad()
    def on_train_start(self, trainer: pl.Trainer, pl_module: Model):
        dm = trainer.datamodule
        if  _find_rotgmm(dm.train_dataset) is None:
            warnings.warn("[OracleDensityOverride] callback is only applicable to GMMDataset; skipping.")
            return
        
        logger.info("[OracleDensityOverride] Computing oracle log-density...")
        # ---- TRAIN ----
        funcs, root_train = get_oracle_funcs(dm.train_dataset)
        oracle_logpdf_2d = funcs.get("oracle_logpdf_2d", None)
        if oracle_logpdf_2d is not None:
            Xtr, Itr = self._collect(dm.train_dataloader(shuffle=False))
            log_den_tr = oracle_logpdf_2d(Xtr)
        err_tr = np.full_like(log_den_tr, fill_value=0.0, dtype=np.float32)
        # dense reorder
        log_den_t  = torch.tensor(log_den_tr, dtype=torch.float32, device=pl_module.device)
        log_err_t  = torch.tensor(err_tr, dtype=torch.float32, device=pl_module.device)
        dense_ld   = torch.empty_like(log_den_t)
        dense_err  = torch.empty_like(log_err_t)
        dense_ld[Itr]  = log_den_t
        dense_err[Itr] = log_err_t
        pl_module.register_buffer('log_den', dense_ld)
        pl_module.register_buffer('log_den_err', torch.clamp(dense_err, min=1e-3))
        pl_module.register_buffer('indexes', Itr.to(pl_module.device))
        # ---- VAL ----
        funcs_val, root_val = get_oracle_funcs(dm.val_dataset)
        oracle_logpdf_2d_val = funcs_val.get("oracle_logpdf_2d", None)
        if oracle_logpdf_2d_val is not None:
            Xva, Iva = self._collect(dm.val_dataloader(shuffle=False))
            log_den_va = oracle_logpdf_2d_val(Xva)
        err_va = np.full_like(log_den_va, fill_value=0.0, dtype=np.float32)

        log_den_v  = torch.tensor(log_den_va, dtype=torch.float32, device=pl_module.device)
        log_err_v  = torch.tensor(err_va, dtype=torch.float32, device=pl_module.device)
        d_ld   = torch.empty_like(log_den_v)
        d_err  = torch.empty_like(log_err_v)
        d_ld[Iva]  = log_den_v
        d_err[Iva] = log_err_v
        pl_module.register_buffer('val_log_den', d_ld)
        pl_module.register_buffer('val_log_den_err', torch.clamp(d_err, min=1e-3))
        pl_module.register_buffer('val_indexes', Iva.to(pl_module.device))

# =====================================
# Weights and score histograms
# Compute the histogram of w_i and w_i(s_i - rho_i) at the end of training
# =====================================
class WeightsAndScoresHistogramCallback(pl.Callback):
    """
    Store the true dataspace log-density and prior log-density in additional model buffers.
    """

    def __init__(self,bins: int = 60,
                 range_percentiles=(1, 99)):
        super().__init__()
        self.bins = bins
        self.range_percentiles = range_percentiles

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: Model):
        dm = trainer.datamodule
        dl = dm.val_dataloader(shuffle=False)
        m = pl_module.module

        WEIGHTS, SCORES = [], []
        device = next(pl_module.parameters()).device
        for (x,_), _idx in dl:
            x = x.to(device)
            x = x.view(x.size(0), -1)
            
            val_log_den_err = pl_module.val_log_den_err[_idx]
            weights = (val_log_den_err.clamp_min(1e-3)).pow(-1)
            rho = pl_module.val_log_den[_idx]
            qz = m.encode(x)
            z = qz.rsample()
            s = m.prior.log_prob(z)
            scores = weights * (rho - s)

            WEIGHTS.append(weights.cpu())
            SCORES.append(scores.cpu())

        WEIGHTS = torch.cat(WEIGHTS, dim=0)
        SCORES = torch.cat(SCORES, dim=0)

        # save stats to a file
        out_dir = os.path.join(trainer.logger.log_dir, 'figures')
        os.makedirs(out_dir, exist_ok=True)

        # plot the histograms of the weights and scores
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(WEIGHTS.numpy(), bins=self.bins, alpha=0.6, density=True)
        ax.set_title("Weights Histogram")
        ax.set_xlabel("weights")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'weights_hist.png'), dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(SCORES.numpy(), bins=self.bins, alpha=0.6, density=True)
        ax.set_title("Weighted Scores Histogram")
        ax.set_xlabel("weights * (logpdf - prior_logpdf)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'train_scores_hist.png'), dpi=120)
        plt.close(fig)
        logger.info(f"[WeightsAndScoresHistogramCallback] saved weights and scores histograms to {out_dir}")

# =====================================
# Diagnistics Computation Callback:
#   - true logpdf v. model logpdf on val set
#   - true prior logpdf v. model prior logpdf on val set
#   - KL divergence between true GMM and learned prior
# =====================================
class OracleDiagnosticsCallback(pl.Callback):
    """
    Store the true dataspace log-density and prior log-density in additional model buffers.
    """

    def __init__(self,bins: int = 60,
                 range_percentiles=(1, 99)):
        super().__init__()
        self.bins = bins
        self.range_percentiles = range_percentiles

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: Model):
        dm = trainer.datamodule
        if  _find_rotgmm(dm.val_dataset) is None:
            warnings.warn("[OracleDiagnosticsCallback] callback is only applicable to GMMDataset; skipping.")
            return
        logging.info("[OracleDiagnosticsCallback] Computing oracle log-densities in data and prior space...")

        # ---- VAL ----
        funcs, root_train = get_oracle_funcs(dm.val_dataset)
        oracle_logpdf = funcs.get("oracle_logpdf", None)
        oracle_logpdf_2d = funcs.get("oracle_logpdf_2d", None)
        dl = dm.val_dataloader(shuffle=False)
        m = pl_module.module

        TRUE_LOGPDF, TRUE_LOGPDF2D, LOGPDF, LOGPDF2D = [], [], [], []
        if oracle_logpdf_2d is not None:
            device = next(pl_module.parameters()).device
            for (x,_), _idx in dl:
                x = x.to(device)
                x = x.view(x.size(0), -1)
                
                qz, _ = m.forward(x) 
                z = qz.rsample()  ##TODO: use qz.loc instead?
                logpdf = m.elbo(x)
                true_logpdf = oracle_logpdf(x).cpu()
                logpdf2d = m.prior.log_prob(z).cpu()
                true_logpdf2d = oracle_logpdf_2d(x).cpu()
                # true_gmm_logpdf_2d_z = root_train.mixture.log_prob(z).cpu()
                
                TRUE_LOGPDF.append(true_logpdf)
                TRUE_LOGPDF2D.append(true_logpdf2d)
                LOGPDF.append(logpdf.cpu())
                LOGPDF2D.append(logpdf2d)

                
            TRUE_LOGPDF = torch.cat(TRUE_LOGPDF, dim=0)
            TRUE_LOGPDF2D = torch.cat(TRUE_LOGPDF2D, dim=0)
            LOGPDF = torch.cat(LOGPDF, dim=0)
            LOGPDF2D = torch.cat(LOGPDF2D, dim=0)

        # Compute useful statistcs/diagnostics/plots
        ## stats
        stats = {}
        from scipy.stats import ks_2samp, wasserstein_distance
        if TRUE_LOGPDF.numel() > 0:
            stats['true_logpdf_mean'] = TRUE_LOGPDF.mean().item()
            stats['true_logpdf_std'] = TRUE_LOGPDF.std().item()
            stats['logpdf_mean'] = LOGPDF.mean().item()
            stats['logpdf_std'] = LOGPDF.std().item()
            stats['logpdf_corr'] = torch.corrcoef(torch.stack([TRUE_LOGPDF, LOGPDF]))[0,1].item()
            stats['logpdf_ks'] = ks_2samp(TRUE_LOGPDF.cpu().numpy(), LOGPDF.cpu().numpy()).statistic
            stats['logpdf_wass'] = wasserstein_distance(TRUE_LOGPDF.cpu().numpy(), LOGPDF.cpu().numpy())

        if TRUE_LOGPDF2D.numel() > 0:
            stats['true_logpdf2d_mean'] = TRUE_LOGPDF2D.mean().item()
            stats['true_logpdf2d_std'] = TRUE_LOGPDF2D.std().item()
            stats['logpdf2d_mean'] = LOGPDF2D.mean().item()
            stats['logpdf2d_std'] = LOGPDF2D.std().item()
            stats['logpdf2d_corr'] = torch.corrcoef(torch.stack([TRUE_LOGPDF2D, LOGPDF2D]))[0,1].item()
            stats['logpdf2d_ks'] = ks_2samp(TRUE_LOGPDF2D.cpu().numpy(), LOGPDF2D.cpu().numpy()).statistic
            stats['logpdf2d_wass'] = wasserstein_distance(TRUE_LOGPDF2D.cpu().numpy(), LOGPDF2D.cpu().numpy())

        ## KL between true GMM and learned prior
        if isinstance(m.prior, GMMPrior):
            gt_prior = root_train.mixture
            kl_gt_p = kl_divergence_mixture_mixture(gt_prior, m.prior._mixture(), num_samples=10000)
            kl_p_gt = kl_divergence_mixture_mixture(m.prior._mixture(), gt_prior, num_samples=10000)
            stats['prior_kl_reverse'] = kl_p_gt.item()
            stats['prior_kl'] = kl_gt_p.item()
       
        
        # save stats to a file
        out_dir = os.path.join(trainer.logger.log_dir, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        stats_file = os.path.join(out_dir, 'diagnostics.txt')
        with open(stats_file, 'w') as f:
            for k,v in stats.items():
                f.write(f'{k}: {v}\n')
        logger.info(f"[OracleDiagnosticsCallback] saved diagnostics stats to {stats_file}")
        
        # plot the histograms of the logpdfs
        out_dir = os.path.join(trainer.logger.log_dir, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        if TRUE_LOGPDF.numel() > 0:
            fig = self._make_figs(TRUE_LOGPDF.cpu().numpy(), LOGPDF.cpu().numpy())
            fig.savefig(os.path.join(out_dir, 'val_logpdf_hist.png'), dpi=120)
            plt.close(fig)
        if TRUE_LOGPDF2D.numel() > 0:
            fig = self._make_figs(TRUE_LOGPDF2D.cpu().numpy(), LOGPDF2D.cpu().numpy())
            fig.savefig(os.path.join(out_dir, 'val_logpdf2d_hist.png'), dpi=120)
            plt.close(fig)

            # scatter plot of true v. model logpdf
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(TRUE_LOGPDF2D.cpu().numpy(), LOGPDF2D.cpu().numpy(), s=5, alpha=0.5)
            lims = [min(TRUE_LOGPDF2D.min(), LOGPDF2D.min()).item(), max(TRUE_LOGPDF2D.max(), LOGPDF2D.max()).item()]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel('true prior logpdf'); ax.set_ylabel('model prior logpdf')
            ax.set_title('Val prior logpdf (2D latents)')
            fig.savefig(os.path.join(out_dir, 'val_logpdf2d_scatter.png'), dpi=120)
            plt.close(fig)
        logger.info(f"[OracleDiagnosticsCallback] saved diagnostics plots to {out_dir}")

    def _make_figs(self, d_ext_all, s_prior_all):
        # --- common histogram ranges (robust) ---
        p_lo, p_hi = self.range_percentiles
        # raw comparison range uses external vs  prior
        lo_raw = np.nanpercentile(np.concatenate([d_ext_all, s_prior_all]), p_lo)
        hi_raw = np.nanpercentile(np.concatenate([d_ext_all, s_prior_all]), p_hi)
       

        # --- RAW plot ---
        fig1, ax1 = plt.subplots(figsize=(7,4))
        ax1.hist(d_ext_all, bins=self.bins, range=(lo_raw, hi_raw), alpha=0.6, density=True, label="external (val)")
        ax1.hist(s_prior_all, bins=self.bins, range=(lo_raw, hi_raw), alpha=0.6, density=True, label="prior log p(z)")
        ax1.set_title("Densities (raw space)")
        ax1.set_xlabel("logpdf")
        ax1.set_ylabel("density")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', frameon=True)
        fig1.tight_layout()

        return fig1
# =====================================
# Aggregated Posterior Plot Callback
# =====================================
class AggregatedPosteriorPlotCallback(pl.Callback):
    """
    On fit end, plot the aggregated variational posterior q_agg(z)
    for 2D latents by collecting {mu_q(x)} (optionally samples) on the
    validation set and rendering a KDE heatmap + optional overlays:
      - GMMPrior means (if present)
      - prior samples (sparse) for reference
    Saves: figures/agg_posterior_kde.png
    """
    def __init__(self, samples_per_x: int = 1, grid_res: int = 200, pct_range=(1, 99), every_n_epochs: int = 5):
        super().__init__()
        self.S = samples_per_x
        self.grid_res = grid_res
        self.pct_range = pct_range
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def _get_dens(self, trainer: pl.Trainer, m: VAE):
        device = next(m.parameters()).device
        dl = trainer.datamodule.val_dataloader(shuffle=False)

        Zs = []
        for (x,_), _idx in dl:
            x = x.to(device)
            qz = m.encode(x)
            if self.S <= 1:
                Zs.append(qz.loc.detach().cpu())
            else:
                z = qz.rsample((self.S,))   # (S,B,2)
                Zs.append(z.reshape(-1, 2).detach().cpu())
        if not Zs:
            return
        Z = torch.cat(Zs, dim=0).numpy()  # (N,2)

        # grid limits by robust percentiles
        lo = np.percentile(Z, self.pct_range[0], axis=0)
        hi = np.percentile(Z, self.pct_range[1], axis=0)
        pad = 0.05 * (hi - lo)
        lo -= pad; hi += pad

        # evaluate KDE if available
        Xg = np.linspace(lo[0], hi[0], self.grid_res)
        Yg = np.linspace(lo[1], hi[1], self.grid_res)
        XX, YY = np.meshgrid(Xg, Yg)
        grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

        dens = None
        try:
            from sklearn.neighbors import KernelDensity
            # Bandwidth heuristic: Scott's rule for 2D
            n = max(1, Z.shape[0]); d = 2
            std = Z.std(axis=0).mean()
            bw = std * (n ** (-1.0/(d+4)))
            kde = KernelDensity(kernel='gaussian', bandwidth=max(bw, 1e-3)).fit(Z)
            dens = np.exp(kde.score_samples(grid)).reshape(self.grid_res, self.grid_res)
        except Exception:
            # simple 2D histogram as fallback
            H, xedges, yedges = np.histogram2d(Z[:,0], Z[:,1], bins=self.grid_res, range=[[lo[0],hi[0]],[lo[1],hi[1]]], density=True)
            dens = H.T  # align with imshow
            XX, YY = np.meshgrid((xedges[:-1]+xedges[1:])/2.0, (yedges[:-1]+yedges[1:])/2.0)

        return dens, lo, hi, Z
    
    def _make_fig(self, dens, lo, hi, Z, pl_module):
        # figure
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(dens, origin='lower', extent=[lo[0], hi[0], lo[1], hi[1]], aspect='auto')
        ax.set_title('Aggregated variational posterior $q_\\mathrm{agg}(z)$')
        ax.set_xlabel('z₁'); ax.set_ylabel('z₂')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('density')

        # overlay sparse scatter of Z to show support
        idx = np.random.choice(Z.shape[0], size=min(2000, Z.shape[0]), replace=False)
        ax.scatter(Z[idx,0], Z[idx,1], s=3, alpha=0.25, linewidths=0)

        # overlay prior means if GMMPrior
        prior = getattr(pl_module, 'prior', None)
        if isinstance(prior, GMMPrior):
            MU = prior.loc.detach().cpu().numpy()
            ax.scatter(MU[:,0], MU[:,1], s=80, marker='x', linewidths=1.5, label='GMM means', c='k')
            ax.legend(loc='best', frameon=True)

        fig.tight_layout()
        return fig
        
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Model):
        if (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        m = pl_module.module
        if getattr(m, 'n_latent', None) != 2:
            warnings.warn("[AggregatedPosteriorPlotCallback] only applicable to 2D latents; skipping.")
            return
        logger.info("[AggregatedPosteriorPlotCallback] Computing aggregated posterior plot...")
        dens, lo, hi, Z = self._get_dens(trainer, m)
        # figure
        fig = self._make_fig(dens, lo, hi, Z, m)

        out_dir = os.path.join(getattr(trainer.logger, 'log_dir', '.'), 'figures', 'epoch_plots')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f'agg_posterior_kde_epoch_{trainer.current_epoch:03d}.png'), dpi=180)
        plt.close(fig)

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        m = pl_module.module
        if getattr(m, 'n_latent', None) != 2:
            warnings.warn("[AggregatedPosteriorPlotCallback] only applicable to 2D latents; skipping.")
            return
        logger.info("[AggregatedPosteriorPlotCallback] Computing aggregated posterior plot...")
        dens, lo, hi, Z = self._get_dens(trainer, m)
        # figure
        fig = self._make_fig(dens, lo, hi, Z, m)

        out_dir = os.path.join(getattr(trainer.logger, 'log_dir', '.'), 'figures')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'agg_posterior_kde.png'), dpi=180)
        plt.close(fig)

# =====================================
# Ground Truth Mixture Plot Callback
# =====================================
class GroundTruthMixturePlotCallback(pl.Callback):
    """On fit start, plot the ground-truth 2D GMM (pre-rotation/padding).
    Saves: figures/gt_mixture.png
    """
    def __init__(self, grid_res: int = 400, levels=12):
        super().__init__()
        self.grid_res = grid_res
        self.levels = levels

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("[GroundTruthMixturePlotCallback] Plotting ground-truth mixture...")
        dm = trainer.datamodule
        root = _find_rotgmm(dm.train_dataset)
        if root is None:
            warnings.warn("[GroundTruthMixturePlotCallback] callback is only applicable to GMMDataset; skipping.")
            return

        mus2 = np.asarray(root._loc)
        scale2 = np.asarray(root._scale)
        covs2 = np.stack([np.diag((scale2[k]** 2)) for k in range(mus2.shape[0])], axis=0)
        pi = np.asarray(torch.softmax(root._logits, dim=0).cpu())
        
        # grid covering 3 std around all centers
        all_mu = np.vstack(mus2)
        lo = all_mu.min(axis=0) - 4.0
        hi = all_mu.max(axis=0) + 4.0
        x1 = np.linspace(lo[0], hi[0], self.grid_res)
        x2 = np.linspace(lo[1], hi[1], self.grid_res)
        X1, X2 = np.meshgrid(x1, x2)
        G = np.stack([X1.ravel(), X2.ravel()], axis=1)
        # mixture density
        dens = np.zeros(G.shape[0])
        for k in range(len(pi)):
            mu = mus2[k]
            cov = covs2[k]
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            diff = G - mu
            quad = np.einsum('ni,ij,nj->n', diff, inv, diff)
            dens += pi[k] * np.exp(-0.5*quad) / (2*np.pi*np.sqrt(det))
        dens = dens.reshape(self.grid_res, self.grid_res)
        # plot
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X1, X2, dens, levels=self.levels)
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label='density')
        ax.scatter(all_mu[:,0], all_mu[:,1], c='k', s=40, marker='x', label='true means')
        ax.set_title('Ground-truth 2D mixture (pre-rotation)')
        ax.set_xlabel('u₁'); ax.set_ylabel('u₂')
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        out_dir = os.path.join(getattr(trainer.logger, 'log_dir', '.'), 'figures')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'gt_mixture.png'), dpi=180)
        plt.close(fig)

# =====================================
# Populate Val Density For Eval Callback (kNN proxy if needed, used to plot density histograms even if density_reg=False)
# =====================================
class PopulateValDensityForEval(pl.Callback):
    """Ensure val_log_den buffers exist even when density_reg=False (e.g., std VAE).
    - If dataset is RotatedGMMDataset: use oracle log-pdf.
    - Else: compute a kNN log-density proxy on the validation set.
    """
    def __init__(self, pca_n_components: int = 40):
        super().__init__()
        self.pca_n_components = pca_n_components

    def _collect(self, dl) -> Tuple[np.ndarray, torch.Tensor]:
        X, I = [], []
        for (x, _y), idx in dl:
            X.append(x.view(x.size(0), -1))
            I.append(idx)
        X = torch.cat(X, dim=0)
        I = torch.cat(I, dim=0)
        return X, I

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("[PopulateValDensityForEval] Ensuring val_log_den buffers exist...")
        # If already present, do nothing
        if getattr(pl_module, 'val_log_den', None) is not None:
            logger.info("[PopulateValDensityForEval] val_log_den buffers already exist.")
            return
        dm = trainer.datamodule
        dl_val = dm.val_dataloader(shuffle=False)
        Xval, Ival = self._collect(dl_val)
        device = pl_module.device
        # Case 1: oracle
        funcs_dict, root = get_oracle_funcs(dm.val_dataset)
        if root is not None:
            oracle_logpdf = funcs_dict.get("oracle_logpdf", None)
            log_den = oracle_logpdf(Xval)
            err = np.full_like(log_den, 1e-3, dtype=np.float32)
        else:
            # Case 2: DPA PROXY
            from ..model import _apply_pca, _dpa_log_density
            X_flat = Xval.reshape(Xval.shape[0], -1).cpu().numpy()

            X_emb = _apply_pca(X_flat, n_components=self.pca_n_components, seed=42)

            emb_t = torch.tensor(X_emb, dtype=torch.float32, device=device)
            dense_u = torch.empty_like(emb_t)
            dense_u[Ival] = emb_t  # reorder to dataset order

            dpa_output = _dpa_log_density(X_emb, Zpar=1.96)
            log_den = dpa_output['log_den']
            err = dpa_output['log_den_err']

        # install buffers (dense ordering)
        log_den_t = torch.tensor(log_den, dtype=torch.float32, device=device)
        err_t = torch.tensor(err, dtype=torch.float32, device=device)
        dense_ld = torch.empty_like(log_den_t)
        dense_er = torch.empty_like(err_t)
        dense_ld[Ival] = log_den_t
        dense_er[Ival] = err_t
        pl_module.register_buffer('val_log_den', dense_ld)
        pl_module.register_buffer('val_log_den_err', torch.clamp(dense_er, min=1e-6))
        pl_module.register_buffer('val_indexes', Ival.to(device))

