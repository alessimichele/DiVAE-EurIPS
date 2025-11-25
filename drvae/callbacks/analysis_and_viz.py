# drvae/callbacks/analysis_and_viz.py
import os
import math
import json
from typing import Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import logging
import time
from ..model import Model

# ================================================================
# Shared utilities
# ================================================================
def _module_device_and_dtype(mod: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    """Best-effort to infer device/dtype from a module with no .device attr."""
    for it in (mod.parameters(), mod.buffers()):
        try:
            t = next(it)
            return t.device, t.dtype
        except StopIteration:
            pass
    return torch.device("cpu"), torch.float32


def _log_root(trainer: pl.Trainer) -> str:
    # CSVLogger: trainer.logger.log_dir → ".../logs/<name>/version_X"
    return getattr(trainer.logger, "log_dir", ".")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _fig_dir(trainer: pl.Trainer) -> str:
    return _ensure_dir(os.path.join(_log_root(trainer), "figures"))


def _subfig_dir(trainer: pl.Trainer, sub: str) -> str:
    # e.g., "reconstructions" or "pseudoinputs"
    return _ensure_dir(os.path.join(_fig_dir(trainer), sub))


def _dist_to_image_tensor(px, image_shape: Tuple[int, int, int], likelihood: str) -> torch.Tensor:
    """
    Convert decode() distribution to an image tensor in [0,1] with shape (B,C,H,W).
    - Bernoulli: sigmoid(logits) if available; otherwise .probs
    - Gaussian:  mean = loc (clamped)
    """
    C, H, W = image_shape
    if likelihood == "bernoulli":
        if hasattr(px, "logits"):
            xhat = torch.sigmoid(px.logits)
        else:
            probs = getattr(px, "probs", None)
            if probs is None:
                raise RuntimeError("Bernoulli decode without logits/probs.")
            xhat = probs
    elif likelihood == "gaussian":
        xhat = px.loc
    else:
        raise ValueError(f"Unknown likelihood {likelihood}")
    return xhat.view(-1, C, H, W).clamp(0, 1)


@torch.no_grad()
def get_prior_centers_z_and_x(vae) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Unified access to prior 'centers' in both latent and input space.

    Returns:
      z_centers: (M,D) or None
      x_centers: (M,C,H,W) in [0,1] or None
    Cases:
      - DPAPrior/VampPrior: pseudo_inputs available → encode to z, return pseudo_inputs as x
      - GMMPrior: use prior.loc as z centers; decode means to x images
    """
    C, H, W = vae.input_shape
    device, _ = _module_device_and_dtype(vae)

    # DPAPrior / VampPrior
    if hasattr(vae.prior, "pseudo_inputs") and vae.prior.pseudo_inputs is not None:
        x = vae.prior.pseudo_inputs()
        if x.numel() == 0:
            return None, None
        x = x.to(device)
        qz = vae.encode(x)
        zc = qz.loc.detach()
        return zc, x.detach().clamp(0, 1)

    # GMMPrior
    if hasattr(vae.prior, "loc") and vae.prior.loc is not None:
        zc = vae.prior.loc.detach()
        px = vae.decode(zc.to(device))
        xc = _dist_to_image_tensor(px, (C, H, W), vae.likelihood)
        return zc, xc.detach()

    return None, None


def _get_prior_centers_as_images(vae) -> Optional[torch.Tensor]:
    """Convenience for plotting just the images of centers."""
    _, xc = get_prior_centers_z_and_x(vae)
    return xc


# ================================================================
# Evaluation: ELBO and IWAE
# ================================================================
from ..modules import VAE  # avoid circular import
@torch.no_grad()
def compute_elbo_over_loader(vae: VAE, dataloader) -> torch.Tensor:
    """
    Per-example ELBO over a dataloader:
      ELBO(x) ≈ log p(x|μ_q) − KL(q||p)
    (Deterministic mean-of-q proxy for the expectation.)
    """
    logging.info("Computing ELBO over dataloader...")
    vae.eval()
    device, _ = _module_device_and_dtype(vae)

    elbos = []
    for batch, _ in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        elbo = vae.elbo(x)  # (B,)
        elbos.append(elbo.detach().cpu())

    return torch.cat(elbos, dim=0)


@torch.no_grad()
def compute_iwae_over_loader(vae, dataloader, S: int = 50) -> torch.Tensor:
    """
    IWAE estimate of log p(x) with S importance samples.
    Returns a 1D tensor with per-example log-likelihood estimates.
    """
    logging.info(f"Computing IWAE with S={S} samples per example over dataloader with {len(dataloader.dataset)} examples.")
    vae.eval()
    device, _ = _module_device_and_dtype(vae)

    parts = []
    for batch, _ in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        B_actual = x.size(0)

        x_batch_rep = x.unsqueeze(1).expand(B_actual, S, *x.shape[1:]).contiguous().view(B_actual * S, *x.shape[1:]) # (B*S,C,H,W)
        
        elbo = vae.elbo(x_batch_rep)  # (B*S,)
        elbo = elbo.view(B_actual, S)  # (B,S)
        log_iw = elbo - math.log(S)  # (B,S)
        log_iwae = torch.logsumexp(log_iw, dim=1)  # (B,)
        parts.append(log_iwae.detach().cpu())   

    return torch.cat(parts, dim=0)  # (N,)

@torch.no_grad()
def compute_prior_logpdf_over_loader(vae: VAE, dataloader, S: int = 50) -> torch.Tensor:
    """
    Per-example log p(z) under the prior, where z = μ_q(x).
    Returns a 1D tensor with per-example log p(z) estimates.
    """
    logging.info(f"Computing prior log p(z) with S={S} samples per example over dataloader with {len(dataloader.dataset)} examples.")
    vae.eval()
    device, _ = _module_device_and_dtype(vae)

    logps = []
    for batch, _ in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        B_actual = x.size(0)

        x_batch_rep = x.unsqueeze(1).expand(B_actual, S, *x.shape[1:]).contiguous().view(B_actual * S, *x.shape[1:]) # (B*S,C,H,W)


        qz = vae.encode(x_batch_rep)  # (B*S,D)
        z = qz.loc  # (B*S,D)
        logp = vae.prior.log_prob(z)  # (B*S,)
        logp = logp.view(B_actual, S)  # (B,S)
        logp_mean = torch.logsumexp(logp - math.log(S), dim=1) # (B,)
        ##TODO: or should be torch.logsumexp(logp, dim=1) - np.log(S)???
        logps.append(logp_mean.detach().cpu())
    return torch.cat(logps, dim=0)  # (N,)


# ================================================================
# Callback 1: Reconstructions / Pseudo-inputs / Final samples
# ================================================================
class ImageLoggingCallback(pl.Callback):
    """
    Saves:
      - figures/reconstructions/epoch_XXX.png  (two-row grid: originals top, reconstructions bottom)
      - figures/pseudoinputs/epoch_XXX.png     (VAMP/DPAPrior pseudo-inputs or GMM decoded centers)
      - figures/samples_final.png              (prior samples at fit end)
    """
    def __init__(self,
                 n_reconstructions: int = 16,
                 recon_every_n_epochs: int = 1,
                 save_pseudo_every_n_epochs: int = 1):
        super().__init__()
        self.n_reconstructions = n_reconstructions
        self.recon_every = recon_every_n_epochs
        self.pseudo_every = save_pseudo_every_n_epochs
        self._fixed_x = None

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        vae = pl_module.module
        device, _ = _module_device_and_dtype(vae)

        # Freeze a small validation batch to reconstruct across epochs
        dl = trainer.datamodule.val_dataloader(shuffle=False)
        (x, _), _idx = next(iter(dl))
        self._fixed_x = x[: self.n_reconstructions].to(device)

        # Ensure folders
        _subfig_dir(trainer, "reconstructions")
        _subfig_dir(trainer, "pseudoinputs")

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        vae = pl_module.module
        device, _ = _module_device_and_dtype(vae)

        # Reconstructions
        if (trainer.current_epoch % self.recon_every) == 0 and self._fixed_x is not None:
            x = self._fixed_x.to(device)
            qz, _ = vae.forward(x)
            z = qz.loc  # deterministic for stable visuals
            px = vae.decode(z)
            xhat = _dist_to_image_tensor(px, vae.input_shape, vae.likelihood)

            # two-row grid: originals on top, reconstructions bottom
            grid = make_grid(torch.cat([x.cpu(), xhat.cpu()], dim=0),
                             nrow=self.n_reconstructions, pad_value=0.95)
            out_path = os.path.join(_subfig_dir(trainer, "reconstructions"),
                                    f"epoch_{trainer.current_epoch:03d}.png")
            save_image(grid, out_path)

        # Pseudo-inputs / centers
        if (trainer.current_epoch % self.pseudo_every) == 0:
            imgs = _get_prior_centers_as_images(vae)
            if imgs is not None and imgs.numel() > 0:
                M = imgs.size(0)
                nrow = min(16, int(math.ceil(math.sqrt(M))))
                grid = make_grid(imgs.cpu(), nrow=nrow, pad_value=0.95)
                out_path = os.path.join(_subfig_dir(trainer, "pseudoinputs"),
                                        f"epoch_{trainer.current_epoch:03d}.png")
                save_image(grid, out_path)

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        vae = pl_module.module
        device, _ = _module_device_and_dtype(vae)
        S = max(64, self.n_reconstructions)
        z = vae.prior.sample(S, device=device)
        px = vae.decode(z)
        imgs = _dist_to_image_tensor(px, vae.input_shape, vae.likelihood)
        grid = make_grid(imgs.cpu(), nrow=int(math.sqrt(S)), pad_value=0.95)
        out_path = os.path.join(_fig_dir(trainer), "samples_final.png")
        save_image(grid, out_path)

# =====================================
# Jacobian Callback: 
#   - for the Flow aligner, keep track of the log-determinant of the Jacobian during training
# =====================================
from ..modules.flows import FlowAdapter 
class JacobianLoggingCallback(pl.Callback):
    
    def __init__(self):
        super().__init__()
        self.jacobian_logs_per_batch = []
        self.jacobian_logs = []

    def on_train_batch_end(self, trainer, pl_module: Model, outputs, batch, batch_idx):
        vae = pl_module.module
        if hasattr(vae, 'aligner') and hasattr(vae.aligner, 'flow'):
            _, indexes = batch
            u_batch = pl_module.train_u[indexes] 
            _, logdet_inv = vae.aligner.flow.inverse_flow(u_batch)  # (B,D), (B,)

            mean_logdet = logdet_inv.mean().item()
            self.jacobian_logs_per_batch.append(mean_logdet)

    def on_train_epoch_end(self, trainer, pl_module: Model):
        if self.jacobian_logs_per_batch:
            epoch_mean = np.mean(self.jacobian_logs_per_batch)
            epoch_std = np.std(self.jacobian_logs_per_batch)
            self.jacobian_logs.append((epoch_mean, epoch_std))
            logging.info(f"[JacobianLoggingCallback] Epoch {trainer.current_epoch}: mean log-det-Jacobian = {epoch_mean:.3f} ± {epoch_std:.3f}")
            self.jacobian_logs_per_batch = []

        if hasattr(pl_module.module, 'aligner') and hasattr(pl_module.module.aligner, 'flow'):
            save_dir = os.path.join(_fig_dir(trainer), "epoch_plots")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"flow_epoch_{trainer.current_epoch:03d}.png")
            self.plot_the_flow(pl_module.module.aligner.flow, save_path=save_path)

    def plot_the_flow(self, flow_model: FlowAdapter, num_samples=1000, save_path=None):
        import matplotlib.pyplot as plt
        n_flows = len(flow_model.flows)+1
        fig, axes = plt.subplots(1, n_flows, figsize=(n_flows*5, n_flows), sharex=False, sharey=False)
        if n_flows == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            u = flow_model.sample_step(i, num_samples)
            u = u.detach().cpu().numpy()
            if u.shape[1] > 2:
                # apply PCA to reduce to 2D for visualization
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                u = pca.fit_transform(u)
            x = u[:, 0]
            y = u[:, 1]
            x_lim = (x.min() - 1, x.max() + 1)
            y_lim = (y.min() - 1, y.max() + 1)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.scatter(x, y, alpha=0.5)            
            # enforce equal aspect ratio so circles remain circles
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Samples after {i} flows')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def on_fit_end(self, trainer, pl_module: Model):
        # visualize the determinant of the Jacobian over epochs
        if self.jacobian_logs:
            epochs = np.arange(len(self.jacobian_logs))
            means = [log[0] for log in self.jacobian_logs]
            stds = [log[1] for log in self.jacobian_logs]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(epochs, means, yerr=stds, fmt='-o', ecolor='lightgray', elinewidth=3, capsize=0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Mean log-det-Jacobian')
            ax.set_title('Log-determinant of the Jacobian over Epochs')
            ax.grid(True, alpha=0.3)

            fig_dir = _fig_dir(trainer)
            out_path = os.path.join(fig_dir, "jacobian_logdet_over_epochs.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            logging.info(f"[JacobianLoggingCallback] Saved Jacobian log-det plot to {out_path}")


# ================================================================
# Callback: Training time measurements
# ================================================================
class TrainingTimeCallback(pl.Callback):
    """
    Measures per-training-step duration and the initialization time at fit start.

    Behavior:
      - on_fit_start: record `init_time` (ISO string) and save to `figures/timing.json`
      - on_train_batch_start: note batch start time
      - on_train_batch_end: record batch duration (seconds) in current epoch buffer
      - on_train_epoch_end: compute mean duration for the epoch, append to `epoch_means_batch`, save JSON to `figures/timing.json`
    """
    def __init__(self):
        super().__init__()
        self._batch_start = None
        self.batch_durations = []  # batch durations for current epoch
        self.epoch_means_batch = [] 
        self.epoch_means_total = []
        self.init_time = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: Model) -> None:
        init_time_start = time.time()
        pl_module.on_fit_start()  # call model's on_fit_start to ensure any setup is done before timing
        init_time_end = time.time()
        self.data = {
            "init_time": float(init_time_end - init_time_start),
        }
        
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx, dataloader_idx=0) -> None:
        # mark the start of the batch
        self._batch_start = time.time()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self._batch_start is None:
            return
        dur = time.time() - self._batch_start
        self.batch_durations.append(dur)
        # reset batch start to avoid accidental reuse
        self._batch_start = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # compute mean duration for this epoch and store it
        if self.batch_durations:
            mean_d = float(np.mean(self.batch_durations))
            sum_d = float(np.sum(self.batch_durations))
            self.epoch_means_batch.append(mean_d)
            self.epoch_means_total.append(sum_d)
            logging.info(f"[TrainingTimeCallback] Epoch {trainer.current_epoch}: mean step time = {mean_d:.6f} s")
        else:
            # no batches seen this epoch (possible if zero-step training)
            self.epoch_means_batch.append(None)
            self.epoch_means_total.append(None)
        # reset per-epoch buffer
        self.batch_durations = []

    def on_fit_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        # write timing.json with updated epoch_means_batch
        out_dir = _fig_dir(trainer)
        out_path = os.path.join(out_dir, "timing.json")
        self.data['epoch_means_batch'] = self.epoch_means_batch
        self.data['epoch_means_total'] = self.epoch_means_total
        
        try:
            with open(out_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            logging.exception("[TrainingTimeCallback] Failed to write timing.json on fit start")


     

# ================================================================
# Callback 2: End-of-training ELBO & IWAE (JSON report)
# ================================================================
class EndOfTrainingEvalCallback(pl.Callback):
    """
    At training end:
      - compute ELBO and IWAE(S) on the validation set
      - save JSON: figures/metrics_eval.json
        {
          "elbo_mean": ...,
          "elbo_std": ...,
          "iwae_S": S,
          "iwae_mean": ...,
          "iwae_std": ...
        }
    """
    def __init__(self, iwae_S: int = 50):
        super().__init__()
        self.iwae_S = iwae_S

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        vae = pl_module.module
        vae.eval()
        val_loader = trainer.datamodule.val_dataloader(shuffle=False)

        elbo = compute_elbo_over_loader(vae, val_loader)  # (N,)
        iwae = compute_iwae_over_loader(vae, val_loader, S=self.iwae_S)  # (N,)
        prior_logpdf = compute_prior_logpdf_over_loader(vae, val_loader, S=self.iwae_S)  # (N,)

        report: Dict[str, float] = {
            "elbo_mean": float(elbo.mean().item()),
            "elbo_std": float(elbo.std(unbiased=False).item()),
            "iwae_S": float(self.iwae_S),
            "iwae_mean": float(iwae.mean().item()),
            "iwae_std": float(iwae.std(unbiased=False).item()),
            "prior_logpdf_mean": float(prior_logpdf.mean().item()),
            "prior_logpdf_std": float(prior_logpdf.std(unbiased=False).item()),
        }
        out_path = os.path.join(_fig_dir(trainer), "metrics_eval.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"[Eval] ELBO mean={report['elbo_mean']:.3f}±{report['elbo_std']:.3f} | "
              f"IWAE(S={self.iwae_S}) mean={report['iwae_mean']:.3f}±{report['iwae_std']:.3f} | "
              f"prior logpdf mean={report['prior_logpdf_mean']:.3f}±{report['prior_logpdf_std']:.3f}")


# ================================================================
# Callback 3: Metrics plot from metrics.csv  (fixed: train/val merge)
# ================================================================
class MetricsPlotCallback(pl.Callback):
    def __init__(self, metrics_csv_name: str = "metrics.csv"):
        super().__init__()
        self.metrics_csv_name = metrics_csv_name

    def on_fit_end(self, trainer, pl_module):
        logging.info("[MetricsPlotCallback] Generating metrics plots...")
        log_dir = getattr(trainer.logger, "log_dir", ".")
        csv = os.path.join(log_dir, self.metrics_csv_name)
        if not os.path.isfile(csv):
            logging.warning(f"[MetricsPlotCallback] No CSV at {csv}")
            return

        df = pd.read_csv(csv)
        if "epoch" not in df.columns:
            logging.warning(f"[MetricsPlotCallback] CSV has no epoch column.")
            return

        # normalize epoch/step
        df = df.dropna(subset=["epoch"]).copy()
        df["epoch"] = df["epoch"].astype(int)
        if "step" not in df.columns:
            df["step"] = df.groupby("epoch").cumcount()

        # Build epoch-level frames separately, then merge
        def last_rows_with(prefix: str) -> pd.DataFrame:
            cols = [c for c in df.columns if c.startswith(prefix)] + ["epoch", "step"]
            sub = df[cols].dropna(how="all", subset=[c for c in cols if c.startswith(prefix)]).copy()
            if sub.empty:
                return pd.DataFrame(columns=cols)
            sub = sub.sort_values(["epoch", "step"]).groupby("epoch", as_index=False).tail(1)
            return sub.drop(columns=["step"])

        tr = last_rows_with("train_")
        va = last_rows_with("val_")
        df_epoch = pd.merge(tr, va, on="epoch", how="outer").sort_values("epoch")

        fig_dir = _fig_dir(trainer)
        total_png = os.path.join(fig_dir, "metrics_total.png")
        comps_png = os.path.join(fig_dir, "metrics_components.png")
        # ---- Total loss (train/val) ----
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        if "train_total_loss" in df_epoch.columns:
            ax1.plot(df_epoch["epoch"], df_epoch["train_total_loss"], marker="o", label="train_total_loss")
        if "val_total_loss" in df_epoch.columns:
            ax1.plot(df_epoch["epoch"], df_epoch["val_total_loss"], marker="o", label="val_total_loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("total loss")
        ax1.grid(True, alpha=0.3); ax1.legend()
        fig1.tight_layout()
        fig1.savefig(total_png, dpi=160)
        plt.close(fig1)

        # ---- Components (3 subplots) ----
        fig2, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        ax_rec, ax_kl, ax_reg = axes

        # Reconstruction
        if "train_reconstruction_loss" in df_epoch.columns:
            ax_rec.plot(df_epoch["epoch"], df_epoch["train_reconstruction_loss"], marker="o", label="train")
        if "val_reconstruction_loss" in df_epoch.columns:
            ax_rec.plot(df_epoch["epoch"], df_epoch["val_reconstruction_loss"], marker="o", label="val")
        ax_rec.set_title("Reconstruction"); ax_rec.set_xlabel("epoch"); ax_rec.set_ylabel("value")
        ax_rec.grid(True, alpha=0.3); ax_rec.legend(fontsize=8)

        # KL
        if "train_kl_divergence_loss" in df_epoch.columns:
            ax_kl.plot(df_epoch["epoch"], df_epoch["train_kl_divergence_loss"], marker="o", label="train")
        if "val_kl_divergence_loss" in df_epoch.columns:
            ax_kl.plot(df_epoch["epoch"], df_epoch["val_kl_divergence_loss"], marker="o", label="val")
        ax_kl.set_title("KL divergence"); ax_kl.set_xlabel("epoch")
        ax_kl.grid(True, alpha=0.3); ax_kl.legend(fontsize=8)

        # Regularization (train-only + optional val calibration)
        did_any = False
        if "train_regularization_loss" in df_epoch.columns:
            ax_reg.plot(df_epoch["epoch"], df_epoch["train_regularization_loss"], marker="o", label="train (reg)")
            did_any = True
        if "val_calibration_loss" in df_epoch.columns:
            ax_reg.plot(df_epoch["epoch"], df_epoch["val_calibration_loss"], marker="o", label="val (calibration)")
            did_any = True
        ax_reg.set_title("Regularization"); ax_reg.set_xlabel("epoch")
        ax_reg.grid(True, alpha=0.3)
        if did_any: ax_reg.legend(fontsize=8)

        fig2.tight_layout()
        fig2.savefig(comps_png, dpi=160)
        plt.close(fig2)
        logging.info(f"[MetricsPlotCallback] Metrics figures saved to {fig_dir}/metrics_*.png")

# ================================================================
# 4) UMAP plot of latent space (+ prior centers)  (REPLACEMENT)
# ================================================================
class UMAPPlotCallback(pl.Callback):
    def __init__(self,
                 max_val_points: Optional[int] = 5000,
                 point_size: float = 4.0,
                 use_umap_transform: bool = True):
        """
        use_umap_transform=True → fit UMAP on data points, then project centers with transform() (recommended).
        If False, we concatenate centers to data and fit UMAP jointly (slightly different positioning).
        """
        super().__init__()
        self.max_val_points = max_val_points
        self.point_size = point_size
        self.use_umap_transform = use_umap_transform

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module: Model):
        logging.info("[UMAPPlotCallback] Generating UMAP plot of latent μ(x)...")
        try:
            import umap
        except Exception:
            logging.warning("[UMAPPlotCallback] umap-learn not installed; skipping.")
            return

        m = pl_module.module
        device, _ = _module_device_and_dtype(m)

        # Gather latent μ(x) + labels
        dl = trainer.datamodule.val_dataloader(shuffle=False)
        Zs, Ys = [], []
        n_seen = 0
        for (x, y), _idx in dl:
            x = x.to(device)
            qz = m.encode(x)
            mu = qz.loc
            Zs.append(mu.cpu().numpy())
            if y is None:
                Ys.append(np.zeros(mu.shape[0], dtype=int))
            else:
                Ys.append(y.cpu().numpy())
            n_seen += mu.shape[0]
            if self.max_val_points and n_seen >= self.max_val_points:
                break

        if not Zs:
            logging.warning("[UMAPPlotCallback] No val data found.")
            return

        Z = np.vstack(Zs)              # (N, D)
        y_raw = np.concatenate(Ys)     # (N,)

        # Categorical labels for stable legend
        y_cat = pd.Categorical(y_raw)
        y_codes = y_cat.codes          # integers 0..K-1 in label order
        y_names = list(map(str, y_cat.categories))

        # Prior centers in z-space
        zc, _xc = get_prior_centers_z_and_x(m)
        zc_np = zc.detach().cpu().numpy() if zc is not None else None

        reducer = umap.UMAP(n_components=2, random_state=42)

        if self.use_umap_transform and zc_np is not None and zc_np.shape[0] > 0:
            Z2 = reducer.fit_transform(Z)
            zc2 = reducer.transform(zc_np)
        else:
            # Fit on concatenated (data + centers) so centers are embedded jointly
            if zc_np is not None and zc_np.shape[0] > 0:
                Z_joint = np.concatenate([Z, zc_np], axis=0)
                Z2_joint = reducer.fit_transform(Z_joint)
                Z2 = Z2_joint[:Z.shape[0]]
                zc2 = Z2_joint[Z.shape[0]:]
            else:
                Z2 = reducer.fit_transform(Z)
                zc2 = None

        # Plot
        fig_dir = _fig_dir(trainer)
        out_path = os.path.join(fig_dir, "umap_latent.png")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter by category to get a clean legend
        for code, name in enumerate(y_names):
            mask = (y_codes == code)
            if not np.any(mask):
                continue
            ax.scatter(
                Z2[mask, 0], Z2[mask, 1],
                s=self.point_size, alpha=0.7, linewidths=0, label=name
            )

        if zc2 is not None:
            ax.scatter(zc2[:, 0], zc2[:, 1], c="k", s=80, marker="x", label="prior centers")

        ax.set_title("UMAP of latent μ(x)")
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=2, fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        logging.info(f"[UMAPPlotCallback] UMAP figure saved to {out_path}")

# ================================================================
# 5) Density histograms (external vs prior vs aligned-prior)
# ================================================================

class DensityHistogramCallback(pl.Callback):

    def __init__(self, max_val_batches: int = None,
                 bins: int = 60,
                 range_percentiles=(1, 99),
                 every_n_epochs: int = 5):
        super().__init__()
        self.max_val_batches = max_val_batches
        self.bins = bins
        self.range_percentiles = range_percentiles
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def _get_den(self, trainer: pl.Trainer, pl_module: Model):
        from ..modules.aligners import DirectAligner, FlowAligner
        if not getattr(pl_module, "density_eval", False):
            logging.warning("[DensityHistogramCallback] density_eval=False; skipping.")
            return
        if not isinstance(pl_module.module.aligner, (DirectAligner, FlowAligner)):
            logging.warning(f"[DensityHistogramCallback] aligner {pl_module.module.aligner} is not supported; skipping.")
            return
        
        val_log_den = getattr(pl_module, "val_log_den", None)
        val_err     = getattr(pl_module, "val_log_den_err", None)
        if val_log_den is None:
            logging.warning("[DensityHistogramCallback] No val_log_den buffer; skipping.")
            return

        m: VAE = pl_module.module
        device, _ = _module_device_and_dtype(m)
        dl = trainer.datamodule.val_dataloader(shuffle=False)

        d_ext_all, s_prior_all = [], []
        n_batches = 0

        for (x, _y), idx in dl:
            x   = x.to(device)
            idx = idx.to(device)

            # external densities for this batch (val set aligned via 'indexes' buffer order)
            d_ext = val_log_den[idx].detach().cpu().numpy()

            # q(z|x), one sample z ~ q for prior scores
            qz = m.encode(x)
            z  = qz.rsample()
            s_prior = m.prior.log_prob(z).detach().cpu().numpy()
            
            d_ext_all.append(d_ext)
            s_prior_all.append(s_prior)

            n_batches += 1
            if self.max_val_batches and n_batches >= self.max_val_batches:
                break

        if not d_ext_all:
            logging.warning("[DensityHistogramCallback] No validation batches found.")
            return

        d_ext_all = np.concatenate(d_ext_all, axis=0)
        s_prior_all = np.concatenate(s_prior_all, axis=0)
        
        return d_ext_all, s_prior_all

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
        ax1.legend()
        fig1.tight_layout()

        return fig1

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        d_ext_all, s_prior_all = self._get_den(trainer, pl_module)

        fig1 = self._make_figs(d_ext_all, s_prior_all)
        out_dir = os.path.join(getattr(trainer.logger, 'log_dir', '.'), 'figures', 'epoch_plots')
        os.makedirs(out_dir, exist_ok=True)
        fig1.savefig(os.path.join(out_dir, f"density_hist_epoch{trainer.current_epoch:03d}.png"), dpi=160)
        plt.close(fig1)
   

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        d_ext_all, s_prior_all = self._get_den(trainer, pl_module)

        fig1 = self._make_figs(d_ext_all, s_prior_all)
        log_dir = _fig_dir(trainer)
        fig1.savefig(os.path.join(log_dir, "density_hist_raw.png"), dpi=160)
        plt.close(fig1)

        # --- quick stats (optionally KS/Wasserstein if scipy present) ---
        stats = {
            "raw": {
                "external_mean": float(np.nanmean(d_ext_all)),
                "external_std": float(np.nanstd(d_ext_all)),
                "prior_mean": float(np.nanmean(s_prior_all)),
                "prior_std": float(np.nanstd(s_prior_all)),
            },
        }
        try:
            from scipy.stats import ks_2samp, wasserstein_distance
            stats["raw"]["ks"] = float(ks_2samp(d_ext_all, s_prior_all).statistic)
            stats["raw"]["wasserstein"] = float(wasserstein_distance(d_ext_all, s_prior_all))
        except Exception:
            pass

        with open(os.path.join(log_dir, "density_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        logging.info("[DensityHistogramCallback] Saved density_hist_raw.png, density_stats.json")
        

    






































"""
def _probit(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * u.clamp(1e-6, 1 - 1e-6) - 1.0)

# ---------- alignment helpers ----------
@torch.no_grad()
def align_prior_scores(vae, s_prior_np: np.ndarray, qz: torch.distributions.Normal = None):
    ###
    Map prior scores s = log p(z) into the *external-density space* as used during training
    of the aligner. Returns (aligned_np, space_tag, used_fallback).

    space_tag:
      - "raw"  -> comparable to raw external densities
      - "d_hat" (PIT) -> compare against external densities transformed to probit(PIT)
    ###
    aligner = getattr(vae, "aligner", None)
    if aligner is None:
        return s_prior_np, "raw", True

    device, dtype = _module_device_and_dtype(vae)
    s = torch.as_tensor(s_prior_np, device=device, dtype=dtype)

    # Avoid circular imports in user projects: import types here
    from drvae.modules.aligners import (
        DirectAligner, ZScoreAligner, PITAligner, SimpleABAligner, GMMAwareAligner
    )

    # 0) Direct → identity
    if isinstance(aligner, DirectAligner):
        return s_prior_np, "raw", False

    # 1) ZScore: pred = std_d^{-1}( g( std_s(s) ) )
    if isinstance(aligner, ZScoreAligner):
        s_hat = aligner.std_s.normalize(s)
        pred_hat = aligner.g(s_hat)
        pred = aligner.std_d.denormalize(pred_hat)
        return pred.detach().cpu().numpy(), "raw", False

    # 2) PIT: pred_hat = g( Φ^{-1}( CDF_s(s) ) ), lives in d_hat space
    if isinstance(aligner, PITAligner):
        us = aligner.cdf_s.cdf(s).clamp(1e-6, 1 - 1e-6)
        s_hat = _probit(us)
        pred_hat = aligner.g(s_hat)
        return pred_hat.detach().cpu().numpy(), "d_hat", False

    # Below need per-sample features → require qz
    if qz is None:
        return s_prior_np, "raw", True

    mu = qz.loc
    logvar = 2.0 * torch.log(qz.scale + 1e-8)

    # 3) SimpleAB: trained a(x)*d + b(x) ≈ s  → invert: d ≈ (s - b)/a
    if isinstance(aligner, SimpleABAligner):
        a, b = aligner._ab_from_feats(mu, logvar)  # (B,), (B,)
        a = a.clamp_min(1e-4)
        d_aligned = (s - b) / a
        return d_aligned.detach().cpu().numpy(), "raw", False
        

    # 4) GMMAware: pred = a(H,μ) * g(s) + b(H,μ)
    if isinstance(aligner, GMMAwareAligner):
        # sample z for responsibilities context (consistent with training)
        z_ctx = qz.rsample() if qz is not None else mu
        if hasattr(vae.prior, "responsibilities"):
            r = vae.prior.responsibilities(z_ctx)   # (B,K)
            H = (-(r.clamp_min(1e-8) * r.clamp_min(1e-8).log()).sum(-1, keepdim=True))
        else:
            H = torch.zeros(mu.shape[0], 1, device=mu.device, dtype=mu.dtype)
        ctx = torch.cat([H, mu.pow(2).mean(-1, keepdim=True)], dim=-1)
        a_raw, b = aligner.ctx_head(ctx).chunk(2, dim=-1)
        a = 1.0 + 0.1 * torch.tanh(a_raw)
        gs = aligner.g(s)
        pred = a.squeeze(-1) * gs + b.squeeze(-1)
        return pred.detach().cpu().numpy(), "raw", False

    # Fallback
    return s_prior_np, "raw", True

@torch.no_grad()
def transform_external_for_compare(vae, d_ext_np: np.ndarray, space_tag: str) -> np.ndarray:
    
    #If aligned scores live in a transformed space (e.g., PIT's 'd_hat'),
    #transform the external densities to that space for a fair comparison.
    
    if space_tag != "d_hat":
        return d_ext_np

    aligner = getattr(vae, "aligner", None)
    try:
        from drvae.modules.aligners import PITAligner
    except Exception:
        PITAligner = type("PITAligner", (), {})  # harmless dummy

    if not isinstance(aligner, PITAligner):
        return d_ext_np

    device, dtype = _module_device_and_dtype(vae)
    d = torch.as_tensor(d_ext_np, device=device, dtype=dtype)
    ud = aligner.cdf_d.cdf(d).clamp(1e-6, 1 - 1e-6)
    d_hat = _probit(ud)
    return d_hat.detach().cpu().numpy()

# ---------- the callback ----------
class DensityHistogramCallback(pl.Callback):
    ###
    Density Histogram Analysis
    --------------------------

    This callback produces two figures and one JSON summary to assess the
    alignment between external densities (e.g. DPA/kNN log-densities) and
    the model’s prior scores.

    Saved under: <log_dir>/figures/

    Figures
    -------

    1. figures/density_hist_raw.png
       Overlayed histograms in the original (raw) score space:
         - External densities (val): the precomputed DPA/kNN log-density
           for each validation sample.
         - Prior log-densities (raw): one sample z ~ q(z|x) per validation
           example, scored under the current prior log p(z).
       → Shows the pre-alignment mismatch between external densities and
         the model prior.

    2. figures/density_hist_aligned.png
       Overlayed histograms in the aligned comparison space:
         - External (transformed if needed):
             * PIT: d → d^ = Φ⁻¹(CDF_d(d))
             * ZScore/others: raw d
         - Aligned prior: prior scores s = log p(z) mapped through the
           learned aligner:
             * ZScore:  s → std_s(s) → g → denorm_d → d_pred
             * PIT:     s → CDF_s(s), Φ⁻¹ → g → d^_pred
             * SimpleAB (trained d→s): invert via d ≈ (s - b)/a
             * GMMAware: d_pred = a(ctx) * g(s) + b(ctx)
       → If alignment is working, these histograms should overlap much more
         closely than the raw ones.

    JSON Metrics
    ------------

    figures/density_stats.json
    Example structure:

    {
      "raw": {
        "external_mean": -21.68,
        "external_std": 3.66,
        "prior_mean": -26.28,
        "prior_std": 3.46,
        "ks": 0.6391,
        "wasserstein": 4.6037
      },
      "aligned": {
        "external_mean": -21.68,
        "external_std": 3.66,
        "aligned_prior_mean": -22.40,
        "aligned_prior_std": 4.49,
        "ks": 0.2183,
        "wasserstein": 1.0055
      }
    }

    Where:
      - means/stds: distribution moments
      - ks: Kolmogorov–Smirnov statistic (0 = perfect match)
      - wasserstein: Earth-Mover distance (lower = better alignment)

    Interpretation
    --------------
    Use density_hist_raw.png to see the scale mismatch before alignment,
    and density_hist_aligned.png to confirm whether the aligner has made
    prior scores comparable to external densities. Good alignment should
    shrink KS/Wasserstein and bring the two histograms into closer overlap.
    ###
    def __init__(self, max_val_batches: int = None, bins: int = 60, range_percentiles=(1, 99)):
        super().__init__()
        self.max_val_batches = max_val_batches
        self.bins = bins
        self.range_percentiles = range_percentiles

    @torch.no_grad()
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        log_dir = _fig_dir(trainer)

        # Access model & buffers
        if not getattr(pl_module, "density_eval", False):
            logging.warning("[DensityHistogramCallback] density_eval=False; skipping.")
            return
        m = pl_module.module
        device, _ = _module_device_and_dtype(m)

        # Need validation external densities computed by your Model
        val_log_den = getattr(pl_module, "val_log_den", None)
        val_err     = getattr(pl_module, "val_log_den_err", None)
        if val_log_den is None:
            logging.warning("[DensityHistogramCallback] No val_log_den buffer; skipping.")
            return

        dl = trainer.datamodule.val_dataloader(shuffle=False)

        d_ext_all, s_prior_all, s_aligned_all = [], [], []
        n_batches = 0

        for (x, _y), idx in dl:
            x   = x.to(device)
            idx = idx.to(device)

            # external densities for this batch (val set aligned via 'indexes' buffer order)
            d_ext = val_log_den[idx].detach().cpu().numpy()

            # q(z|x), one sample z ~ q for prior scores
            qz = m.encode(x)
            z  = qz.rsample()
            s_prior = m.prior.log_prob(z).detach().cpu().numpy()

            # aligned prior scores (PLUS a tag for space)
            s_aligned, space_tag, _ = align_prior_scores(m, s_prior, qz=qz)
            d_for_plot = transform_external_for_compare(m, d_ext, space_tag)

            d_ext_all.append(d_for_plot)
            s_prior_all.append(s_prior)
            s_aligned_all.append(s_aligned)

            n_batches += 1
            if self.max_val_batches and n_batches >= self.max_val_batches:
                break

        if not d_ext_all:
            logging.warning("[DensityHistogramCallback] No validation batches found.")
            return

        d_ext_all = np.concatenate(d_ext_all, axis=0)
        s_prior_all = np.concatenate(s_prior_all, axis=0)
        s_aligned_all = np.concatenate(s_aligned_all, axis=0)

        # --- common histogram ranges (robust) ---
        p_lo, p_hi = self.range_percentiles
        # raw comparison range uses raw external vs raw prior
        lo_raw = np.nanpercentile(np.concatenate([d_ext_all, s_prior_all]), p_lo)
        hi_raw = np.nanpercentile(np.concatenate([d_ext_all, s_prior_all]), p_hi)
        # aligned comparison range uses external (transformed if PIT) vs aligned
        lo_aln = np.nanpercentile(np.concatenate([d_ext_all, s_aligned_all]), p_lo)
        hi_aln = np.nanpercentile(np.concatenate([d_ext_all, s_aligned_all]), p_hi)

        # --- RAW plot ---
        fig1, ax1 = plt.subplots(figsize=(7,4))
        ax1.hist(d_ext_all, bins=self.bins, range=(lo_raw, hi_raw), alpha=0.6, density=True, label="external (val)")
        ax1.hist(s_prior_all, bins=self.bins, range=(lo_raw, hi_raw), alpha=0.6, density=True, label="prior log p(z) (raw)")
        ax1.set_title("Densities (raw space)")
        ax1.set_xlabel("score")
        ax1.set_ylabel("density")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(log_dir, "density_hist_raw.png"), dpi=160)
        plt.close(fig1)

        # --- ALIGNED plot ---
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.hist(d_ext_all, bins=self.bins, range=(lo_aln, hi_aln), alpha=0.6, density=True, label="external (transformed if needed)")
        ax2.hist(s_aligned_all, bins=self.bins, range=(lo_aln, hi_aln), alpha=0.6, density=True, label="aligned prior")
        ax2.set_title("Densities (aligned space)")
        ax2.set_xlabel("score")
        ax2.set_ylabel("density")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(log_dir, "density_hist_aligned.png"), dpi=160)
        plt.close(fig2)

        # --- quick stats (optionally KS/Wasserstein if scipy present) ---
        stats = {
            "raw": {
                "external_mean": float(np.nanmean(d_ext_all)),
                "external_std": float(np.nanstd(d_ext_all)),
                "prior_mean": float(np.nanmean(s_prior_all)),
                "prior_std": float(np.nanstd(s_prior_all)),
            },
            "aligned": {
                "external_mean": float(np.nanmean(d_ext_all)),
                "external_std": float(np.nanstd(d_ext_all)),
                "aligned_prior_mean": float(np.nanmean(s_aligned_all)),
                "aligned_prior_std": float(np.nanstd(s_aligned_all)),
            }
        }
        try:
            from scipy.stats import ks_2samp, wasserstein_distance
            stats["raw"]["ks"] = float(ks_2samp(d_ext_all, s_prior_all).statistic)
            stats["raw"]["wasserstein"] = float(wasserstein_distance(d_ext_all, s_prior_all))
            stats["aligned"]["ks"] = float(ks_2samp(d_ext_all, s_aligned_all).statistic)
            stats["aligned"]["wasserstein"] = float(wasserstein_distance(d_ext_all, s_aligned_all))
        except Exception:
            pass

        with open(os.path.join(log_dir, "density_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        logging.info("[DensityHistogramCallback] Saved density_hist_raw.png, density_hist_aligned.png and density_stats.json")
        """
