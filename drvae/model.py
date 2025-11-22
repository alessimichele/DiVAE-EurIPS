# drvae/model.py
import pytorch_lightning as pl
from .modules import VAE
import torch
from typing import Dict, Any, Union, List, Optional
import numpy as np
import warnings
import logging
import dadac
import os
from .modules.aligners import FlowAligner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model(pl.LightningModule):
    def __init__(self,
                 module_config: Dict, 
                 density_reg: bool=True, 
                 density_eval: bool=True,
                 external_density_config: Optional[Dict] = None,
                 optimizer_config: Optional[Dict] = None,
                 kl_weight_config: Optional[Dict] = None, 
                 reg_weight_config: Optional[Dict] = None,
                 **kwargs):
        super().__init__()
        ##TODO: flow mod
        self.automatic_optimization = False

        # ---- Validate + build module ----
        _validate_config(module_config, required_keys=['module', 'args'])
        self.module = self._set_module(module_config)

         # ---- Runtime flags ----
        self.density_reg = bool(density_reg)
        self.density_eval = bool(density_eval)

        # ---- Config blocks with safe defaults ----
        self.kl_weight_cfg = kl_weight_config or {"mode": "fixed", "value": 1.0}
        self.reg_weight_cfg = reg_weight_config or {"mode": "fixed", "value": 1.0}
        self.optimizer_config = optimizer_config or {"optimizer": "adam", "lr": 1e-3}

        
        external_density_config = external_density_config or {"method": "dpa"}
        _validate_config(external_density_config, required_keys=["method"])
        self.external_density_config = dict(external_density_config)

        self.sync_dist = True

        # ---- SAVE ONLY SERIALIZABLE HPARAMS ----
        # (module_config should already be strings/numbers/lists, since your enc/dec use string ids)
        hparams_to_save = {
            "module_config": _jsonify(module_config),  # ensure lists not tuples, etc.
            "density_reg": self.density_reg,
            "density_eval": self.density_eval,
            "optimizer_config": dict(self.optimizer_config),
            "kl_weight_config": dict(self.kl_weight_cfg),
            "reg_weight_config": dict(self.reg_weight_cfg),
        }
        if self.density_reg:
            hparams_to_save["external_density_config"] = dict(self.external_density_config)

        # Anything non-serializable (like tensors) is NOT included here.
        self.save_hyperparameters(hparams_to_save)

        self.log_den: torch.nn.UninitializedBuffer
        self.log_den_err: torch.nn.UninitializedBuffer
        self.indexes: torch.nn.UninitializedBuffer
        self.train_u: torch.nn.UninitializedBuffer

        self.val_log_den: torch.nn.UninitializedBuffer
        self.val_log_den_err: torch.nn.UninitializedBuffer
        self.val_indexes: torch.nn.UninitializedBuffer
        self.val_u: torch.nn.UninitializedBuffer

    def _set_module(self, module_config: Dict) -> VAE:
        module = module_config['module']
        args = module_config.get('args', {})
        if module == 'vae':
            _validate_config(args, required_keys=['input_shape', 'n_latent', 
                                                  'encoder_config', 
                                                  'decoder_config', 'likelihood', 'gaussian_std', 
                                                  'prior_config', 
                                                  'aligner_config'])
            return VAE(**args)
        else:
            raise ValueError("Unsupported module type")

    def on_fit_start(self) -> None:
        logging.info("Starting training...")
        if self.density_reg:
            self._compute_external_density()
        else:
            self.register_buffer('log_den', None)
            self.register_buffer('log_den_err', None)
            self.register_buffer('indexes', None)
        if self.density_eval:
                self._compute_external_density(split="val") 

    @property
    def current_kl_weight(self) -> float:
        cfg = self.kl_weight_cfg
        if cfg["mode"] == "fixed":
            return float(cfg["value"])
        if cfg["mode"] == "epoch_warmup":
            return _interp_warmup(self.current_epoch, self.global_step,
                                  n_epochs=cfg.get("epochs", 10),
                                  w_min=cfg.get("min", 0.0),
                                  w_max=cfg.get("max", 1.0))
        if cfg["mode"] == "step_warmup":
            return _interp_warmup(self.current_epoch, self.global_step,
                                  n_steps=cfg.get("steps", 1000),
                                  w_min=cfg.get("min", 0.0),
                                  w_max=cfg.get("max", 1.0))
        raise ValueError(f"Unknown kl_weight mode {cfg}")

    @property
    def current_reg_weight(self) -> float:
        cfg = self.reg_weight_cfg
        if cfg["mode"] == "fixed":
            return float(cfg["value"])
        if cfg["mode"] == "epoch_warmup":
            return _interp_warmup(self.current_epoch, self.global_step,
                                  n_epochs=cfg.get("epochs", self.trainer.max_epochs // 2),
                                  w_min=cfg.get("min", 0.05),
                                  w_max=cfg.get("max", 0.5))
        if cfg["mode"] == "step_warmup":
            return _interp_warmup(self.current_epoch, self.global_step,
                                  n_steps=cfg.get("steps", 2000),
                                  w_min=cfg.get("min", 0.05),
                                  w_max=cfg.get("max", 0.5))
        raise ValueError(f"Unknown reg_weight mode {cfg}")

    def training_step(self, batch, batch_idx):
        if self.density_reg and isinstance(self.module.aligner, FlowAligner):
            optimizer, flow_optimizer = self.optimizers()
        else:
            optimizer = self.optimizers()

        out, indexes = batch
        x, y = out if isinstance(out, (tuple, list)) else (out, None)

        log_den = self.log_den[indexes] if self.density_reg else None
        log_den_err = self.log_den_err[indexes] if self.density_reg else None
        u_batch = self.train_u[indexes] if self.density_reg else None
        
        
        if self.density_reg and isinstance(self.module.aligner, FlowAligner):
            self.module.aligner.flow.base_dist = self.module.prior
            flow_loss = self.module.aligner.flow.forward(u_batch)
            flow_optimizer.zero_grad()
            flow_loss.backward()
            flow_optimizer.step()
            self.log("train_flow_loss", flow_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=self.sync_dist)

        loss_cls = self.module.loss_function(x, 
                                             log_den=log_den, 
                                             log_den_err=log_den_err,
                                             kl_weight=self.current_kl_weight,
                                             reg_weight=self.current_reg_weight,
                                             u_batch=u_batch,
                                            )
        loss = loss_cls.total_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.log_dict(loss_cls.log_dict("train_"), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out, indexes = batch
        x, y = out if isinstance(out, (tuple, list)) else (out, None)
        
        # (A) train-like val loss (no reg term)
        loss_cls = self.module.loss_function(x)
        self.log_dict(loss_cls.log_dict("val_"), on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss_cls.total_loss

    def configure_optimizers(self):
        if self.density_reg and isinstance(self.module.aligner, FlowAligner):
            flow_optimizer = torch.optim.Adam(self.module.aligner.flow.parameters(), lr=1e-3)
            # get the flow parameters so that they are not included into the vae optimizer
            flow_params_id = set(map(id, self.module.aligner.flow.parameters()))
            optimizer = torch.optim.Adam(
                (p for p in self.module.parameters() if id(p) not in flow_params_id),
                lr=self.optimizer_config.get("lr", 1e-3)
            )
            return optimizer, flow_optimizer
        else:
            optimizer = torch.optim.Adam(self.module.parameters(), lr=self.optimizer_config.get("lr", 1e-3))
            return optimizer
            
    
    def _compute_external_density(self, split="train") -> None:
        ed_method = self.external_density_config.get('method', 'dpa')
        ed_pca_components = self.external_density_config.get('pca_components', self.module.n_latent)
        logging.info("Computing external density estimates using method: %s", ed_method)

        dataloader = (self.trainer.datamodule.train_dataloader(shuffle=False)
                      if split=="train" else
                      self.trainer.datamodule.val_dataloader(shuffle=False))
        X, Y, I = _get_data_from_dataloader(dataloader)
        X_flat = X.reshape(X.shape[0], -1).cpu().numpy()

        seed = self.external_density_config.get('pca_seed', 42)

        if ed_pca_components is not None and ed_pca_components > 0 and ed_pca_components < X_flat.shape[1]:
            X_emb = _apply_pca(X_flat, n_components=ed_pca_components, seed=seed)
        else:
            X_emb = X_flat

        emb_t = torch.tensor(X_emb, dtype=torch.float32, device=self.device)
        dense_u = torch.empty_like(emb_t)
        dense_u[I] = emb_t  # reorder to dataset order

        if ed_method == "dpa":
            dpa_zpar = self.external_density_config.get('dpa_zpar', 1.96)
            dpa_output = _dpa_log_density(X_emb, Zpar=dpa_zpar)
            log_den = dpa_output['log_den']
            log_den_err = dpa_output['log_den_err']

        elif ed_method == "knn":
            ed_knn_k = self.external_density_config.get('knn_k', 50)
            ed_knn_metric = self.external_density_config.get('knn_metric', 'euclidean')
            knn_output = _knn_log_density(X_emb, k=ed_knn_k, metric=ed_knn_metric)
            log_den = knn_output['log_den']
            log_den_err = knn_output['log_den_err']
        
        else:
            raise ValueError(f"Unknown external density method: {ed_method}")
        
        log_den_t = torch.tensor(log_den, dtype=torch.float32, device=self.device)
        log_den_err_t = torch.tensor(log_den_err, dtype=torch.float32, device=self.device)
        dense_log_den     = torch.empty_like(log_den_t)
        dense_log_den_err = torch.empty_like(log_den_err_t)
        dense_log_den[I]     = log_den_t # dataset order
        dense_log_den_err[I] = log_den_err_t # dataset order
        dense_log_den_err = torch.clamp(dense_log_den_err, min=1e-6) # avoid 0 err


        if split == "train":
            u = dense_u.to(self.device)
            u_mean = u.mean(dim=0, keepdim=True)
            u_std = u.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
            u_std = u_std
            u_z = (u - u_mean) / u_std

            self.register_buffer('log_den', dense_log_den.to(self.device))
            self.register_buffer('log_den_err', dense_log_den_err.to(self.device))
            self.register_buffer('indexes', I.to(self.device))
            self.register_buffer('train_u', u_z)   # (N_train, L)
            self.register_buffer('train_u_mean', u_mean)
            self.register_buffer('train_u_std', u_std)
        else:
            u = dense_u.to(self.device)
            u_mean = u.mean(dim=0, keepdim=True)
            u_std = u.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
            u_z = (u - u_mean) / u_std
            
            logdir = self.trainer.logger.log_dir
            savedir = os.path.join(logdir, "figures")
            os.makedirs(savedir, exist_ok=True)
            _plot_features_hist(u_z, save_path=os.path.join(savedir, f"val_u_hist_component.png"))
            
            self.register_buffer('val_log_den', dense_log_den.to(self.device))
            self.register_buffer('val_log_den_err', dense_log_den_err.to(self.device))
            self.register_buffer('val_indexes', I.to(self.device))
            self.register_buffer('val_u', u_z)     # (N_val, L)
            self.register_buffer('val_u_mean', u_mean)
            self.register_buffer('val_u_std', u_std)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint.get('state_dict', {})
        self._register_buffer_if_missing('log_den', state_dict)
        self._register_buffer_if_missing('log_den_err', state_dict)
        self._register_buffer_if_missing('indexes', state_dict)
        self._register_buffer_if_missing('val_log_den', state_dict)
        self._register_buffer_if_missing('val_log_den_err', state_dict)
        self._register_buffer_if_missing('val_indexes', state_dict)
        self._register_buffer_if_missing('train_u', state_dict)
        self._register_buffer_if_missing('train_u_mean', state_dict)
        self._register_buffer_if_missing('train_u_std', state_dict)
        self._register_buffer_if_missing('val_u', state_dict)
        self._register_buffer_if_missing('val_u_mean', state_dict)
        self._register_buffer_if_missing('val_u_std', state_dict)
        return checkpoint

    def _register_buffer_if_missing(self, buffer_name: str, state_dict: dict):
        """Helper method to register a buffer if it is missing."""
        if buffer_name not in self._buffers and buffer_name in state_dict:
            self.register_buffer(buffer_name, state_dict[buffer_name])
            logger.info(f"Registered buffer {buffer_name} from state_dict.")

def _plot_features_hist(u_z, save_path: str) -> None:
    """
    This function accept from 2 up to 5 dimensions and the save_path where to save the plots.
    It will generate one plot with subplots for each dimension (from 2 to 5, it depends on the input).
    In each subplot there will be the hist for that dimension.
    """
    import matplotlib.pyplot as plt
    def _plot_hist_dim(ax, data, dim):
        ax.hist(data.cpu().numpy(), bins=50)
        ax.set_title(f"Histogram of component {dim+1}")
    n_dims = u_z.shape[1]
    if n_dims < 2 or n_dims > 5:
        logging.warning("This function supports from 2 up to 5 dimensions.")
        return
    fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4))
    for i in range(n_dims):
        _plot_hist_dim(axs[i], u_z[:, i], i)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def _to_numpy_float64(X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.cpu().detach().numpy().astype('float64')
    elif isinstance(X, np.ndarray):
        return X.astype('float64')
    else:
        raise ValueError(f"Unsupported type {type(X)}")

def _get_data_from_dataloader(dataloader):
    X = []
    Y = []
    I = []
    for batch in dataloader:
        out, indexes = batch
        x, y = out
        X.append(x)
        Y.append(y)
        I.append(indexes)
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    I = torch.cat(I, dim=0)
    assert I.shape[0] == X.shape[0] == Y.shape[0]
    return X, Y, I

def _apply_pca(X, n_components=50, seed=42):
    from sklearn.decomposition import PCA
    logging.info("Applying PCA to reduce to %d dimensions", n_components)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(X)
    
def _dpa_log_density(X: Union[np.ndarray, torch.Tensor], 
                     Zpar: float = 1.96) -> Dict:
    """
    Compute an external log-density proxy via DPA.
    """
    logging.info("Computing DPA log-density with Zpar=%.2f", Zpar)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        data = _to_numpy_float64(X)
        dpa = dadac.Data(data, verbose=False)
        dpa.compute_distances(k=200)
        dpa.compute_id_2NN()
        dpa.compute_density_PAk()
        dpa.compute_clustering_ADP(Z=Zpar, halo=False)

        labels = np.unique(dpa.cluster_assignment)
        cluster_indices = [np.flatnonzero(dpa.cluster_assignment == lab).tolist()
                   for lab in labels]
          
        return {
            'curr_id': dpa.id,
            'cluster_centers': dpa.cluster_centers,
            'cluster_indices': cluster_indices,
            'cluster_assignment': dpa.cluster_assignment,
            'log_den': dpa.log_den,
            'log_den_err': dpa.log_den_err,
            'log_den_bord': dpa.log_den_bord,
            'log_den_bord_err': dpa.log_den_bord_err
        }

def _knn_log_density(
    X: Union[np.ndarray, torch.Tensor],
    k: int = 50,
    metric: str = "euclidean",
    ) -> Dict[str, np.ndarray]:
    """
    Compute an external log-density proxy via kNN distances (method-agnostic vs DPA).
    Returns dict with keys: 'log_den', 'log_den_err', 'idx' (indices mapping to original X).
    """
    logging.info("Computing kNN log-density with k=%d and metric=%s", k, metric)
    from sklearn.neighbors import NearestNeighbors
    X = _to_numpy_float64(X)
    k_eff = int(min(k + 1, max(2, X.shape[0]))) # +1 to drop self
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, n_neighbors=k_eff)
    # drop self-distance at position 0
    d_k = dists[:, 1:] # (N, k)
    eps = 1e-8
    # log-density proxy: -log(mean distance)
    mean_d = d_k.mean(axis=1)
    log_den = -np.log(mean_d + eps)
    # Uncertainty proxy: std of log distances / sqrt(k)
    logd = np.log(d_k + eps)
    log_den_err = logd.std(axis=1) / np.sqrt(d_k.shape[1])
    return {"log_den": log_den, "log_den_err": log_den_err}

def _validate_config(config: Dict, required_keys: List[str]) -> None:
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must contain '{key}' key.")

def _interp_warmup(epoch: int, step: int, *, n_epochs=None, n_steps=None, w_min=0.0, w_max=1.0):
    if w_min > w_max:
        raise ValueError("w_min > w_max")
    if n_epochs:
        if epoch < n_epochs:
            t = epoch / max(1, n_epochs)
            return w_min + (w_max - w_min) * t
    elif n_steps:
        if step < n_steps:
            t = step / max(1, n_steps)
            return w_min + (w_max - w_min) * t
    return w_max

def _jsonify(x):
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]  # convert tuple->list
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # fallback to str for odd scalars/enums
    return str(x)