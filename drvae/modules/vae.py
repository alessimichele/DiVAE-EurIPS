 #drvae/modules/vae.py 

from .base import BaseVAE
import torch
from typing import Tuple, Dict, Any, Union, Optional, Literal, List
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Bernoulli
from ..loss import Loss
from .aligners import BaseAligner

class VAE(BaseVAE):
    def __init__(self,
                 input_shape: Tuple[int, ...] = (1, 28, 28),
                 n_latent: int = 10,
                 encoder_config: Dict = None,
                 decoder_config: Dict = None,
                 likelihood: Literal['bernoulli', 'gaussian'] = 'bernoulli',
                 gaussian_std: float = 0.1,
                 prior_config: Dict = None,
                 aligner_config: Dict = None,
                 ):
        super().__init__(input_shape, n_latent, encoder_config, decoder_config, likelihood, gaussian_std, prior_config)
        
        self.aligner = self._set_aligner(aligner_config)
        self.configs['aligner_config'] = self.aligner.get_hparams()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> Union[Tuple[Normal, Normal], Tuple[Normal, Bernoulli]]:
        qz_dist = self.encode(x)
        z = qz_dist.rsample()
        px_dist = self.decode(z)
        return qz_dist, px_dist
    
    def loss_function(self, x: torch.Tensor, **kwargs: Any) -> Loss:
        log_den = kwargs.get('log_den', None)
        log_den_err = kwargs.get('log_den_err', None)

        qz_dist, px_dist = self.forward(x)
        rec_loss = self._compute_reconstruction_loss(x, px_dist).mean()
        kl_div_loss = self.prior.kl(qz_dist).mean()
        reg_term = self._compute_regularization_loss(qz_dist, **kwargs) if log_den is not None and log_den_err is not None else None
        kl_weight = kwargs.get('kl_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 1.0)        
        return Loss(reconstruction=rec_loss, kl_div=kl_div_loss, reg_term=reg_term, kl_weight=kl_weight, reg_weight=reg_weight)

    @torch.no_grad()
    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        qz_dist, px_dist = self.forward(x)
        # per-example reconstruction loss (already sums over pixels, NOT mean)
        rec_loss_per = self._compute_reconstruction_loss(x, px_dist)   # (B,)
        # per-example KL
        kl_per = self.prior.kl(qz_dist)                                 # (B,)
        # ELBO = E_q[log p(x|z)] - KL = -(rec + kl) with your sign convention
        return -(rec_loss_per + kl_per)                                 # (B,)

    def _compute_regularization_loss(self, qz_dist: Normal, **kwargs) -> torch.Tensor:
        # We operate on s = log p(z) directly (no mc-avg needed for aligners; they’re robust)
        return self.aligner(qz_dist, self.prior, **kwargs)

    def _compute_reconstruction_loss(self, x: torch.Tensor, px_dist: Union[Normal, Bernoulli]) -> torch.Tensor:
        x = x.reshape(-1, int(np.prod(self.input_shape)))
        if self.likelihood == 'bernoulli':
            return F.binary_cross_entropy_with_logits(px_dist.logits, x, reduction='none').sum(dim=-1)
        elif self.likelihood == 'gaussian':
            return -px_dist.log_prob(x).sum(dim=-1)
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")

    def _set_aligner(self, aligner_config: Optional[Dict]) -> BaseAligner:
        r"""
        detach_encoder: bool = False for every aligner except DirectAligner (default: True).
        If False, gradients from the aligner loss will propagate back to the encoder.
        """
        if aligner_config is None:
            from .aligners import DirectAligner
            return DirectAligner(detach_encoder=True)
        
        _validate_config(aligner_config, required_keys=['aligner', 'args'])
        aligner = aligner_config['aligner']
        aligner_args: Dict = aligner_config.get('args', {})
        if aligner == 'direct':
            from .aligners import DirectAligner
            return DirectAligner(detach_encoder=aligner_args.get('detach_encoder', False))
        elif aligner == 'flow':
            from .aligners import FlowAligner
            return FlowAligner(
                n_latent=self.n_latent,
                hidden=aligner_args.get('hidden', 128),
                K=aligner_args.get('K', 4),
                detach_encoder=aligner_args.get('detach_encoder', True),
                huber_delta=aligner_args.get('huber_delta', 1.0),
                penalty=aligner_args.get('penalty', 1e-3),
                add_consistency=aligner_args.get('add_consistency', False),
                consistency_weight=aligner_args.get('consistency_weight', 0.0),
            )
        else:
            raise ValueError(f"Unknown aligner: {aligner}")

        

def _validate_config(config: Dict, required_keys: List[str]) -> None:
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must contain '{key}' key.")