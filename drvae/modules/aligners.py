# drvae/modules/aligners.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
from .prior import Prior
from .flows import FlowAdapter

class BaseAligner(nn.Module):
    def __init__(self, detach_encoder: bool = False, huber_delta: float = 1.0, penalty: float = 1e-3):
        super().__init__()
        self.detach_encoder = detach_encoder
        self.huber_delta = huber_delta
        self.penalty = penalty  # small identity regularization

    def _robust_rmse(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        # pred: (B,), target: (B,), weight: (B,)
        weight = weight.clamp_max(9.99999*1e5)
        if torch.all(weight == 9.99999*1e5):
            weight = torch.ones_like(weight)
        
        # runnato cosi per eurIPS
        h = F.huber_loss(pred, target, reduction='none', delta=self.huber_delta)
        return torch.sqrt((h * weight).mean())

    def get_hparams(self):
        keys = ['detach_encoder', 'huber_delta', 'penalty']
        return {k: getattr(self, k) for k in keys}


class DirectAligner(BaseAligner):
    """No alignment."""
    def __init__(self, detach_encoder: bool = False):
        super().__init__(detach_encoder=detach_encoder)

    def forward(self, qz: Normal, prior: Prior, **kwargs):
        log_den = kwargs.get('log_den', None)
        log_den_err = kwargs.get('log_den_err', None)
        if log_den is None or log_den_err is None:
            raise ValueError("DirectAligner.forward() requires log_den and log_den_err keyword arguments")
        
        z = qz.rsample()
        if self.detach_encoder:
            z = z.detach()
        s = prior.log_prob(z)                       # (B,)

        w = (log_den_err.clamp_min(1e-3)).pow(-1)
        reg = self._robust_rmse(s, log_den, w)

        return reg 
    
    def get_hparams(self):
        keys = ['detach_encoder']
        hpars = {k: getattr(self, k) for k in keys}
        hpars['aligner'] = 'direct'
        return hpars


class FlowAligner(BaseAligner):
    """
    Align external log-density in PCA space, \tilde d(u),
    with the model-implied log-density at u via
      s_flow(u) = log p_Z(h^{-1}(u)) + log|det J_{h^{-1}}(u)|
    where h is an invertible adapter between z-space and u-space.
    """
    def __init__(self, n_latent: int, hidden: int = 128, K: int = 4,
                 detach_encoder: bool = False, huber_delta: float = 1.0, penalty: float = 1e-3,
                 add_consistency: bool = False, consistency_weight: float = 0.0):
        super().__init__(detach_encoder=detach_encoder, huber_delta=huber_delta, penalty=penalty)
        self.flow = FlowAdapter(dim=n_latent, n_hidden=hidden, K=K)
        self.add_consistency = bool(add_consistency)
        self.consistency_weight = float(consistency_weight)

    def forward(self,
                qz: Normal,
                prior: Prior,
                **kwargs):
        """
        qz: Normal (B, D)
        prior: Prior on z
        log_den_u: external log-density \tilde d(u) for the batch (B,)
        log_den_u_err: its stderr (B,)
        u_batch: PCA coordinates for the same batch (B, D)  [D == n_latent]
        """
        if 'u_batch' not in kwargs:
            raise ValueError("FlowAligner.forward() requires u_batch keyword argument")
        u_batch = kwargs['u_batch']
        log_den = kwargs.get('log_den', None)
        log_den_err = kwargs.get('log_den_err', None)
        if log_den is None or log_den_err is None:
            raise ValueError("ProportionalAligner.forward() requires log_den and log_den_err keyword arguments")
        
        z = qz.rsample()
        _, logdet = self.flow.forward_flow(u_batch)  # u -> z (B, D), (B,)
       
        s_flow = prior.log_prob(z) + logdet    # (B,)
        # Robust Huber with precision weights
        w = (log_den_err.clamp_min(1e-3)).pow(-1)
        reg = self._robust_rmse(s_flow, log_den, w)
        return reg


    def get_hparams(self):
        return {
            "aligner": "flow",
            "detach_encoder": self.detach_encoder,
            "huber_delta": self.huber_delta,
            "penalty": self.penalty,
            "add_consistency": self.add_consistency,
            "consistency_weight": self.consistency_weight,
        }