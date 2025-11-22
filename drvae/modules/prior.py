# drvae/modules/prior.py
import torch
from torch import nn
from torch.distributions import Normal, Independent, Categorical, MixtureSameFamily
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional
import numpy as np

class Prior(nn.Module):
    """Base prior interface: implement log_prob and sample; default KL via MC."""
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Return log p(z) with shape (B,)."""
        raise NotImplementedError

    def sample(self, num_samples: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def kl(self, qz: Normal, n_mc: int = 50) -> torch.Tensor:
        """KL(qz || p) per sample (B,). MC default for general priors."""
        if n_mc == 1:
            z = qz.rsample()                      # (B, D)
            return (qz.log_prob(z).sum(-1) - self.log_prob(z))  # (B,)
        z = qz.rsample((n_mc,))                   # (n_mc, B, D)
        logq = qz.log_prob(z).sum(-1)             # (n_mc, B)
        logp = self.log_prob(z)                   # (n_mc, B) via broadcasting
        return (logq - logp).mean(0)              # (B,)

    def get_hparams(self) -> Dict:
        """Return a dictionary of hyperparameters for logging."""
        raise NotImplementedError
        

class StandardNormalPrior(Prior):
    def __init__(self, n_latent: int):
        super().__init__()
        self.n_latent = n_latent
        self.register_buffer("_loc",  torch.zeros(n_latent))
        self.register_buffer("_scale", torch.ones(n_latent))

    @property
    def dist(self) -> Independent:
        return Independent(Normal(self._loc, self._scale), 1)  # event=(D,)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(z)  # (B,)

    def sample(self, num_samples: int, device=None) -> torch.Tensor:
        d = self.dist if device is None else Independent(Normal(self._loc.to(device), self._scale.to(device)), 1)
        return d.rsample((num_samples,))  # (S, D)

    def kl(self, qz: Normal, n_mc: int = 1) -> torch.Tensor:
        # Analytic per dim, then sum → (B,)
        mu, std = qz.loc, qz.scale
        return 0.5 * (std.pow(2) + mu.pow(2) - 1 - 2*std.log()).sum(-1)
    
    def get_hparams(self) -> Dict:
        return {'prior': 'standard_normal', 'n_latent': self.n_latent}

class GMMPrior(Prior):
    def __init__(self, n_latent: int, n_components: int = 10, min_std: float = 0.05):
        super().__init__()
        self.n_latent = n_latent
        self.K = n_components
        self.min_std = min_std
        self.logits = nn.Parameter(torch.ones(self.K))           # (K,)
        self.loc = nn.Parameter(torch.randn(self.K, n_latent)*0.1)  # (K,D)
        self.s_raw = nn.Parameter(torch.ones(self.K, n_latent) * 0.307)  # softplus param

    def std(self):
        """
        Softplus does:
            softplus(x) = log(1 + exp(x)) ∈ (0, ∞)
        At the beginning, s_raw=0 → F.softplus(0)=log(2)=0.693 → std=0.693+min_std
        As s_raw→-∞, softplus(x)→0 → std→min_std
        As s_raw→+∞, softplus(x)~x → std~x+min_std

        To have at the beginning std=1.0, with min_std=0.05 we need s_raw to be initialized with ~0.307, so that
            softplus(0.307) = log(1 + exp(0.307)) = 0.95
        
        """
        return F.softplus(self.s_raw) + self.min_std             # (K,D)

    def _mixture(self) -> MixtureSameFamily:
        comp = Independent(Normal(self.loc, self.std()), 1)      # K comps, diag cov
        mix  = Categorical(logits=self.logits)
        return MixtureSameFamily(mix, comp)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self._mixture().log_prob(z)                       # (B,)

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        # r_bk ∝ π_k N(z_b ; μ_k, Σ_k)
        std  = self.std()                                        # (K,D)
        comp = Independent(Normal(self.loc, std), 1)             # (K)
        log_comp = comp.log_prob(z[:, None, :])                  # (B,K)
        log_mix  = F.log_softmax(self.logits, dim=-1)            # (K,)
        return F.softmax(log_comp + log_mix, dim=-1)             # (B,K)
    

    def sample(self, num_samples: int, device=None) -> torch.Tensor:
        z = self._mixture().sample((num_samples,))   # (S, D)
        return z if device is None else z.to(device)
    
    def get_hparams(self) -> Dict:
        return {'prior': 'gmm', 'n_latent': self.n_latent, 'n_components': self.K, 'min_std': self.min_std}

class VampPrior(Prior):

    def __init__(self, encoder: nn.Module, input_shape: Tuple[int, ...], n_pseudo_inputs: int = 512, hidden: int = None):
        super().__init__()
        self.encoder = encoder
        self.n_pseudo_inputs = n_pseudo_inputs # M
        self.input_shape = input_shape

        self._idle: nn.UninitializedBuffer
        self.register_buffer('_idle', torch.eye(self.n_pseudo_inputs, requires_grad=False))  # (M,M)
        if hidden is None:
            self.embedding = nn.Sequential(
                nn.Linear(self.n_pseudo_inputs, int(np.prod(input_shape))),
                nn.Hardtanh(0.0, 1.0)
            )
        else:
            self.embedding = nn.Sequential(
                nn.Linear(self.n_pseudo_inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, int(np.prod(input_shape))),
                nn.Hardtanh(0.0, 1.0)
            )

        self.logits = nn.Parameter(torch.zeros(self.n_pseudo_inputs))  # (M,)

    def pseudo_inputs(self) -> torch.Tensor:
        x = self.embedding(self._idle)  # (M, C*H*W)
        C, H, W = self.input_shape
        x = x.view(self.n_pseudo_inputs, C, H, W)
        return x

    def _components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        U = self.pseudo_inputs()   # (M, C, H, W)
        mu, logvar = self.encoder(U)                  
        std = (0.5 * logvar).exp()
        return mu, std                                   # (M,D), (M,D)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        from torch.distributions import MixtureSameFamily
        mu, std = self._components()                     # (M,D)
        comp = Independent(Normal(mu, std), 1)           # K comps, diag cov
        mix  = Categorical(logits=self.logits)
        m = MixtureSameFamily(mix, comp)
        return m.log_prob(z)                             # (B,)

    def _mixture(self) -> MixtureSameFamily:
        mu, std = self._components()                     # (M,D)
        comp = Independent(Normal(mu, std), 1)           # M comps, diag cov
        mix  = Categorical(logits=self.logits)
        return MixtureSameFamily(mix, comp)

    def sample(self, num_samples: int = 1, device = None) -> torch.Tensor:
        z = self._mixture().sample((num_samples,))  # (S,D)
        return z if device is None else z.to(device)
        mu, std = self._components()
        if device is not None:
            mu, std = mu.to(device), std.to(device)   # (M,D)
            logits = self.logits.to(device)           # (M,)
        else: 
            logits = self.logits
        cat = Categorical(logits=logits)              # (M,)
        k = cat.sample((num_samples,))
        eps = torch.randn_like(mu[k])
        return mu[k] + eps * std[k]                  # (S,D)
    
    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        # r_bk ∝ π_k N(z_b ; μ_k, Σ_k)
        mu, std = self._components()                     # (M,D)
        comp = Independent(Normal(mu, std), 1)  # (M)
        log_comp = comp.log_prob(z[:, None, :])           # (B,M)
        log_mix  = F.log_softmax(self.logits, dim=-1)     # (M,)
        return F.softmax(log_comp + log_mix, dim=-1)      # (B,M)

    def get_hparams(self) -> Dict:
        return {'prior': 'vamp', 'n_pseudo_inputs': self.n_pseudo_inputs}
