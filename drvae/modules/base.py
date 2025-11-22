# drvae/modules/base.py
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict, List, Optional, Literal
from torch.distributions import Normal, Bernoulli
from ..loss import Loss
from .prior import Prior

class BaseVAE(nn.Module, ABC):

    def __init__(self,
                 input_shape: Tuple[int, ...] = (1, 28, 28),
                 n_latent: int = 10,
                 encoder_config: Dict = None,
                 decoder_config: Dict = None,
                 likelihood: Literal['bernoulli', 'gaussian'] = 'bernoulli',
                 gaussian_std: float = 0.1,
                 prior_config: Dict = None,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.n_latent = n_latent
        self.likelihood = likelihood
        self.gaussian_std = gaussian_std

        self.encoder = self._encoder_init(encoder_config)
        self.decoder = self._decoder_init(decoder_config)

        self.prior = self._set_prior(prior_config)

        self.configs = {
            'encoder_config': encoder_config,
            'decoder_config': decoder_config,
            'prior_config': self.prior.get_hparams(),
            }


    def encode(self, x: torch.Tensor) -> Normal:
        mu, log_var = self.encoder(x)
        qz_dist = Normal(mu, torch.exp(0.5 * log_var))
        return qz_dist

    def decode(self, z: torch.Tensor) -> Union[Normal, Bernoulli]:
        out = self.decoder(z)
        out = out.view(-1, int(np.prod(self.input_shape)))
        if self.likelihood == 'bernoulli':
            px_dist = Bernoulli(logits=out)
        elif self.likelihood == 'gaussian':
            sigma = torch.ones_like(out) * self.gaussian_std
            px_dist = Normal(out, sigma)
        return px_dist

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[Tuple[Normal, Normal], Tuple[Normal, Bernoulli]]:
        pass

    @abstractmethod
    def loss_function(self, x: torch.Tensor, **kwargs: Any) -> Loss:
        pass

    def _encoder_init(self, config: Union[Dict, None]) -> nn.Module:
        if config is None:
            from ..encoders import LinearEncoder
            return LinearEncoder(self.input_shape, [300], self.n_latent)
        _validate_config(config, required_keys=['class', 'args'])
        _validate_config(config['args'], required_keys=['input_shape', 'h_dims', 'n_latent'])
        return _build_model_from_config(config)

    def _decoder_init(self, config: Union[Dict, None]) -> nn.Module:
        if config is None:
            from ..decoders import LinearDecoder
            return LinearDecoder(self.n_latent, [300], self.input_shape)
        _validate_config(config, required_keys=['class', 'args'])
        _validate_config(config['args'], required_keys=['n_latent', 'h_dims', 'output_shape'])
        return _build_model_from_config(config)

    def _set_prior(self, prior_config: Optional[Dict]) -> Prior:
        if prior_config is None:
            from .prior import StandardNormalPrior
            return StandardNormalPrior(self.n_latent)
        
        _validate_config(prior_config, required_keys=['prior'])
        prior = prior_config['prior']
        prior_args = prior_config.get('args', {})
        if 'n_latent' in prior_args:
            prior_args.pop("n_latent", None)
            
        if prior == 'standard':
            from .prior import StandardNormalPrior
            pz = StandardNormalPrior(self.n_latent)
        elif prior == 'gmm':
            from .prior import GMMPrior
            pz = GMMPrior(n_latent=self.n_latent, **prior_args)
        elif prior == 'vamp':
            from .prior import VampPrior
            pz = VampPrior(encoder=self.encoder, input_shape=self.input_shape, **prior_args)
        elif prior == 'flow':
            raise NotImplementedError()
        elif prior == 'dpa':
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown prior: {prior}")
        return pz
        
        
def _build_model_from_config(config: Dict) -> nn.Module:
    module_cls = config['class']
    module_args = config.get('args', {})
    if module_cls == 'linear_encoder':
        from ..encoders import LinearEncoder
        return LinearEncoder(**module_args)
    elif module_cls == 'gated_linear_encoder':
        from ..encoders import GatedLinearEncoder
        return GatedLinearEncoder(**module_args)
    elif module_cls == 'mnist_encoder':
        from ..encoders import MnistEncoder
        return MnistEncoder(**module_args)
    elif module_cls == 'linear_decoder':
        from ..decoders import LinearDecoder
        return LinearDecoder(**module_args)
    elif module_cls == 'gated_linear_decoder':
        from ..decoders import GatedLinearDecoder
        return GatedLinearDecoder(**module_args)
    elif module_cls == 'mnist_decoder':
        from ..decoders import MnistDecoder
        return MnistDecoder(**module_args)
    else:
        raise ValueError(f"Unknown module class: {module_cls}")
        

def _validate_config(config: Dict, required_keys: List[str]) -> None:
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config must contain '{key}' key.")
