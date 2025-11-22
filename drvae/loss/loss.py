# drvae/loss/loss.py
import torch
from typing import Union

class Loss:
    def __init__(self, 
                reconstruction: torch.Tensor, 
                kl_div: torch.Tensor, 
                kl_weight: float = 1.0,
                reg_term: Union[torch.Tensor, None] = None,
                reg_weight: float = 1.0,):
        if not isinstance(reconstruction, torch.Tensor) or not isinstance(kl_div, torch.Tensor) or (reg_term is not None and not isinstance(reg_term, torch.Tensor)):
            raise ValueError("Loss components must be torch.Tensor instances")
        
        self.reconstruction = reconstruction
        self.kl_div = kl_div
        self.kl_weight = kl_weight
        self.reg_term = reg_term
        self.reg_weight = reg_weight
        
    
    @property
    def total_loss(self):
        if self.reg_term is None:
            return self.reconstruction + self.kl_weight*self.kl_div
        else:
            return self.reconstruction + self.kl_weight*self.kl_div + self.reg_weight*self.reg_term
        
    @property
    def elbo(self):
        return - (self.reconstruction + self.kl_weight*self.kl_div)

    def log_dict(self, prefix=""):
        log_data = {
            f"{prefix}reconstruction_loss": self.reconstruction.item(),
            f"{prefix}kl_divergence_loss": self.kl_div.item(),
            f"{prefix}kl_weight": self.kl_weight,
        }
        
        if self.reg_term is not None:
            log_data[f"{prefix}regularization_loss"] = self.reg_term.item()
            log_data[f"{prefix}reg_weight"] = self.reg_weight
        
        log_data[f"{prefix}total_loss"] = self.total_loss.item()
        
        return log_data