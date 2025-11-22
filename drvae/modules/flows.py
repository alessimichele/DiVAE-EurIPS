import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import logging

class LinearNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class CouplingLayer(nn.Module):
    
    def __init__(self, network: nn.Module, mask: torch.Tensor):
        super().__init__()
        self.network = network
        self.mask: nn.UninitializedBuffer
        self.register_buffer('mask', mask.float())

    def forward(self, z: torch.Tensor, ldj: torch.Tensor, reverse: bool=False, scale_factor: float=2.0):
        """
        z: input to the flow
        ldj: The current ldj of the previous flows. 
             The ldj of this layer will be added to this tensor.
        reverse: If True, we apply the inverse of the layer. 
        orig_img (optional) - Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input)
        z_in = z * self.mask # (B, D)
        nn_out = self.network(z_in) # (B, 2*n_unmasked) = (B, D) since n_unmasked = n_masked = D/2
        s_unmasked, t_unmasked = nn_out.chunk(2, dim=1)
        
        # Mask outputs (only transform the second part)
        inv_mask = (1.0 - self.mask).bool() # (B, D)
        s = torch.zeros_like(z)
        t = torch.zeros_like(z)
        s[:, inv_mask] = s_unmasked
        t[:, inv_mask] = t_unmasked

        s = torch.tanh(s) * scale_factor  # es: scale_factor = 2.0

        # Affine transformation
        if not reverse:
            # x -> u
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            # u -> x
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=1)
            
        return z, ldj


class FlowAdapter(nn.Module):
    r"""
    forward flow: u --> z through f^-1, z = f^-1(u)

    inverse flow: z --> u through f, u = f(z)

    Actually:
    f^-1: p(u), Dequantization, CL, CL, ..., CL, p(z)
    """
    def __init__(self, dim: int, n_hidden: int = 128, K: int = 4):
        super().__init__()
        self.dim = dim
        masks = []
        base = torch.arange(dim) % 2
        self.base_dist = Independent(Normal(torch.zeros(self.dim),torch.ones(self.dim)), 1)

        for k in range(K):
            m = (base if k % 2 == 0 else 1 - base).float()
            masks.append(m)

        
        flows = []
        for m in masks:
            n_unmasked = int((1.0 - m).sum().item())
            net = LinearNetwork(in_dim=dim, out_dim=2 * n_unmasked, hidden_dim=n_hidden)
            flows.append(CouplingLayer(network=net, mask=m))
        self.flows = nn.ModuleList(flows)

    def forward_flow(self, x: torch.Tensor):
        # u -> z (f^-1)
        z, ldj = x, torch.zeros_like(x.sum(dim=1))  
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def inverse_flow(self, u: torch.Tensor):
        # z -> u (f), z.shape = u.shape
        z, ldj = u, torch.zeros_like(u.sum(dim=1))
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z, ldj

    def compute_loglikelihood(self, x, return_ll: bool=False):
        """
        log p_z (forward_flow(u)) = log p_z (f^-1(u))

        Given a batch of images, return the likelihood of those. 
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.forward_flow(x)        
        log_pz = self.base_dist.log_prob(z) # (B, )
        
        log_px = ldj + log_pz
        nll = -log_px # B, 2
        if return_ll:
            return log_px
        else:
            return nll.mean()

    def forward(self,x: torch.Tensor):
        """
        Forward 'for torch', meaning the direction we follow during the training, which is x --> u (f^-1)
        """
        return self.compute_loglikelihood(x)

    def sample_step(self, idx, num_samples=1000) -> torch.Tensor:
        z = self.base_dist.sample(num_samples)
        u, ldj = z, torch.zeros(z.shape[0], device=z.device)
        for i, flow in enumerate(reversed(self.flows)):
            if i == idx:
                break
            u, ldj = flow(u, ldj, reverse=True)
        return u
        

if __name__ == "__main__":

    ## Simple flow
    model = FlowAdapter(dim=5, n_hidden=16, K=6)
    for f in model.flows:
        print(f.mask)

    x = torch.randn(10, 5)
    u, ldj = model.forward_flow(x)
    print("x", x)
    print("u", u)
    print("ldj", ldj)