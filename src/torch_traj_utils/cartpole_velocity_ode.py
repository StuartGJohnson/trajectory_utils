import torch
import numpy as np
from torch_traj_utils.cartpole import CartpoleEnvironmentParams
from torch_traj_utils.torch_ode import TorchODE


class CartpoleVelocityODE(TorchODE):
    ep: CartpoleEnvironmentParams

    def __init__(self, ep: CartpoleEnvironmentParams, dt: float):
        super().__init__(dt)
        self.ep = ep
        self.g = 9.81
        super().setup(dt)

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4] (x, θ, dx, dθ)
        u: [..., 1] (velocity)
        returns ds/dt with shape [..., 4]
        """
        # these may be useful later
        # m_p = self.pole_mass
        # m_c = self.cart_mass
        L  = self.ep.pole_length
        tau = self.ep.cart_tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)

        ddx  = (u[..., 0] - dx) / tau
        # mass at tip of pole
        #ddth = -(g * sin_th + ddx * cos_th) / L
        # mass along pole
        ddth = -1.5 * (g * sin_th + ddx * cos_th) / L

        ds = torch.stack((dx, dth, ddx, ddth), dim=-1)
        return ds


