import torch
import numpy as np
from torch_traj_utils.cartpole import CartpoleEnvironmentParams
from torch_traj_utils.torch_ode import TorchODE


class CartpoleForceODE(TorchODE):
    ep: CartpoleEnvironmentParams

    def __init__(self, ep: CartpoleEnvironmentParams, dt: float):
        super().__init__(dt)
        self.ep = ep
        self.g = 9.81
        super().setup(dt)

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4] (x, theta(th), dx, dtheta(dth))
        u: [..., 1] (force)
        returns ds/dt with shape [..., 4]
        """
        m_p = self.ep.pole_mass
        m_c = self.ep.cart_mass
        L  = self.ep.pole_length
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        # mass at tip of pole
        # d = m_c + m_p * (sin_th ** 2)
        # mass along pole
        d = 4 * m_c + m_p * (1 + 3 * sin_th ** 2)

        # mass at tip of pole
        # ddx = (u[..., 0] + m_p * g * cos_th * sin_th + m_p * L * sin_th * (dth**2)) / d
        # ddth = -(cos_th * u[..., 0] +  g * sin_th * (m_c + m_p) + m_p * L * sin_th * cos_th * (dth ** 2)) / (d * L)
        # mass along pole
        ddx = ( 4 * u[..., 0] + 3 * m_p * g * cos_th * sin_th + 2 * m_p * L * sin_th * (dth **2) ) / d
        ddth = -(6 * cos_th * u[0, ...] + 6 * g * sin_th * (m_c + m_p) + 3 * m_p * L * sin_th * cos_th * (dth ** 2 )) / (d * L)
        ds = torch.stack((dx, dth, ddx, ddth), dim=-1)
        return ds



