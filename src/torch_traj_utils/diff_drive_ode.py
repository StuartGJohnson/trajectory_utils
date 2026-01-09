import torch
import numpy as np
from torch_traj_utils.torch_ode import TorchODE


class DiffDriveODE(TorchODE):

    def __init__(self, dt: float):
        super().__init__(dt)
        super().setup(dt)

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 3] (x, y, θ)
        u: [..., 2] (v, omega)
        returns ds/dt with shape [..., 4]
        """

        x, y, th = s[..., 0], s[..., 1], s[..., 2]
        v, omega = u[..., 0], u[..., 1]
        sin_th, cos_th = torch.sin(th), torch.cos(th)

        dx = v * cos_th
        dy = v * sin_th
        dth = omega

        ds = torch.stack((dx, dy, dth), dim=-1)
        return ds


