"""
Cartpole energy calcs. Differentiable (for optimization) and non-differentiable - for
metrics. There will need to be a differentiable version for the casadi/ipopt solver.
"""

import torch
import numpy as np
from torch_traj_utils.cartpole import CartpoleEnvironmentParams

class CartpoleEnergy:
    ep: CartpoleEnvironmentParams
    def __init__(self, ep: CartpoleEnvironmentParams):
        self.ep = ep
        self.g = 9.81

    def energy_torch(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4] (x, θ, dx, dθ)
        returns ds/dt with shape [..., 4]
        """
        m_p = self.ep.pole_mass
        m_c = self.ep.cart_mass
        L  = self.ep.pole_length
        #tau = self.ep.cart_tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        #sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        total_energy = m_p * L**2 * dth**2 / 6 + m_p * L * cos_th * (dth*dx - g) / 2 + (m_p + m_c) * dx**2 / 2
        return total_energy

    def energy(self, s: np.ndarray) -> np.ndarray:
        """
        s: [..., 4] (x, θ, dx, dθ)
        returns ds/dt with shape [..., 4]
        numpy version.
        """
        m_p = self.ep.pole_mass
        m_c = self.ep.cart_mass
        L  = self.ep.pole_length
        #tau = self.ep.cart_tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        #sin_th = torch.sin(th)
        cos_th = np.cos(th)
        total_energy = m_p * L**2 * dth**2 / 6 + m_p * L * cos_th * (dth*dx - g) / 2 + (m_p + m_c) * dx**2 / 2
        return total_energy

    def compute_energy_torch(self, s: np.ndarray) -> np.ndarray:
        S = torch.as_tensor(s, dtype=torch.float64)
        tot_energy = self.energy_torch(S)
        return tot_energy.detach().cpu().numpy()

    def compute_energy(self, s: np.ndarray) -> np.ndarray:
        tot_energy = self.energy(s)
        return tot_energy

    def energy_pole_torch(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4]  (x, θ, dx, dθ)
        returns ds/dt with shape [..., 4]
        """
        m_p = self.ep.pole_mass
        m_c = self.ep.cart_mass
        L  = self.ep.pole_length
        #tau = self.tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        #sin_th = torch.sin(th)
        cos_th = torch.cos(th)
        pole_energy = m_p * L**2 * dth**2 / 6 - m_p * L * cos_th * g / 2
        return pole_energy

    def energy_pole(self, s: np.ndarray) -> np.ndarray:
        """
        s: [..., 4]  (x, θ, dx, dθ)
        returns ds/dt with shape [..., 4].
        numpy version.
        """
        m_p = self.ep.pole_mass
        m_c = self.ep.cart_mass
        L  = self.ep.pole_length
        #tau = self.tau
        g  = self.g

        x, th, dx, dth = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        #sin_th = torch.sin(th)
        cos_th = np.cos(th)
        pole_energy = m_p * L**2 * dth**2 / 6 - m_p * L * cos_th * g / 2
        return pole_energy

    def compute_energy_pole_torch(self, s: np.ndarray) -> np.ndarray:
        S = torch.as_tensor(s, dtype=torch.float64)
        pole_energy = self.energy_pole_torch(S)
        return pole_energy.detach().cpu().numpy()

    def compute_energy_pole(self, s: np.ndarray) -> np.ndarray:
        pole_energy = self.energy_pole(s)
        return pole_energy