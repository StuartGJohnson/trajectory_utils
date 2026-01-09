import torch
import numpy as np
from torch_traj_utils.cartpole import CartpoleEnvironmentParams
from torch_traj_utils.casadi_ode import CasadiODE
import casadi as ca


class CartpoleVelocityCasODE(CasadiODE):
    ep: CartpoleEnvironmentParams

    def __init__(self, ep: CartpoleEnvironmentParams):
        super().__init__()
        self.ep = ep
        self.g = 9.81

    def ode(self, x: ca.MX, u: ca.MX) -> ca.MX:
        # x = [pos, theta, vel, theta_dot]
        L = float(self.ep.pole_length)
        tau = float(self.ep.cart_tau)
        g = float(self.g)

        pos = x[0]
        th = x[1]
        dx = x[2]
        dth = x[3]
        v_cmd = u[0]

        sin_th = ca.sin(th)
        cos_th = ca.cos(th)

        ddx = (v_cmd - dx) / tau
        ddth = -1.5 * (g * sin_th + ddx * cos_th) / L

        return ca.vertcat(dx, dth, ddx, ddth)

