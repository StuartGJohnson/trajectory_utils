import numpy as np
from torch_traj_utils.casadi_ode import CasadiODE
import casadi as ca

class DiffDriveCasODE(CasadiODE):

    def __init__(self):
        super().__init__()

    def ode(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """
        s: [..., 3] (x, y, θ)
        u: [..., 2] (v, omega)
        returns ds/dt with shape [..., 4]
        """
        x, y, th = x[0], x[1], x[2]
        v, omega = u[0], u[1]
        sin_th, cos_th = ca.sin(th), ca.cos(th)

        dx = v * cos_th
        dy = v * sin_th
        dth = omega

        return ca.vertcat(dx, dy, dth)


