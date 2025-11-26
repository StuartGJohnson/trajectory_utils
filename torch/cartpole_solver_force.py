"""
Force (on cart) controlled cartpole. A specialization of SCPSolver.
"""

from scp_solver import SCPSolver, SolverParams

import cvxpy as cvx
import torch
from cartpole import CartpoleEnvironmentParams
from cartpole_energy import CartpoleEnergy

class CartpoleSolverForce(SCPSolver):
    ep: CartpoleEnvironmentParams
    energy: CartpoleEnergy
    def __init__(self, sp: SolverParams, ep: CartpoleEnvironmentParams):
        super().__init__(sp)
        self.ep = ep
        self.g = 9.81
        self.energy = CartpoleEnergy(ep)
        self.setup()

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = self.opt_problem_objective(self.s_cvx, self.u_cvx)
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        constraints += [self.s_cvx[0] == self.s0]
        constraints += [cvx.abs(self.u_cvx) <= self.params.u_max]
        constraints += [cvx.abs(self.s_cvx) <= self.params.s_max]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.params.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.params.rho]
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob

    def opt_problem_objective(self, s: cvx.Expression, u: cvx.Expression) -> cvx.Expression:
        objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.params.P)
        objective += cvx.sum([cvx.sum(cvx.huber(self.params.Q @ (self.s_cvx[i] - self.s_goal))) + cvx.quad_form(self.u_cvx[i], self.params.R) for i in range(self.N)])
        # objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.P) + cvx.sum(
        #     [cvx.quad_form(self.s_cvx[i] - self.s_goal, self.Q) + cvx.quad_form(self.u_cvx[i], self.R) for i in
        #      range(self.N)])
        return objective

    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        s: [..., 4]  (x, theta(th), dx, dtheta(dth))
        u: [..., 1]  (force)
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
