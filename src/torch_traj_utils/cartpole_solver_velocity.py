"""
Cart velocity (servo) controlled cartpole.
"""

from torch_traj_utils.scp_solver import SCPSolver, SolverParams

import cvxpy as cvx
import torch
import numpy as np
from torch_traj_utils.cartpole import CartpoleEnvironmentParams
from torch_traj_utils.cartpole_energy import CartpoleEnergy
from torch_traj_utils.cartpole_velocity_ode import CartpoleVelocityODE

class CartpoleSolverVelocity(SCPSolver):
    ep: CartpoleEnvironmentParams
    energy: CartpoleEnergy
    def __init__(self, sp: SolverParams, ep: CartpoleEnvironmentParams):
        super().__init__(sp)
        self.ep = ep
        self.g = 9.81
        self.energy = CartpoleEnergy(ep)
        t_ode = CartpoleVelocityODE(ep, self.params.dt)
        self.setup(t_ode)

    def opt_problem(self) -> cvx.Problem:
        # set up cvxpy optimization
        objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.params.P)
        objective += cvx.sum([cvx.sum(cvx.huber(self.params.Q @ (self.s_cvx[i] - self.s_goal))) + cvx.quad_form(self.u_cvx[i], self.params.R) for i in range(self.N)])
        #objective = cvx.quad_form((self.s_cvx[self.N] - self.s_goal), self.P) + cvx.sum(
        #    [cvx.quad_form(self.s_cvx[i] - self.s_goal, self.Q) + cvx.quad_form(self.u_cvx[i], self.R) for i in range(self.N)])
        constraints = [self.s_cvx[i + 1] == self.c_param[i] + self.A_param[i] @ self.s_cvx[i] +
                       self.B_param[i] @ self.u_cvx[i] for i in range(self.N)]
        constraints += [self.s_cvx[0] == self.s0]
        constraints += [cvx.abs(self.u_cvx) <= self.params.u_max]
        constraints += [cvx.abs(self.s_cvx) <= self.params.s_max]
        constraints += [cvx.max(cvx.abs(self.s_cvx - self.s_prev_param)) <= self.params.rho]
        constraints += [cvx.max(cvx.abs(self.u_cvx - self.u_prev_param)) <= self.params.rho]
        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        return prob
