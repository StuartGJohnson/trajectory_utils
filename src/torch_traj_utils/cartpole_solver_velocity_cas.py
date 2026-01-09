"""
CasADi/IPOPT cartpole velocity-servo solver (direct OCP, multiple shooting).

This is a new solver (not SCP/CVXPY). It solves for the whole trajectory in one NLP.

State:  s = [x, theta, dx, dtheta]
Control: u = [v_cmd]   (velocity command / setpoint)
Dynamics (matching the existing velocity-servo model):
  ddx   = (u - dx)/tau
  ddth  = -1.5*(g*sin(th) + ddx*cos(th))/L
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

import casadi as ca

from torch_traj_utils.cartpole import CartpoleEnvironmentParams
from torch_traj_utils.cas_solver import CasadiOCPSolver, CasadiSolverParams
from torch_traj_utils.cartpole_velocity_cas_ode import CartpoleVelocityCasODE
from torch_traj_utils.cartpole_velocity_ode import CartpoleVelocityODE
from torch_traj_utils.cartpole_energy import CartpoleEnergy

class CartpoleSolverVelocityCas(CasadiOCPSolver):
    def __init__(self, sp: CasadiSolverParams, ep: CartpoleEnvironmentParams):
        super().__init__(sp)
        self.ep = ep
        self.g = 9.81
        self.energy = CartpoleEnergy(ep)
        c_ode = CartpoleVelocityCasODE(ep)
        t_ode = CartpoleVelocityODE(ep, sp.dt)
        self.setup(c_ode, t_ode)


