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
from torch_traj_utils.cas_solver import CasadiSolver, CasadiSolverParams
from torch_traj_utils.cartpole_velocity_cas_ode import CartpoleVelocityCasODE
from torch_traj_utils.cartpole_velocity_ode import CartpoleVelocityODE
from torch_traj_utils.cartpole_energy import CartpoleEnergy

class CartpoleSolverVelocityCas(CasadiSolver):
    def __init__(self, sp: CasadiSolverParams, ep: CartpoleEnvironmentParams):
        super().__init__(sp)
        self.ep = ep
        self.g = 9.81
        self.energy = CartpoleEnergy(ep)
        c_ode = CartpoleVelocityCasODE(ep)
        t_ode = CartpoleVelocityODE(ep, sp.dt)
        self.setup(c_ode, t_ode)


    def build_problem(self) -> None:
        sp = self.params
        N_max = self.N if sp.N_max < 0 else sp.N_max
        n, m = self.n, self.m

        # Parameters
        x0_p = ca.MX.sym("x0", n)
        x_goal_p = ca.MX.sym("x_goal", n)
        mask_stage = ca.MX.sym("mask_stage", N_max)
        mask_terminal = ca.MX.sym("mask_terminal", N_max + 1)

        # Decision variables
        X = ca.MX.sym("X", (N_max + 1) * n)
        U = ca.MX.sym("U", N_max * m)
        dt = ca.MX.sym("dt") if sp.optimize_time else ca.DM(float(sp.dt))

        def Xk(k: int) -> ca.MX:
            return X[k * n : (k + 1) * n]

        def Uk(k: int) -> ca.MX:
            return U[k * m : (k + 1) * m]

        g = []
        J = 0

        # initial condition
        g.append(Xk(0) - x0_p)

        # dynamics + cost
        u_prev = Uk(0)
        for k in range(N_max):
            xk = Xk(k)
            uk = Uk(k)
            x_next = Xk(k + 1)

            # Masked dynamics: x_next = rk4(xk, uk) if mask_stage[k] else x_next = xk
            ode_next = self.c_ode.rk4(xk, uk, dt)
            g.append(mask_stage[k] * (x_next - ode_next) + (1 - mask_stage[k]) * (x_next - xk))

            # Masked stage cost - I am not included dt scaling here (put this in your
            # definitions of Q, R and Rd
            J = J + mask_stage[k] * self.stage_cost(xk, uk, x_goal_p, u_prev)
            # small regularization for unused controls (from Gemini, commented out)
            #J = J + (1 - mask_stage[k]) * 1e-6 * ca.sumsqr(uk)

            u_prev = uk

        # Masked terminal cost
        for k in range(1, N_max + 1):
            J = J + mask_terminal[k] * self.terminal_cost(Xk(k), x_goal_p)

        # optional time penalty
        if sp.optimize_time and sp.time_weight != 0.0:
            actual_N = ca.sum1(mask_stage)
            J = J + float(sp.time_weight) * (actual_N * dt)

        # pack variables
        if sp.optimize_time:
            w = ca.vertcat(X, U, dt)
            dt_index = (N_max + 1) * n + N_max * m
            self.var_slices = {
                "X": slice(0, (N_max + 1) * n),
                "U": slice((N_max + 1) * n, (N_max + 1) * n + N_max * m),
                "dt": slice(dt_index, dt_index + 1),
            }
        else:
            w = ca.vertcat(X, U)
            self.var_slices = {
                "X": slice(0, (N_max + 1) * n),
                "U": slice((N_max + 1) * n, (N_max + 1) * n + N_max * m),
            }

        g = ca.vertcat(*g)

        # bounds
        lbx = []
        ubx = []

        # state bounds: |x| <= s_max
        s_max = np.asarray(sp.s_max, dtype=float).reshape(n)
        for _ in range(N_max + 1):
            lbx.extend((-s_max).tolist())
            ubx.extend((+s_max).tolist())

        # control bounds: |u| <= u_max
        u_max = np.asarray(sp.u_max, dtype=float).reshape(m)
        for _ in range(N_max):
            lbx.extend((-u_max).tolist())
            ubx.extend((+u_max).tolist())

        if sp.optimize_time:
            lbx.append(float(sp.dt_min))
            ubx.append(float(sp.dt_max))

        # equality constraints g == 0
        lbg = [0.0] * int(g.shape[0])
        ubg = [0.0] * int(g.shape[0])

        self.nlp = {
            "x": w,
            "f": J,
            "g": g,
            "p": ca.vertcat(x0_p, x_goal_p, mask_stage, mask_terminal),
        }

        self.bounds = {
            "lbx": ca.DM(lbx),
            "ubx": ca.DM(ubx),
            "lbg": ca.DM(lbg),
            "ubg": ca.DM(ubg),
        }

        opts = {
            "ipopt.max_iter": int(sp.ipopt_max_iter),
            "ipopt.tol": float(sp.ipopt_tol),
            "ipopt.print_level": int(sp.ipopt_print_level),
            "ipopt.sb": sp.ipopt_sb,
            "print_time": bool(sp.print_time),
        }

        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, opts)
