"""
Generic CasADi+IPOPT multiple-shooting optimal control solver.

This file is intentionally independent of the existing CVXPY/Torch SCP framework.
It provides a small base class you can subclass for specific systems (e.g., cartpole).

Key features:
- Multiple shooting with RK4 integration
- Box constraints on state/control
- Optional optimization of dt (thus total horizon time T = N*dt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from torch_traj_utils.torch_ode import TorchODE
from torch_traj_utils.casadi_ode import CasadiODE
from torch_traj_utils.trajectory_solver import TrajectorySolverParams, TrajectorySolver

import numpy as np
import casadi as ca
import time

@dataclass
class CasadiSolverParams(TrajectorySolverParams):
    # Optional time optimization: optimize dt within [dt_min, dt_max]
    optimize_time: bool = False
    dt_min: float = 0.01
    dt_max: float = 0.10
    time_weight: float = 0.0  # adds time_weight * (N*dt) to objective

    # IPOPT options
    ipopt_max_iter: int = 2000
    ipopt_tol: float = 1e-6
    ipopt_print_level: int = 0
    ipopt_sb: str = "yes"
    print_time: bool = True


class CasadiOCPSolver(TrajectorySolver):
    """
    Base class for a direct multiple-shooting OCP.

    Subclass responsibilities:
    - implement ode(self, x, u) -> xdot (CasADi expressions)
    - optionally override stage_cost / terminal_cost
    """
    t_ode: TorchODE
    c_ode: CasadiODE

    def __init__(self, sp: CasadiSolverParams):
        self.params: CasadiSolverParams = sp

        self.n = int(sp.Q.shape[0])
        self.m = int(sp.R.shape[0])

        self.s0: Optional[np.ndarray] = None
        self.s_goal: Optional[np.ndarray] = None

        self.nlp: Optional[Dict[str, Any]] = None
        self.solver: Optional[ca.Function] = None
        self.var_slices: Dict[str, slice] = {}
        self.N = 0

    def setup(self, c_ode:CasadiODE, t_ode: TorchODE):
        # these virtual-method calls do not belong in the constructor!
        #self.prob = self.opt_problem()
        self.t_ode = t_ode
        self.c_ode = c_ode

    def set_trajectory(self, s_init, u_init):
        """simply set the trajectory."""
        self.s_init = s_init
        self.u_init = u_init

    def get_ode(self) -> TorchODE:
        return self.t_ode

    def get_params(self) -> TrajectorySolverParams:
        return self.params

    # --------- dynamics / cost hooks ---------

    def stage_cost(self, x: ca.MX, u: ca.MX, x_goal: ca.MX, u_prev: ca.MX) -> ca.MX:
        dx = x - x_goal
        du = u - u_prev
        return ca.mtimes([dx.T, ca.DM(self.params.Q), dx]) + ca.mtimes([u.T, ca.DM(self.params.R), u]) + ca.mtimes([du.T, ca.DM(self.params.Rd), du])

    def terminal_cost(self, xN: ca.MX, x_goal: ca.MX) -> ca.MX:
        dx = xN - x_goal
        return ca.mtimes([dx.T, ca.DM(self.params.P), dx])

    # --------- integration ---------

    # def rollout(self, s0: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Rolls out a control sequence U from starting state s0 using RK4.
    #     U: (N, m)
    #     returns X: (N+1, n)
    #     """
    #     N = U.shape[0]
    #     X = np.zeros((N + 1, self.n))
    #     X[0, :] = s0
    #     dt = self.params.dt
    #
    #     if self.rk4_fn is None:
    #         # Create a CasADi function for numerical evaluation
    #         x_sym = ca.MX.sym("x", self.n)
    #         u_sym = ca.MX.sym("u", self.m)
    #         dt_sym = ca.MX.sym("dt")
    #         next_x = self.rk4(x_sym, u_sym, dt_sym)
    #         self.rk4_fn = ca.Function("rk4", [x_sym, u_sym, dt_sym], [next_x])
    #
    #     for k in range(N):
    #         X[k + 1, :] = np.array(self.rk4_fn(X[k, :], U[k, :], dt)).flatten()
    #
    #     return X, U

    # --------- public API ---------

    def reset(self, s0: np.ndarray, s_goal: np.ndarray, N:int) -> None:
        s0 = np.asarray(s0, dtype=float).reshape(-1)
        s_goal = np.asarray(s_goal, dtype=float).reshape(-1)
        if s0.shape[0] != self.n:
            raise ValueError(f"s0 must be shape ({self.n},), got {s0.shape}")
        if s_goal.shape[0] != self.n:
            raise ValueError(f"s_goal must be shape ({self.n},), got {s_goal.shape}")

        self.s0 = s0
        self.s_goal = s_goal
        self.N = N

        self.build_problem()

    def initialize_trajectory(self):
        """Set and rollout the zero control trajectory."""
        n = self.params.Q.shape[0]  # state dimension
        m = self.params.R.shape[0]  # control dimension
        # the zero control trajectory
        self.u_init = np.zeros((self.N, m))
        self.x_init = np.zeros((self.N + 1, n))
        self.x_init[0] = self.s0
        self.x_init, self.u_init = self.t_ode.rollout(self.x_init, self.u_init, self.N)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float, int]:
        """
        Returns:
          X: (N+1, n)
          U: (N, m)
          various helpful things
        """
        if self.solver is None or self.nlp is None:
            raise RuntimeError("Call reset(s0, s_goal) before solve().")

        sp = self.params
        N, n, m = self.N, self.n, self.m

        #if self.dt_init is None:
        dt_init = float(sp.dt)

        start_time = time.time()

        x_init = np.asarray(self.x_init, dtype=float).reshape(N + 1, n)
        u_init = np.asarray(self.u_init, dtype=float).reshape(N, m)

        w0 = []
        w0.extend(x_init.reshape(-1).tolist())
        w0.extend(u_init.reshape(-1).tolist())
        if sp.optimize_time:
            w0.append(dt_init)

        sol = self.solver(
            x0=ca.DM(w0),
            lbx=self.bounds["lbx"],
            ubx=self.bounds["ubx"],
            lbg=self.bounds["lbg"],
            ubg=self.bounds["ubg"],
        )

        w_opt = np.array(sol["x"]).reshape(-1)

        # unpack
        sx = self.var_slices["X"]
        su = self.var_slices["U"]
        X = w_opt[sx].reshape(N + 1, n)
        U = w_opt[su].reshape(N, m)

        if sp.optimize_time:
            dt = float(w_opt[self.var_slices["dt"]][0])
        else:
            dt = float(sp.dt)

        end_time = time.time()
        elapsed_time = end_time - start_time

        info = {
            "cost": float(sol["f"]),
            "dt": dt,
            "T": float(N * dt),
            "status": str(self.solver.stats().get("return_status", "")),
            "success": bool(self.solver.stats().get("success", False)),
        }
        status = str(self.solver.stats().get("return_status", ""))
        converged = bool(self.solver.stats().get("success", False))

        print(f"Solve time: {elapsed_time:.4f} seconds, converged: {converged}, status: {status}")

        # todo: looks like cost during casadi iterations is not available
        J = np.array([])

        # todo: how to get iterations?
        iterations = -1
        return X, U, J, converged, status, elapsed_time, iterations

    # --------- build NLP ---------

    def build_problem(self) -> None:
        sp = self.params
        N, n, m = self.N, self.n, self.m

        x0 = ca.DM(self.s0)
        x_goal = ca.DM(self.s_goal)

        X = ca.MX.sym("X", (N + 1) * n)  # flattened
        U = ca.MX.sym("U", N * m)        # flattened
        dt = ca.MX.sym("dt") if sp.optimize_time else ca.DM(float(sp.dt))

        def Xk(k: int) -> ca.MX:
            return X[k * n : (k + 1) * n]

        def Uk(k: int) -> ca.MX:
            return U[k * m : (k + 1) * m]

        g = []
        J = 0

        # initial condition
        g.append(Xk(0) - x0)

        # dynamics + cost
        # this could be 0, but then what happens with a non-zero
        # u_init?
        u_prev = Uk(0)
        for k in range(N):
            xk = Xk(k)
            uk = Uk(k)
            x_next = self.c_ode.rk4(xk, uk, dt)
            g.append(Xk(k + 1) - x_next)
            J = J + self.stage_cost(xk, uk, x_goal, u_prev)
            u_prev = uk

        # terminal cost
        J = J + self.terminal_cost(Xk(N), x_goal)

        # optional time penalty
        if sp.optimize_time and sp.time_weight != 0.0:
            J = J + float(sp.time_weight) * (N * dt)

        # pack variables
        if sp.optimize_time:
            w = ca.vertcat(X, U, dt)
            dt_index = (N + 1) * n + N * m
            self.var_slices = {
                "X": slice(0, (N + 1) * n),
                "U": slice((N + 1) * n, (N + 1) * n + N * m),
                "dt": slice(dt_index, dt_index + 1),
            }
        else:
            w = ca.vertcat(X, U)
            self.var_slices = {
                "X": slice(0, (N + 1) * n),
                "U": slice((N + 1) * n, (N + 1) * n + N * m),
            }

        g = ca.vertcat(*g)

        # bounds
        lbx = []
        ubx = []

        # state bounds: |x| <= s_max
        s_max = np.asarray(sp.s_max, dtype=float).reshape(n)
        for _ in range(N + 1):
            lbx.extend((-s_max).tolist())
            ubx.extend((+s_max).tolist())

        # control bounds: |u| <= u_max
        u_max = np.asarray(sp.u_max, dtype=float).reshape(m)
        for _ in range(N):
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
