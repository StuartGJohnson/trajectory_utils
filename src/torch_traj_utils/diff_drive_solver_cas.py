"""
Differential drive robot control trajectory planner.
"""
from torch_traj_utils.cas_solver import CasadiSolver, CasadiSolverParams

from torch_traj_utils.diff_drive_ode import DiffDriveODE
from torch_traj_utils.diff_drive_cas_ode import DiffDriveCasODE
import torch
import casadi as ca
import numpy as np
from typing import Tuple, Callable, Any, List

from torch_traj_utils.scalar_field_interpolator_cas import ScalarFieldInterpolatorCas
from torch_traj_utils.scalar_field_interpolator import SDF

class DiffDriveSolverCas(CasadiSolver):
    sdf_interpolator: ScalarFieldInterpolatorCas
    u_goal: np.ndarray
    u_min: np.ndarray
    u_final: np.ndarray
    s_min: np.ndarray
    s_max: np.ndarray

    def __init__(self, sp:CasadiSolverParams):
        super().__init__(sp=sp)
        c_ode = DiffDriveCasODE()
        t_ode = DiffDriveODE(sp.dt)
        self.setup(c_ode, t_ode)

    def reset_custom(self, s0:np.ndarray, u_goal:np.ndarray, u_final:np.ndarray, u_min:np.ndarray, N:int, sdf: SDF):
        """
        N : int
            The time horizon (N * dt) of the solver.
        u_goal : numpy.ndarray
            The goal controls.
        u_final : numpy.ndarray
            The target control at the final state.
        s0 : numpy.ndarray
            The initial state.
        """
        # prepare for a new solve
        self.s0 = s0
        self.s_goal = np.array([])
        self.N = N
        self.sdf_interpolator = ScalarFieldInterpolatorCas(sdf.sdf, sdf.ox, sdf.oy, sdf.res)
        self.u_final = u_final
        self.u_goal = u_goal
        self.u_min = u_min
        # compute state min and max from the SDF
        x_min = sdf.ox
        x_max = sdf.ox + sdf.x_size
        y_min = sdf.oy
        y_max = sdf.oy + sdf.y_size
        self.s_min = np.array([x_min, y_min, -np.inf])
        self.s_max = np.array([x_max, y_max, np.inf])
        n = self.params.Q.shape[0]
        m = self.params.R.shape[0]
        self.build_problem()

    def stage_cost(self, u: ca.MX, u_goal: ca.MX) -> ca.MX:
        du = u - u_goal
        return ca.mtimes([du.T, ca.DM(self.params.R), du])

    def terminal_cost(self, xN: ca.MX, x_goal: ca.MX) -> ca.MX:
        dx = xN - x_goal
        return ca.mtimes([dx.T, ca.DM(self.params.P), dx])


    def build_problem(self) -> None:
        sp = self.params
        N, n, m = self.N, self.n, self.m

        x0 = ca.DM(self.s0).reshape((n, 1))
        u_goal = ca.DM(self.u_goal).reshape((m, 1))

        # sizes
        nxvars = (N + 1) * n
        nuvars = N * m
        nw = nxvars + nuvars + (1 if sp.optimize_time else 0)

        # one flat decision variable (pure symbolic)
        w = ca.MX.sym("w", nw, 1)

        # views into w
        Xv = w[0:nxvars]  # (nxvars,1)
        Uv = w[nxvars:nxvars + nuvars]  # (nuvars,1)

        # reshape into matrices in *time-major* order: rows are time
        # Build X as (N+1, n) with row k == state at time k
        X = ca.reshape(Xv, n, N + 1).T  # (N+1, n)
        U = ca.reshape(Uv, m, N).T  # (N,   m)

        # dt (either symbolic var or constant DM)
        if sp.optimize_time:
            dt = w[nxvars + nuvars]  # scalar MX (1x1)
        else:
            dt = ca.DM(float(sp.dt))

        def Xk(k: int) -> ca.MX:
            return X[k,:].T

        def Uk(k: int) -> ca.MX:
            return U[k, :].T

        g_dyn = []
        J = 0

        # initial condition
        g_dyn.append(Xk(0)[0:2] - x0[0:2])

        # dynamics + cost
        # this could be 0, but then what happens with a non-zero
        # u_init?
        for k in range(N):
            xk = Xk(k)
            uk = Uk(k)
            x_next = self.c_ode.rk4(xk, uk, dt)
            g_dyn.append(Xk(k + 1) - x_next)
            J = J + self.stage_cost(uk, u_goal)

        g_obs = self.sdf_interpolator.interpolator(X, U)
        g_obs = ca.reshape(g_obs, -1, 1)
        g_dyn = ca.vertcat(*g_dyn)
        g = ca.vertcat(g_dyn, g_obs)
        
        # optional time penalty
        if sp.optimize_time and sp.time_weight != 0.0:
            J = J + float(sp.time_weight) * (N * dt)

        nxvars = int(X.numel())      # (N+1)*n
        nuvars = int(U.numel())
        
        # pack variables
        if sp.optimize_time:
            dt_index = nxvars + nuvars # (N + 1) * n + N * m
            self.var_slices = {
                "X": slice(0, nxvars),
                "U": slice(nxvars, nxvars + nuvars),
                "dt": slice(dt_index, dt_index + 1),
            }
        else:
            self.var_slices = {
                "X": slice(0, nxvars),
                "U": slice(nxvars, nxvars + nuvars),
            }

        # bounds
        lbx = []
        ubx = []

        # state bounds: |x| <= s_max
        s_min = np.asarray(self.s_min, dtype=float).reshape(n)
        s_max = np.asarray(self.s_max, dtype=float).reshape(n)
        for _ in range(N + 1):
            lbx.extend((s_min).tolist())
            ubx.extend((s_max).tolist())

        # control bounds: |u| <= u_max
        u_max = np.asarray(sp.u_max, dtype=float).reshape(m)
        for _ in range(N):
            lbx.extend((-u_max).tolist())
            ubx.extend((+u_max).tolist())

        if sp.optimize_time:
            lbx.append(float(sp.dt_min))
            ubx.append(float(sp.dt_max))

        # equality constraints
        lbg_dyn = [0.0] * int(g_dyn.shape[0])
        ubg_dyn = [0.0] * int(g_dyn.shape[0])
        lbg_obs = [0.0] * int(g_obs.shape[0])
        ubg_obs = [ca.inf] * int(g_obs.shape[0])

        # combine
        lbg = lbg_dyn + lbg_obs
        ubg = ubg_dyn + ubg_obs

        assert int(g_dyn.shape[0]) == len(lbg_dyn) == len(ubg_dyn)
        assert int(g_obs.shape[0]) == len(lbg_obs) == len(ubg_obs)
        assert int(g.shape[0]) == len(lbg) == len(ubg)

        # w is your decision vector
        #gobs_fun = ca.Function("gobs", [w], [g_obs])

        #thingy = ca.jacobian(g_obs, w)
        #print("Jacobian nnz:", thingy.sparsity().nnz())

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


    def initialize_trajectory(self):
        super().initialize_trajectory()
        # not a good idea. this will generally be infeasible - and it is not
        # easy for cvxpy to rescue itself from trajectories that go
        # off the map!
        # for i in range(self.u_init.shape[0]):
        #     self.u_init[i,:] = self.u_goal
