"""
Cartpole experts.
"""

from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams
from torch_traj_utils.cartpole_solver_force import CartpoleSolverForce
import numpy as np

class CartpoleVelocitySwingupExpert(TrajectoryExpert):
    ep: CartpoleEnvironmentParams
    sp: SolverParams
    solver: CartpoleSolverVelocity

    def __init__(self, ep: CartpoleEnvironmentParams, sp: SolverParams):
        self.ep = ep
        self.sp = sp
        self.solver = CartpoleSolverVelocity(sp=self.sp, ep=self.ep)

    def trajectory(self, sc: TrajectoryScenario) -> Trajectory:
        """compute a trajectory to the time horizon"""
        t = np.arange(sc.t0, sc.T + self.sp.dt, self.sp.dt)
        N = t.size - 1
        self.solver.reset(sc.s0, sc.s_goal, N)
        self.solver.initialize_trajectory()
        s, u, J, conv, status, time, iters = self.solver.solve()
        # update with a rollout
        s,u = self.solver.rollout(s, u)
        en_tot = self.solver.energy.compute_energy(s)
        en_pole = self.solver.energy.compute_energy_pole(s)
        return Trajectory(sc=sc,
                          s=s,
                          u=u,
                          J=J,
                          dt=self.sp.dt,
                          N=N,
                          conv=conv,
                          status=status,
                          energy1=en_tot,
                          energy2=en_pole,
                          time=time,
                          iters=iters)


class CartpoleForceSwingupExpert(TrajectoryExpert):
    ep: CartpoleEnvironmentParams
    sp: SolverParams
    solver: CartpoleSolverForce

    def __init__(self, ep: CartpoleEnvironmentParams, sp: SolverParams):
        self.ep = ep
        self.sp = sp
        self.solver = CartpoleSolverForce(sp=self.sp, ep=self.ep)

    def trajectory(self, sc: TrajectoryScenario) -> Trajectory:
        """compute a trajectory to the time horizon"""
        t = np.arange(sc.t0, sc.T + self.sp.dt, self.sp.dt)
        N = t.size - 1
        self.solver.reset(sc.s0, sc.s_goal, N)
        self.solver.initialize_trajectory()
        s, u, J, conv, status, time, iters = self.solver.solve()
        # update with a rollout
        s,u = self.solver.rollout(s, u)
        en_tot = self.solver.energy.compute_energy(s)
        en_pole = self.solver.energy.compute_energy_pole(s)
        return Trajectory(sc=sc,
                          s=s,
                          u=u,
                          J=J,
                          dt=self.sp.dt,
                          N=N,
                          conv=conv,
                          status=status,
                          energy1=en_tot,
                          energy2=en_pole,
                          time=time,
                          iters=iters)
