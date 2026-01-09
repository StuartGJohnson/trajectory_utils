"""
Cartpole experts.
"""
from torch_traj_utils.cartpole_energy import CartpoleEnergy
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams
from torch_traj_utils.cartpole_solver_force import CartpoleSolverForce
from torch_traj_utils.trajectory_solver import TrajectorySolverParams, TrajectorySolver
from torch_traj_utils.torch_ode import TorchODE
from torch_traj_utils.cartpole_solver_velocity_cas import CartpoleSolverVelocityCas, CasadiSolverParams
import numpy as np

class CartpoleSwingupExpert(TrajectoryExpert):
    ep: CartpoleEnvironmentParams
    sp: TrajectorySolverParams
    solver: TrajectorySolver
    t_ode: TorchODE
    energy: CartpoleEnergy

    def __init__(self, ep: CartpoleEnvironmentParams, solver: TrajectorySolver):
        self.ep = ep
        self.sp = solver.get_params()
        self.solver = solver
        self.t_ode = solver.get_ode()
        self.energy = CartpoleEnergy(ep)

    def trajectory(self, sc: TrajectoryScenario) -> Trajectory:
        """compute a trajectory to the time horizon"""
        N = np.round((sc.T - sc.t0) / self.sp.dt).astype(int) + 1
        self.solver.reset(sc.s0, sc.s_goal, N)
        self.solver.initialize_trajectory()
        s, u, J, conv, status, time, iters = self.solver.solve()
        # update with a rollout
        s,u = self.t_ode.rollout(s, u, N)
        en_tot = self.energy.compute_energy(s)
        en_pole = self.energy.compute_energy_pole(s)
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

