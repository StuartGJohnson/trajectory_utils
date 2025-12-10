"""
Force (on cart) controlled cartpole. A specialization of SCPSolver.
"""

from torch_traj_utils.trajectory import TrajectoryScenario
from torch_traj_utils.cartpole_solver_velocity import CartpoleEnvironmentParams, SolverParams
from torch_traj_utils.cartpole_expert import CartpoleForceSwingupExpert
from torch_traj_utils.plot_trajectory import plot_trajectory
from torch_traj_utils.cartpole_solver_force import CartpoleSolverForce

from torch_traj_utils.animate_cartpole import animate_cartpole
import matplotlib.pyplot as plt
import numpy as np
from torch_traj_utils.cartpole_solver_velocity import CartpoleEnvironmentParams, SolverParams
import pickle

def main():
    env_params = CartpoleEnvironmentParams(pole_length=0.395,
                            pole_mass=0.087,
                            cart_mass=0.230,
                            cart_length=0.044,
                            track_length=0.44,
                            max_cart_force=1.77,
                            max_cart_speed=0.8,
                            cart_tau=0.25,
                            n=4,
                            m=1)

    solver_params = SolverParams(dt=0.05,
                              P=1e3 * np.eye(4),
                              Q=np.diag([10, 2, 1, 0.25]), #Q = np.diag([1e-2, 1.0, 1e-3, 1e-3]) (quadratic cost)
                              R=0.001 * np.eye(1),
                              rho=0.05,
                              rho_u=0.02,
                              eps=0.005,
                              cvxpy_eps=1e-4,
                              max_iters=10000,
                              u_max=np.array([1.77]),
                              s_max=np.array([0.44 / 2.0, 1000, 0.8, 1000])[None, :])

    # goal state: pole upright
    s_goal = np.array([0.0, np.pi, 0.0, 0.0])
    # start state: pole down-ish
    s0 = np.array([0.0, 0.0, 0.0, 0.0])

    scenario = TrajectoryScenario(s_goal=s_goal, s0=s0, t0=0.0, T=4.0)

    expert = CartpoleForceSwingupExpert(ep=env_params, sp=solver_params)

    traj = expert.trajectory(scenario)

    plot_trajectory(solver_params=solver_params, env_params=env_params,
                    traj=traj, filename_base="cartpole_force",
                    animate=True)

    # pickle!
    pickle_fest = [env_params, solver_params, traj]
    with open("cartpole_force.pkl", "wb") as f:
        pickle.dump(pickle_fest, f)

if __name__ == "__main__":
    main()