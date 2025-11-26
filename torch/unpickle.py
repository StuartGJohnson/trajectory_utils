"""
Cart velocity (servo) controlled cartpole.
"""

from trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from plot_trajectory import plot_trajectory
import numpy as np
import pickle

def main():
    # unpickle!
    #pickle_fest = [env_params, solver_params, traj]
    with open("cartpole_velocity.pkl","rb") as f:
        pickle_fest = pickle.load(f)

    env_params = pickle_fest[0]
    traj = pickle_fest[2]
    # plot!
    plot_trajectory(env_params=env_params,
                    traj=traj, filename_base="cartpole_velocity_unpickle",
                    animate=True)


if __name__ == "__main__":
    main()