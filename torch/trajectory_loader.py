
from trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

#def convert_states(state_opt: np.ndarray, state_ml: np.ndarray) -> np.ndarray:

def main(tdir: str):
    good_data = 0
    num_data = 0
    # use limits from the trajectory solver for normalization
    state_normalization = [0.22, 1, 0.8, 5*np.pi]
    action_normalization = 0.8
    all_states = []
    all_actions = []
    for file in os.listdir(tdir):
        file = os.path.join(tdir, file)
        # unpickle!
        #pickle_fest = [env_params, solver_params, traj]
        with open(file,"rb") as f:
            pickle_fest = pickle.load(f)
        traj: Trajectory = pickle_fest[2]
        num_data += 1
        if traj.conv:
            good_data += 1
            states = traj.s
            actions = traj.u
            norm_states = states / state_normalization
            norm_action = actions / action_normalization
            # convert states to sin,cos format
            state_dim = norm_states.shape[1]
            n_states = norm_states.shape[0]
            ml_states = np.zeros((n_states-1, state_dim+1))
            ml_states[:,0] = norm_states[1:,0]
            ml_states[:,1] = np.cos(norm_states[1:,1])
            ml_states[:,2] = np.sin(norm_states[1:,1])
            ml_states[:, 3] = norm_states[1:, 2]
            ml_states[:, 4] = norm_states[1:, 3]
            all_states.append(ml_states)
            all_actions.append(norm_action)
    all_states = np.concatenate(all_states,0)
    all_actions = np.concatenate(all_actions, 0)
    fname = tdir + ".npz"
    np.savez(fname,first_array=all_states, second_array=all_actions)
    print(good_data/num_data)
    plt.figure()
    plt.plot(all_states,'+')
    plt.show()
    plt.figure()
    plt.plot(all_actions,'+')
    plt.show()


if __name__ == "__main__":
    main("trajectories_big_1")