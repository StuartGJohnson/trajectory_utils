
from torch_traj_utils.trajectory import TrajectoryScenario
from torch_traj_utils.cartpole_solver_velocity_cas import CartpoleSolverVelocityCas, CasadiSolverParams, CartpoleEnvironmentParams
from torch_traj_utils.cartpole_expert import CartpoleSwingupExpert
from torch_traj_utils.plot_trajectory import plot_trajectory
import numpy as np
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

    solver_params = CasadiSolverParams(
        dt=0.02,
        Q=np.diag([100, 1.0, 0.5, 0.5]),
        R=0.5*np.diag([50.0]),
        Rd=0.5*np.diag([50.0]),
        P=np.diag([200.0, 2000.0, 1000.0, 1000.0]),
        #P=np.diag([1.0, 1.0, 1.0, 1.0]),
        s_max=np.array([0.44 / 2.0, 1.5*np.pi, 0.8, 5*np.pi])[None, :],
        u_max=np.array([0.8]),
        optimize_time=False,     # set False to keep fixed dt
        dt_min=0.001,
        dt_max=0.02,
        time_weight=0.0,
        max_solve_secs=-1.0,
        ipopt_max_iter = 2000,
        ipopt_tol = 1e-6,
        ipopt_print_level = 0,
        ipopt_sb = "yes",
        print_time = False
    )

    s0 = np.array([0.0, 0.0, 0.0, 0.0])
    s_goal = np.array([0.0, np.pi, 0.0, 0.0])

    scenario = TrajectoryScenario(s_goal=s_goal, s0=s0, t0=0.0, T=4.0)

    solver = CartpoleSolverVelocityCas(ep=env_params, sp=solver_params)

    expert = CartpoleSwingupExpert(ep=env_params, solver=solver)

    traj = expert.trajectory(scenario)

    plot_trajectory(solver_params=solver_params,
                    env_params=env_params,
                    traj=traj, filename_base="cartpole_velocity_cas",
                    animate=True)

    # pickle!
    pickle_fest = [env_params, solver_params, traj]
    with open("cartpole_velocity_cas.pkl", "wb") as f:
        pickle.dump(pickle_fest, f)


if __name__ == "__main__":
    main()