import numpy as np
from scp_solver import SolverParams
from cartpole import CartpoleEnvironmentParams
from trajectory import Trajectory,TrajectoryScenario
import matplotlib.pyplot as plt
from animate_cartpole import animate_cartpole

def plot_trajectory(solver_params: SolverParams, env_params: CartpoleEnvironmentParams, traj: Trajectory, filename_base: str, animate=False):
    # Plot state and control trajectories
    t = np.arange(traj.sc.t0, traj.sc.T + traj.dt, traj.dt)
    N = t.size - 1
    fig, ax = plt.subplots(2, env_params.n, dpi=150, figsize=(11, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    labels_s = (r"$x(t)$", r"$\theta(t)$", r"$\dot{x}(t)$", r"$\dot{\theta}(t)$")
    labels_u = (r"$u(t)$",)
    for i in range(env_params.n):
        ax[0,i].plot(t, traj.s[:, i])
        ax[0,i].axhline(traj.sc.s_goal[i], linestyle="--", color="tab:orange")
        if solver_params.s_max[0,i] < 20:
            ax[0,i].axhline(solver_params.s_max[0,i], linestyle="--", color="tab:orange")
            ax[0,i].axhline(-solver_params.s_max[0,i], linestyle="--", color="tab:orange")
        ax[0,i].set_xlabel(r"$t$")
        ax[0,i].set_ylabel(labels_s[i])
    for i in range(env_params.m):
        ax[1,i].plot(t[:-1], traj.u[:, i])
        ax[1,i].axhline(solver_params.u_max, linestyle="--", color="tab:orange")
        ax[1,i].axhline(-solver_params.u_max, linestyle="--", color="tab:orange")
        ax[1,i].set_xlabel(r"$t$")
        ax[1,i].set_ylabel(labels_u[i])
    # add the trajectory plots
    xvec = traj.s[:, 0] + np.sin(traj.s[:, 1])*env_params.pole_length
    yvec = - np.cos(traj.s[:, 1])*env_params.pole_length
    ax[1, 1].plot(xvec, yvec)
    ax[1, 1].set_xlabel("x")
    ax[1, 1].set_ylabel("y")
    # add the energy plot
    ax[1, 2].plot(t, traj.energy1)
    #ax[0, i].axhline(s_goal[i], linestyle="--", color="tab:orange")
    ax[1, 2].set_xlabel(r"$t$")
    ax[1, 2].set_ylabel("E[Total]")
    # add the other energy plot
    ax[1, 3].plot(t, traj.energy2)
    ax[1, 3].plot(t[:-1], traj.u[:, 0]/6)
    #ax[0, i].axhline(s_goal[i], linestyle="--", color="tab:orange")
    ax[1, 3].set_xlabel(r"$t$")
    ax[1, 3].set_ylabel("E[pole] w/ u")
    # cleanup
    #fig.delaxes(ax[1, 3])
    if filename_base != "":
        fimg = filename_base + "_state.png"
        plt.savefig(fimg, bbox_inches="tight")
    plt.show()

    if len(traj.J) > 0:
        # Plot cost history over SCP iterations
        fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 5))
        ax.semilogy(traj.J)
        ax.set_xlabel(r"SCP iteration $i$")
        ax.set_ylabel(r"SCP cost $J(\bar{x}^{(i)}, \bar{u}^{(i)})$")
        if filename_base != "":
            fimg = filename_base + "_cost.png"
            plt.savefig(fimg, bbox_inches="tight")
        plt.show()

    # Animate the solution
    if animate and filename_base != "":
        fig, ani = animate_cartpole(t, traj.s[:, 0], traj.s[:, 1], env_params.pole_length, env_params.cart_length)
        fimg = filename_base + ".gif"
        ani.save(fimg, writer="ffmpeg")