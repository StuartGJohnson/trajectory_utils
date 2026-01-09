"""
Interface for trajectory optimization solvers.
"""
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from typing import Tuple, Callable, Any, List
import time
from dataclasses import dataclass
from torch_traj_utils.torch_ode import TorchODE

@dataclass
class TrajectorySolverParams:
    """the parameters of a general trajectory expert"""
    dt: float
    # terminal state cost
    P: np.ndarray
    # state cost along trajectory
    Q: np.ndarray
    # control cost matrix
    R: np.ndarray
    # control differential cost matrix
    Rd: np.ndarray
    # state max
    s_max: np.ndarray
    # control max
    u_max: np.ndarray
    # max solve time
    max_solve_secs: float


class TrajectorySolver(ABC):

    @abstractmethod
    def get_params(self) -> TrajectorySolverParams:
        raise NotImplementedError()

    @abstractmethod
    def get_ode(self) -> TorchODE:
        raise NotImplementedError()

    @abstractmethod
    def reset(self, s0:np.ndarray, s_goal:np.ndarray, N:int):
        """
        N : int
            The time horizon (N * dt) of the solver.
        s_goal : numpy.ndarray
            The goal state.
        s0 : numpy.ndarray
            The initial state.
        """
        raise NotImplementedError()

    @abstractmethod
    def initialize_trajectory(self):
        raise NotImplementedError()

    @abstractmethod
    def set_trajectory(self, s_init, u_init):
        raise NotImplementedError()

    @abstractmethod
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, float, int]:
        """
        Solve over the time horizon.
        Returns
        -------
        s : numpy.ndarray
            s[k] is the state at time step k
        u : numpy.ndarray
            u[k] is the control at time step k
        J : numpy.ndarray
            J[i] is the SCP cost after the i-th solver iteration
        conv: bool
            convergence state
        """
        raise NotImplementedError()





