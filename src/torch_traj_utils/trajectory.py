import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryScenario:
    """the starting state and goal of an expert trajectory"""
    s_goal: np.ndarray
    s0: np.ndarray
    # starting time of a policy
    t0: float
    # goal time of trajectory
    T: float

@dataclass
class Trajectory:
    sc: TrajectoryScenario
    s: np.ndarray
    u: np.ndarray
    J: np.ndarray
    dt: float
    N: int
    conv: bool
    status: str
    energy1: np.ndarray
    energy2: np.ndarray
    time: float
    iters: int

class TrajectoryExpert:
    def trajectory(self, sc: TrajectoryScenario) -> Trajectory:
        """compute a trajectory to the time horizon"""
        raise NotImplementedError()