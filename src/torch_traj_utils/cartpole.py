import numpy as np
from dataclasses import dataclass
from torch_traj_utils.trajectory import Trajectory, TrajectoryScenario

@dataclass
class CartpoleEnvironmentParams:
    """The physical parameters and constraints of the model."""
    pole_length: float
    pole_mass: float
    cart_mass: float
    cart_length: float
    track_length: float
    max_cart_force: float
    max_cart_speed: float
    # tracking time constant of the cart speed controller
    cart_tau: float
    # these may belong here, but they have ended up in the solver params
    # u_max: np.ndarray
    # s_max: np.ndarray
    # state and control dimension
    n: int
    m: int
