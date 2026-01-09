"""
Interface for trajectory simulators (or integrators, if you will).
This (partially) abstract class is a casadi wrapper of the system dynamics.
"""
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from typing import Tuple, Callable, Any, List
import casadi as ca

class CasadiODE(ABC):

    def __init__(self):
        return

    def rk4(self, x: ca.MX, u: ca.MX, dt: ca.MX) -> ca.MX:
        f = self.ode
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    @abstractmethod
    def ode(self, x: ca.MX, u: ca.MX) -> ca.MX:
        raise NotImplementedError
