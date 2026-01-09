"""
Interface for trajectory simulators (or integrators, if you will).
This (partially) abstract class is a pytorch wrapper of the system dynamics.
"""
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from typing import Tuple, Callable, Any, List
import torch

class TorchODE(ABC):
    fd: Callable

    def __init__(self, dt:float):
        self.dt = dt

    def setup(self, dt:float) -> Callable:
        # these virtual-method calls do not belong in the constructor!
        self.fd = self.build_fd_ode(dt)
        return self.fd

    def discretize(self, f:Callable, dt: float) -> Callable:
        """
        f: (s, u) -> ds/dt   (both torch tensors; supports batching)
        returns fd(s,u) that maps to next state with Runge-Kutta 4th order integration.
            That is, s function describing the discrete-time dynamics, such that
            `s[k+1] = fd(s[k], u[k])`.
        See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods .
        """
        def integrator(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            k1 = dt * f(s, u)
            k2 = dt * f(s + 0.5 * k1, u)
            k3 = dt * f(s + 0.5 * k2, u)
            k4 = dt * f(s + k3, u)
            return s + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return integrator

    def build_fd_ode(self, dt: float) -> Callable:
        return self.discretize(self.ode, dt)

    def rollout(self, s: np.ndarray, u: np.ndarray, N:int) -> Tuple[np.ndarray, np.ndarray]:
        S = torch.as_tensor(s, dtype=torch.float64)
        U = torch.as_tensor(u, dtype=torch.float64)
        for k in range(N):
            S[k + 1] = self.fd(S[k], U[k])
        s = S.detach().cpu().numpy()
        u = U.detach().cpu().numpy()
        return s, u

    def step(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # single step of ode - for closed loop simulation with pytorch objects
        # (e.g.) from NN predictions
        return self.fd(s, u)

    @abstractmethod
    def ode(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        returns
            A tensor ds/dt describing the continuous-time dynamics: `ds/dt = f(s, u)`.
        """
        raise NotImplementedError()