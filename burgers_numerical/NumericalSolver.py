import numpy as np
from abc import ABC, abstractmethod

from util.data_loader import data_loader


class NumericalSolver(ABC):
    def __init__(self, n_spatial, n_temporal, nu=1/(100*np.pi), **kwargs):
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.nu = nu

        self.h = 2/(self.n_spatial-1)  # mesh-size
        self.k = 1/(self.n_temporal-1)  # step-size

        # Discretisation points
        self.x = np.linspace(-1, 1, self.n_spatial)
        self.t = np.linspace(0, 1, self.n_temporal)

        # Set IC: allow to pass a vector containing the IC via kwargs.
        self.u0 = kwargs.get('u0', -np.sin(np.pi * self.x))

        # Solution matrix
        self.u = np.zeros(shape=(self.n_spatial, self.n_temporal))
        self.u[:, 0] = self.u0

        # Load exact
        self.u_exact = data_loader(self.n_spatial, self.n_temporal)

    @abstractmethod
    def time_integrate(self):
        pass

    def get_l2_error(self):
        return np.sqrt(self.k * self.h) * np.linalg.norm(self.u_exact[1:self.n_spatial-2, 1:-1] -
                                                         self.u[1:self.n_spatial-2, 1:-1])

    def get_l_max_error(self):
        return np.amax(abs(self.u_exact - self.u))
