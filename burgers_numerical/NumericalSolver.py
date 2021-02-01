import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error
from util.data_loader import data_loader


class NumericalSolver(ABC):
    def __init__(self, n_spatial: int, n_temporal: int, nu: float = 1/(100*np.pi), **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param nu: The viscosity parameter of the Burgers' equation
        :param kwargs: allows to pass a vector of initial values via the argument u0
        """
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.nu = nu

        self.h = 2/(self.n_spatial-1)  # mesh-size
        self.k = 1/(self.n_temporal-1)  # step-size

        # Discretisation points
        self.x = np.linspace(-1, 1, self.n_spatial)
        self.t = np.linspace(0, 1, self.n_temporal)

        # Set initial conditions: allow to pass a vector containing the IC via kwargs.
        self.u0 = kwargs.get('u0', -np.sin(np.pi * self.x))

        # Solution matrix
        self.u_numerical = np.zeros(shape=(self.n_spatial, self.n_temporal))
        self.u_numerical[:, 0] = self.u0

        # Load exact
        _, _, self.u_exact = data_loader(self.n_spatial, self.n_temporal)

    @abstractmethod
    def time_integrate(self) -> None:
        """
        Time integrates the ODE-system and stores the solution in self.u
        """
        pass

    def get_l2_error(self):
        """
        Computes the error in the L2 norm
        """
        # Option 1: Only on inner points (neglecting initial and boundary conditions)
        # return np.sqrt(1/((self.n_temporal-1)*(self.n_spatial-2))) * np.linalg.norm(
        #     self.u_exact[1:self.n_spatial-1, 1:] - self.u[1:self.n_spatial-1, 1:])
        # Option 2: On all points
        return np.sqrt(1 / (self.n_temporal * self.n_spatial)) * np.linalg.norm(self.u_exact - self.u_numerical)

    def get_mean_squared_error(self):
        """
        Computes the mean squared error
        """
        # Option 1: Only on inner points (neglecting initial and boundary conditions)
        # return mean_squared_error(self.u_exact[1:self.n_spatial-1, 1:], self.u[1:self.n_spatial-1, 1:])
        # Option 2: On all points
        try:
            return mean_squared_error(self.u_exact, self.u_numerical)
        except ValueError:
            return np.inf

    def get_mean_absolute_error(self):
        """
        Computes the mean absolute error
        """
        # Option 1: Only on inner points (neglecting initial and boundary conditions)
        # return mean_absolute_error(self.u_exact[1:self.n_spatial-1, 1:], self.u[1:self.n_spatial-1, 1:])
        # Option 2: On all points
        try:
            return mean_absolute_error(self.u_exact, self.u_numerical)
        except ValueError:
            return np.inf

    def get_l_max_error(self):
        """
        Computes the maximum error on the entire spatio-temporal domain
        """
        return np.amax(abs(self.u_exact - self.u_numerical))
