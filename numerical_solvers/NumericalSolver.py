import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error

from util.data_loader import burgers_data_loader, allen_cahn_data_loader


class NumericalSolver(ABC):
    def __init__(self, n_spatial: int, n_temporal: int, **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        """
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal

        self.h = 2/(self.n_spatial-1)  # mesh-size
        self.k = 1/(self.n_temporal-1)  # step-size

        # Discretisation points
        self.x = np.linspace(-1, 1, self.n_spatial)
        self.t = np.linspace(0, 1, self.n_temporal)

        # Initialise exact and numerical solution
        self.u_numerical = np.zeros(shape=(self.n_spatial, self.n_temporal))
        self.u_exact = np.zeros(shape=(self.n_spatial, self.n_temporal))

    @abstractmethod
    def time_integrate(self) -> None:
        """
        Time integrates the ODE-system and stores the solution in self.u_exact
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


class NumericalSolverBurgers(NumericalSolver, ABC):
    def __init__(self, n_spatial, n_temporal, nu: float = 1/(100*np.pi), **kwargs):
        super().__init__(n_spatial, n_temporal, **kwargs)
        self.nu = nu

        # Set initial conditions: allow to pass a vector containing the IC via kwargs.
        self.u0 = kwargs.get('u0', -np.sin(np.pi * self.x))

        # Solution matrix including initial and boundary data
        self.u_numerical = np.zeros(shape=(self.n_spatial, self.n_temporal))
        self.u_numerical[:, 0] = self.u0

        # Load exact
        _, _, self.u_exact = burgers_data_loader(self.n_spatial, self.n_temporal)


class NumericalSolverAC(NumericalSolver, ABC):
    def __init__(self, n_spatial, n_temporal, **kwargs):
        super().__init__(n_spatial, n_temporal, **kwargs)

        # Set initial conditions: allow to pass a vector containing the IC via kwargs.
        self.u0 = kwargs.get('u0', self.x ** 2 * np.cos(np.pi * self.x))

        # Solution matrix including initial and boundary data
        self.u_numerical = np.zeros(shape=(self.n_spatial, self.n_temporal))
        self.u_numerical[0, :] = -np.ones(n_temporal)
        self.u_numerical[-1, :] = -np.ones(n_temporal)
        self.u_numerical[:, 0] = self.u0

        # Load exact
        _, _, self.u_exact = allen_cahn_data_loader()
