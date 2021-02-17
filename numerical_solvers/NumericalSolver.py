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

    def get_l2_error(self) -> np.float:
        """
        Computes the error in the L2 norm
        """
        return np.sqrt(1 / (self.n_temporal * self.n_spatial)) * np.linalg.norm(self.u_exact - self.u_numerical)

    def get_mean_squared_error(self) -> np.float:
        """
        Computes the mean squared error
        """
        try:
            return mean_squared_error(self.u_exact, self.u_numerical)
        except ValueError:
            return np.inf

    def get_mean_absolute_error(self) -> np.float:
        """
        Computes the mean absolute error
        """
        try:
            return mean_absolute_error(self.u_exact, self.u_numerical)
        except ValueError:
            return np.inf

    def get_l_max_error(self) -> np.float:
        """
        Computes the maximum error on the entire spatio-temporal domain
        """
        return np.amax(abs(self.u_exact - self.u_numerical))


class BurgersNumericalSolver(NumericalSolver, ABC):
    def __init__(self, n_spatial, n_temporal, nu: float = 1/(100*np.pi), **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param nu: viscosity parameter
        :param kwargs:
            - u0: allow to pass a vector containing the initial condition. Should have length=n_spatial
        """
        super().__init__(n_spatial, n_temporal, **kwargs)
        self.nu = nu

        # Load initial condition
        self.u0 = kwargs.get('u0', -np.sin(np.pi * self.x))

        # Set initial condition in solution matrix. Boundary = 0 already included
        self.u_numerical[:, 0] = self.u0

        # Load exact solution
        _, _, self.u_exact = burgers_data_loader(self.n_spatial, self.n_temporal)


class ACNumericalSolver(NumericalSolver, ABC):
    def __init__(self, n_spatial, n_temporal, **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param kwargs:
            - u0: allow to pass a vector containing the initial condition. Should have length=n_spatial
        """
        super().__init__(n_spatial, n_temporal, **kwargs)

        # Load initial condition
        self.u0 = kwargs.get('u0', self.x ** 2 * np.cos(np.pi * self.x))

        # Include initial and boundary data in solution matrix
        self.u_numerical[0, :] = -np.ones(n_temporal)
        self.u_numerical[-1, :] = -np.ones(n_temporal)
        self.u_numerical[:, 0] = self.u0

        # Load exact
        _, _, self.u_exact = allen_cahn_data_loader()
