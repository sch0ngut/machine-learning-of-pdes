import numpy as np
import scipy.sparse as sparse
from numerical_solvers.NumericalSolver import BurgersNumericalSolver


class BurgersHopfCole(BurgersNumericalSolver):
    def __init__(self, n_spatial, n_temporal, nu=1/(100*np.pi), **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param nu: The viscosity parameter of the Burgers' equation
        :param kwargs:
            - u0: allow to pass a vector containing the initial condition. Should have length=n_spatial
        """
        super().__init__(n_spatial, n_temporal, nu, **kwargs)

        # Design matrix: Heat equation
        self.c = self.nu/(self.h**2)
        self.A = sparse.diags([self.c, -2*self.c, self.c], [-1, 0, 1], shape=(self.n_spatial, self.n_spatial)).toarray()
        # Introduce boundary conditions: Heat equation
        self.A[0, 1] = 2*self.c
        self.A[self.n_spatial-1, self.n_spatial-2] = 2*self.c

        # Initial conditions: Heat equation
        theta0 = np.exp(+1 / (2 * self.nu * np.pi) * (1 - np.cos(np.pi * self.x)))

        # Solution matrix: Heat equation
        self.theta = np.empty(shape=(self.n_spatial, self.n_temporal))
        self.theta[:, 0] = theta0

    def time_integrate(self) -> None:
        """
        4th order Runge-Kutta time integrator
        """
        # Solve heat equation
        for n in range(0, self.n_temporal-1):
            k1 = self.A.dot(self.theta[:, n])
            k2 = self.A.dot(self.theta[:, n] + self.k * k1 / 2)
            k3 = self.A.dot(self.theta[:, n] + self.k * k2 / 2)
            k4 = self.A.dot(self.theta[:, n] + self.k * k3)

            self.theta[:, n+1] = self.theta[:, n] + 1 / 6 * self.k * (k1 + 2 * k2 + 2 * k3 + k4)

        # Re-transform to Burgers equation
        for j in range(1, self.n_spatial-1):
            self.u_numerical[j, :] = -self.nu / self.h * (self.theta[j + 1, :] -
                                                          self.theta[j - 1, :]) / self.theta[j, :]
