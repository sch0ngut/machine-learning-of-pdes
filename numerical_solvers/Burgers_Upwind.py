import numpy as np
from scipy import linalg
from numerical_solvers.NumericalSolver import BurgersNumericalSolver


class BurgersUpwind(BurgersNumericalSolver):
    def __init__(self, n_spatial: int, n_temporal: int, order: int = 1, nu: float = 1/(100*np.pi), **kwargs) -> None:
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param order: The order of the upwind scheme. Should be either 1 or 2
        :param nu: The viscosity parameter of the Burgers' equation
        :param kwargs:
            - u0: allow to pass a vector containing the initial condition. Should have length=n_spatial
        """
        super().__init__(n_spatial, n_temporal, nu, **kwargs)
        self.order = order

        # Design matrix
        self.c = (self.k * self.nu) / (self.h ** 2)
        upper = np.concatenate((np.zeros(1), np.repeat(-self.c, self.n_spatial - 3)))
        main = np.repeat(1 + 2 * self.c, self.n_spatial - 2)
        lower = np.concatenate((np.repeat(-self.c, self.n_spatial - 3), np.zeros(1)))
        self.A = np.array([upper, main, lower])
        self.A_inv_band = linalg.solve_banded((1, 1,), self.A, np.eye(self.n_spatial - 2))

    def convection_vec(self, u) -> np.array:
        """
        Calculates the upwind term at a given time point

        :param u: Numerical solution for the given time point
        :return: Upwind term as vector
        """
        n = u.size - 2  # Only required at inner points
        u_tilde = np.zeros(n)

        if self.order == 1:
            # Make use of the fact that solution is positive in [-1,0) and negative in (0,1]
            u_tilde[0:int((n-1)/2)] = u[1:int((n-1)/2)+1] * (u[1:int((n-1)/2)+1] - u[0:int((n-1)/2)])
            u_tilde[int((n+1)/2):n] = u[int((n+1)/2)+1:n+1] * (u[int((n+1)/2)+2:n+2] - u[int((n+1)/2)+1:n+1])

        if self.order == 2:
            # Make use of the fact that solution is positive in [-1,0) and negative in (0,1]
            u_tilde[0] = u[1] * (u[1] - u[0])
            u_tilde[1:int((n - 1) / 2)] = u[2:int((n - 1) / 2)+1] * (3 * u[2:int((n - 1) / 2)+1] -
                                                                     4 * u[1:int((n - 1) / 2)] +
                                                                     u[0:int((n - 1) / 2)-1])
            u_tilde[int((n + 1) / 2):n-1] = u[int((n + 1) / 2)+1:n] * (-u[int((n + 1) / 2)+3:n+2] +
                                                                       4 * u[int((n + 1) / 2)+2:n+1] -
                                                                       3 * u[int((n + 1) / 2)+1:n])
            u_tilde[n - 1] = u[n] * (u[n + 1] - u[n])
            u_tilde = u_tilde / 2

        return u_tilde / self.h

    def time_integrate(self) -> None:
        """
        Forward Euler time integrator
        """
        for n in range(self.n_temporal - 1):
            u_n = self.u_numerical[:, n]
            u_n_tilde = self.convection_vec(u_n)
            self.u_numerical[1:self.n_spatial-1, n+1] = self.A_inv_band.dot(u_n[1:self.n_spatial-1] -
                                                                            self.k * u_n_tilde)
            self.u_numerical[0, n+1] = self.u_numerical[0, n - 1]
            self.u_numerical[self.n_spatial-1, n+1] = self.u_numerical[self.n_spatial-1, n - 1]
