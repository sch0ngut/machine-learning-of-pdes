import numpy as np
from scipy import linalg

from burgers_numerical.NumericalSolver import NumericalSolver


class Upwind(NumericalSolver):
    def __init__(self, num_spatial, num_temporal, order=1, nu=1/(100*np.pi), **kwargs):
        super().__init__(num_spatial, num_temporal, nu, **kwargs)
        self.order = order

        # Design matrix
        self.c = (self.k * self.nu) / (self.h ** 2)
        upper = np.concatenate((np.zeros(1), np.repeat(-self.c, self.H - 2)))
        main = np.repeat(1 + 2 * self.c, self.H - 1)
        lower = np.concatenate((np.repeat(-self.c, self.H - 2), np.zeros(1)))
        self.A = np.array([upper, main, lower])
        self.A_inv_band = linalg.solve_banded((1, 1,), self.A, np.eye(self.H - 1))

    def convection_vec(self, u):
        """
        Calculates the upwind term at a given time point

        :param u: Numerical solution for the given time point
        :return: the upwind term as vector
        """
        n = u.size - 2  # Only required at inner points
        u_tilde = np.zeros(n)

        if self.order == 1:
            # Alternative 1: makes use of the fact that solution is positive in [-1,0) and negative in (0,1]
            u_tilde[0:int((n-1)/2)] = u[1:int((n-1)/2)+1] * (u[1:int((n-1)/2)+1] - u[0:int((n-1)/2)])
            u_tilde[int((n+1)/2):n] = u[int((n+1)/2)+1:n+1] * (u[int((n+1)/2)+2:n+2] - u[int((n+1)/2)+1:n+1])

            # Alternative 2: does not make use of the above -> slower
            # for i in range(n):
            #     if u[i + 1] < 0:
            #         u_tilde[i] = u[i + 1] * (u[i + 2] - u[i + 1])
            #     else:
            #         u_tilde[i] = u[i + 1] * (u[i + 1] - u[i])

        if self.order == 2:

            # Alternative 1: makes use of the fact that solution is positive in [-1,0) and negative in (0,1]
            u_tilde[0] = u[1] * (u[1] - u[0])
            u_tilde[1:int((n - 1) / 2)] = u[2:int((n - 1) / 2)+1] * (3 * u[2:int((n - 1) / 2)+1] - 4 * u[1:int((n - 1) / 2)] + u[0:int((n - 1) / 2)-1])
            u_tilde[int((n + 1) / 2):n-1] = u[int((n + 1) / 2)+1:n] * (-u[int((n + 1) / 2)+3:n+2] + 4 * u[int((n + 1) / 2)+2:n+1] - 3 * u[int((n + 1) / 2)+1:n])
            u_tilde[n - 1] = u[n] * (u[n + 1] - u[n])
            u_tilde = u_tilde / 2

            # Alternative 2: does not make use of the above -> slower
            # if u[1] < 0:
            #     u_tilde[0] = u[1] * (u[2] - u[1])
            # else:
            #     u_tilde[0] = u[1] * (u[1] - u[0])
            # if u[n] < 0:
            #     u_tilde[n - 1] = u[n] * (u[n + 1] - u[n])
            # else:
            #     u_tilde[n - 1] = u[n] * (u[n] - u[n - 1])
            # for i in range(1, n - 1):
            #     if u[i + 1] < 0:
            #         u_tilde[i] = u[i + 1] * (-u[i + 3] + 4 * u[i + 2] - 3 * u[i + 1])
            #     else:
            #         u_tilde[i] = u[i + 1] * (3 * u[i + 1] - 4 * u[i] + u[i - 1])
            # u_tilde = u_tilde/2

        return u_tilde / self.h

    def time_integrate(self):
        for n in range(self.K):
            u_n = self.U[:, n]
            u_n_tilde = self.convection_vec(u_n)
            self.U[1:self.H, n+1] = self.A_inv_band.dot(u_n[1:self.H] - self.k * u_n_tilde)
            self.U[0, n+1] = self.U[0, n - 1]
            self.U[self.H, n+1] = self.U[self.H, n - 1]
