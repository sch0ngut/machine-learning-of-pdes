import numpy as np
import scipy.sparse as sparse
from burgers_numerical.NumericalSolver import NumericalSolver


class HopfCole(NumericalSolver):
    def __init__(self, num_spatial, num_temporal, nu=1/(100*np.pi), **kwargs):
        super().__init__(num_spatial, num_temporal, nu, **kwargs)

        # Design matrix
        self.c = self.nu/(self.h**2)
        self.A = sparse.diags([self.c, -2*self.c, self.c], [-1, 0, 1], shape=(self.H+1, self.H+1)).toarray()
        # BC: Alternative 1
        self.A[0, 1] = 2*self.c
        self.A[self.H, self.H-1] = 2*self.c
        # BC: Alternative 2
        # self.A[0, self.H - 1] = 1
        # self.A[self.H, 1] = 1
        # BC: Alternative 3
        # self.A[0, 0] = 0
        # self.A[0, 1] = 0
        # self.A[self.H, self.H-1] = 0
        # self.A[self.H, self.H] = 0

        # IC: Heat equation
        theta0 = np.exp(+1 / (2 * self.nu * np.pi) * (1 - np.cos(np.pi * self.x)))

        # Solution: Heat equation
        self.Theta = np.empty(shape=(self.H+1, self.K+1))
        self.Theta[:, 0] = theta0

    def time_integrate(self):
        for n in range(0, self.K):
            #             if k % 100 == 0:
            #                 print(f'k = {k}')
            #             Theta[:, k] = A.dot(Theta[:, k-1])
            k1 = self.A.dot(self.Theta[:, n])
            k2 = self.A.dot(self.Theta[:, n] + self.k * k1 / 2)
            k3 = self.A.dot(self.Theta[:, n] + self.k * k2 / 2)
            k4 = self.A.dot(self.Theta[:, n] + self.k * k3)

            self.Theta[:, n+1] = self.Theta[:, n] + 1 / 6 * self.k * (k1 + 2 * k2 + 2 * k3 + k4)

        # Retransform to Burgers
        for j in range(1, self.H):
            self.U[j, :] = -self.nu / self.h * (self.Theta[j + 1, :] - self.Theta[j - 1, :]) / self.Theta[j, :]