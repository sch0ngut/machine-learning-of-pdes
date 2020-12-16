import numpy as np
import scipy.io
from abc import ABC, abstractmethod


class NumericalSolver(ABC):
    def __init__(self, num_spatial, num_temporal, nu=1/(100*np.pi), **kwargs):
        self.H = num_spatial  # H+1: Number of spatial discretisation points
        self.K = num_temporal  # K+1: Number of temporal discretisation points
        self.nu = nu

        self.h = 2/self.H  # mesh-size
        self.k = 1/self.K  # step-size

        # Discretisation points
        self.x = np.linspace(-1, 1, self.H+1)
        self.t = np.linspace(0, 1, self.K+1)

        # Set IC: allow to pass a vector containing the IC via kwargs.
        self.u0 = kwargs.get('u0', -np.sin(np.pi * self.x))

        # Solution matrix
        self.U = np.zeros(shape=(self.H + 1, self.K + 1))
        self.U[:, 0] = self.u0

        # Load exact
        data = scipy.io.loadmat(
            f'burgers_exact/solutions/burgers_exact_N_t={self.K+1}_N_x={self.H+1}.mat')
        self.u_exact = np.real(data['mysol'])

    @abstractmethod
    def time_integrate(self):
        pass

    def get_l2_error(self):
        return np.sqrt(self.k * self.h) * np.linalg.norm(self.u_exact[1:self.H-1, 1:-1]-self.U[1:self.H-1, 1:-1])
        # return 1/np.sqrt((self.K+1)*(self.H+1)) * np.linalg.norm(self.u_exact - self.U)  # Notebook way: all points

    def get_l_max_error(self):
        return np.amax(abs(self.u_exact - self.U))
