from numerical_solvers.NumericalSolver import ACNumericalSolver
import scipy.sparse as sparse


class AllenCahnFTCS(ACNumericalSolver):
    def __init__(self, n_spatial, n_temporal, **kwargs):
        """
        :param n_spatial: Number of spatial discretisation points
        :param n_temporal: Number of temporal discretisation points
        :param nu: The viscosity parameter of the Burgers' equation # NOT NEEDED
        :param kwargs: allows to pass a vector of initial values via the argument u0
        """
        super().__init__(n_spatial, n_temporal, **kwargs)

        # Design matrices
        self.c = 0.0001 / (self.h ** 2)  # CHANGED
        self.A = sparse.diags([self.k * self.c, 1 + self.k * (5 - 2 * self.c), self.k * self.c], [0, 1, 2],
                              shape=(self.n_spatial - 2, self.n_spatial)).toarray()

    def time_integrate(self) -> None:
        """
        Forward Euler time integrator
        """
        for n in range(self.n_temporal - 1):
            u_n = self.u_numerical[:, n]
            u_n_inner = u_n[1:self.n_spatial - 1]
            self.u_numerical[1:self.n_spatial - 1, n + 1] = self.A.dot(u_n) - 5 * self.k * (u_n_inner ** 3)
