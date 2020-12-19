from termcolor import cprint
import scipy
import numpy as np
import sys


def data_loader(H: int, K: int) -> np.ndarray:
    """
    Tries to load the exact solution of the specified granularity

    :param H: H+1: number of spatial discretisation points
    :param K: K+1: number of temporal discretisation points
    :return: The exact solution as a numpy array
    """
    try:
        data = scipy.io.loadmat(
            f'burgers_exact/solutions/burgers_exact_N_t={K + 1}_N_x={H + 1}.mat')
        return np.real(data['mysol'])
    except FileNotFoundError as e:
        cprint(e, "red")
        cprint("Please make sure that the exact solution for the desired granularity exists. Check out "
               "burgers_exact/README.md", "red")
        sys.exit(0)
