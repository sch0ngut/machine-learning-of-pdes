import numpy as np
from burgers_numerical.Upwind import Upwind


# Order of the upwind scheme: either 1 or 2
order = 2

# Discretisation
H_vec = [160, 320, 640, 1280, 2560, 5120]
K = 10**4

# Error vectors initialisation
l_2_errors = np.zeros(len(H_vec))
l_max_errors = np.zeros(len(H_vec))

# Compute solution for each discretisation
for i, H in enumerate(H_vec):
    print(H)
    upwind = Upwind(num_spatial=H, num_temporal=K, order=order)
    upwind.time_integrate()
    l_2_errors[i] = upwind.get_l2_error()
    l_max_errors[i] = upwind.get_l_max_error()

# Print errors
with np.printoptions(formatter={'float': lambda x: format(x, '6.2e')}):
    print(l_2_errors)
    print(l_max_errors)

# Calculate convergence rates
with np.printoptions(precision=4, suppress=True):
    print(np.log(l_2_errors[0:(len(H_vec)-1)]/l_2_errors[1:len(H_vec)])/np.log(2))
    print(np.log(l_max_errors[0:(len(H_vec)-1)]/l_max_errors[1:len(H_vec)])/np.log(2))

