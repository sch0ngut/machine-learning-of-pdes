import numpy as np
from numerical_solvers.Burgers_Upwind import BurgersUpwind


# Order of the upwind scheme: either 1 or 2
order = 1

# Discretisation
n_spatial_vec = [161, 321, 641, 1281, 2561]
n_temporal = 10**3+1
m = len(n_spatial_vec)

# Error vectors initialisation
l_2_errors = np.zeros(m)
l_max_errors = np.zeros(m)

# Compute solution for each discretisation
for i, n_spatial in enumerate(n_spatial_vec):
    print(n_spatial)
    upwind = BurgersUpwind(n_spatial=n_spatial, n_temporal=n_temporal, order=order)
    upwind.time_integrate()
    l_2_errors[i] = upwind.get_l2_error()
    l_max_errors[i] = upwind.get_l_max_error()

# Print errors
with np.printoptions(formatter={'float': lambda x: format(x, '6.2e')}):
    print(l_2_errors)
    print(l_max_errors)

# Calculate convergence rates
with np.printoptions(precision=2, suppress=True):
    print(np.log(l_2_errors[0:(m-1)]/l_2_errors[1:m])/np.log(2))
    print(np.log(l_max_errors[0:(m-1)]/l_max_errors[1:m])/np.log(2))

