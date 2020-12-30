from burgers_numerical.FTCS import FTCS
from util.generate_plots import *

ftcs = FTCS(n_spatial=321, n_temporal=10**3+1)
ftcs.time_integrate()
print(ftcs.get_l2_error())
print(ftcs.get_mean_squared_error())
print(ftcs.get_mean_absolute_error())
# generate_snapshots_plot(u=ftcs.u, t_vec=np.array([0, 0.4, 0.8]))
generate_contour_plot(u=ftcs.u_numerical)
