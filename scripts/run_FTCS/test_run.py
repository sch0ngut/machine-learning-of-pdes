from burgers_numerical.FTCS import FTCS
from util.generate_plots import *

ftcs = FTCS(num_spatial=160, num_temporal=10**3)
ftcs.time_integrate()
print(ftcs.get_l2_error())
# generate_snapshots_plot(u=ftcs.U, H=ftcs.H, K=ftcs.K, t_vec=np.array([0, 0.4, 0.8]))
generate_contour_plot(u=ftcs.U, H=ftcs.H, K=ftcs.K)
