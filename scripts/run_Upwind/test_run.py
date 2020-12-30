from burgers_numerical.Upwind import Upwind
from util.generate_plots import *

upwind = Upwind(n_spatial=161, n_temporal=10**3+1)
upwind.time_integrate()
print(upwind.get_l2_error())
generate_snapshots_plot(u=upwind.u_numerical, t_vec=np.array([0, 0.4, 0.8]))

