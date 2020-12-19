from burgers_numerical.Upwind import Upwind
from util.generate_plots import *

upwind = Upwind(num_spatial=160, num_temporal=10**3)
upwind.time_integrate()
print(upwind.get_l2_error())
generate_snapshots_plot(u=upwind.U, H=upwind.H, K=upwind.K, t_vec=np.array([0, 0.4, 0.8]))

