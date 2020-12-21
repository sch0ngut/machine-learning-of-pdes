from burgers_numerical.HopfCole import HopfCole
from util.generate_plots import *

hopfcole = HopfCole(n_spatial=641, n_temporal=10**4+1)
hopfcole.time_integrate()
print(hopfcole.get_l2_error())
generate_contour_and_snapshots_plot(u=hopfcole.u, t_vec=np.array([0, 0.4, 0.8]))

