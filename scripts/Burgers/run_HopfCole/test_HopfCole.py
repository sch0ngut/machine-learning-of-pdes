from numerical_solvers.Burgers_HopfCole import BurgersHopfCole
from util.generate_plots import *

hopfcole = BurgersHopfCole(n_spatial=641, n_temporal=10**4+1)
hopfcole.time_integrate()
print(hopfcole.get_l2_error())
generate_contour_and_snapshots_plot(u=hopfcole.u_numerical, t_vec=np.array([0, 0.4, 0.8]))

