from burgers_numerical.HopfCole import HopfCole
from util.generate_plots import *

hopfcole = HopfCole(num_spatial=640, num_temporal=10**4)
hopfcole.time_integrate()
print(hopfcole.get_l2_error())
generate_contour_and_snapshots_plot(u=hopfcole.U, H=hopfcole.H, K=hopfcole.K, t_vec=np.array([0, 0.4, 0.8]))

