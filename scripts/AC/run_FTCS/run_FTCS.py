from numerical_solvers.AllenCahn_FTCS import AllenCahnFTCS
from util.generate_plots import *

ftcs = AllenCahnFTCS(n_spatial=512, n_temporal=201)
ftcs.time_integrate()
print(ftcs.get_l2_error())
generate_contour_and_snapshots_plot(u=ftcs.u_numerical)
# generate_contour_and_snapshots_plot(u=ftcs.u_numerical,
#                                     savefig_path='scripts/AC/run_FTCS/contour_and_snapshots_plot.jpg')
