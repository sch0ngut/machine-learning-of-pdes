from numerical_solvers.Burgers_FTCS import BurgersFTCS
from util.generate_plots import *

ftcs = BurgersFTCS(n_spatial=161, n_temporal=10**4+1)
ftcs.time_integrate()
print(ftcs.get_l2_error())
generate_snapshots_plot(u=ftcs.u_numerical, t_vec=np.array([0, 0.35, 0.4, 0.85]))
# generate_snapshots_plot(u=ftcs.u, t_vec=np.array([0, 0.35, 0.4, 0.85]),
#                         savefig_path='scripts/Burgers/3.1_FTCS/Fig2b_oscillations.jpg')
