from numerical_solvers.Burgers_FTCS import BurgersFTCS
from util.generate_plots import *

ftcs = BurgersFTCS(n_spatial=2561, n_temporal=10**4+1)
ftcs.time_integrate()
print(ftcs.get_l2_error())
generate_snapshots_plot(u=ftcs.u_numerical, t_vec=np.array([0, 0.03, 0.04]))
# Zoomed and saved under 'scripts/run_FTCS/instabilities.jpg'
