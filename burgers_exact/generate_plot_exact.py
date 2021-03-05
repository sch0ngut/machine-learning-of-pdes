from util.data_loader import burgers_data_loader
from util.generate_plots import *

# Resolution
n_spatial = 1281
n_temporal = 1001

# Load data
_, _, u_exact = burgers_data_loader(n_spatial=n_spatial, n_temporal=n_temporal)

# generate_contour_and_snapshots_plot(u=u_exact)
generate_contour_and_snapshots_plot(u=u_exact, savefig_path='plots/Fig1_burgers_exact.jpg')

