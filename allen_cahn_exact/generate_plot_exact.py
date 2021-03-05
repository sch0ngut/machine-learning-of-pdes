import scipy.io

from util.data_loader import allen_cahn_data_loader
from util.generate_plots import *

# Load data
_, _, u_exact = allen_cahn_data_loader()

# generate_contour_and_snapshots_plot(u=u_exact)
generate_contour_and_snapshots_plot(u=u_exact, savefig_path='plots/allen_cahn_exact.jpg')

