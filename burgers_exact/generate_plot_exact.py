import scipy.io
from util.generate_plots import *

# Resolution
n_spatial = 1281
n_temporal = 101

# Load data
data = scipy.io.loadmat(f'burgers_exact/solutions/burgers_exact_N_t={n_temporal}_N_x={n_spatial}.mat')

u_exact = np.real(data['mysol'])

generate_contour_and_snapshots_plot(u=u_exact)
# generate_contour_and_snapshots_plot(u=u_exact, savefig_path='plots/Burgers_exact.jpg')

