import scipy.io
from util.generate_plots import *

# Resolution
n_spatial = 1281
n_temporal = 1001

# Load data
data = scipy.io.loadmat(f'burgers_exact/solutions/burgers_exact_K={n_temporal-1}_H={n_spatial-1}.mat')

u_exact = np.real(data['mysol'])

generate_contour_and_snapshots_plot(u=u_exact)
# generate_contour_and_snapshots_plot(u=u_exact, savefig_path='burgers_exact/burgers_exact.jpg')

