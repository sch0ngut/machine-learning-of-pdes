import scipy.io
from util.generate_plots import *

# Resolution
H = 1280
K = 100

# Load data
data = scipy.io.loadmat(f'burgers_exact/solutions/burgers_exact_N_t={K+1}_N_x={H+1}.mat')

u_exact = np.real(data['mysol'])

generate_contour_and_snapshots_plot(u=u_exact, H=H, K=K, savefig_path='plots/Burgers_exact.jpg')

