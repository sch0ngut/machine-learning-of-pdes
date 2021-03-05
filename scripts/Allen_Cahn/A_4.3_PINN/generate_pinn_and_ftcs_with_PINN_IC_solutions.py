from machine_learning_solver.PINN import AllenCahnPINN
import tensorflow.keras.losses as losses
import numpy as np
from sklearn.metrics import mean_squared_error
from numerical_solvers.AllenCahn_FTCS import AllenCahnFTCS
from util.generate_plots import generate_contour_and_snapshots_plot

# Part 1: solve AC equation using PINN

pinn = AllenCahnPINN(loss_obj=losses.MeanSquaredError(), n_nodes=20, n_layers=8)
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
pinn.perform_training(max_n_epochs=1000, min_mse=0.05, track_losses=True, batch_size='full')

# Print loss data frame and plot solution
print(pinn.loss_df)
generate_contour_and_snapshots_plot(pinn.u_pred, train_feat=pinn.train_feat, legend_loc='center left')
# generate_contour_and_snapshots_plot(pinn.u_pred, train_feat=pinn.train_feat, legend_loc='center left',
#                                     savefig_path='plots/Fig10_PINN_contour_and_snapshots_plot.jpg')

# Part 2: use initial initial condition predicted by PINN for FTCS

# Generate initial data for Upwind solver
n_spatial = 512
n_temporal = 201
feat = np.column_stack((np.linspace(-1, 1, n_spatial), np.zeros(n_spatial)))
initial_data = pinn.network(feat)[:, 0]

# Run solver
ftcs = AllenCahnFTCS(n_spatial=n_spatial, n_temporal=n_temporal, u0=initial_data, order=2)
ftcs.time_integrate()

# Plot
generate_contour_and_snapshots_plot(u=ftcs.u_numerical)
# generate_contour_and_snapshots_plot(u=ftcs.u_numerical,
#                                     savefig_path='plots/Fig12_FTCS_with_PINN_IC_contour_and_snapshots_plot.jpg')

# Compute error between Upwind and PINN solution
x = np.linspace(-1, 1, n_spatial)
t = np.linspace(0, 1, n_temporal)
x_mesh, t_mesh = np.meshgrid(x, t)
eval_feat = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))
pinn_u_pred = np.reshape(pinn.network(eval_feat), (n_temporal, n_spatial)).T
print(f"L2(Upwind - PINN): {np.sqrt(mean_squared_error(pinn_u_pred, ftcs.u_numerical))}")
