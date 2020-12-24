from burgers_ml.PINN import PINN
from burgers_numerical.Upwind import Upwind
from util.generate_plots import generate_contour_and_snapshots_plot
import pandas as pd
import numpy as np
import tensorflow as tf

# Part 1: Build and train PINN
tf.random.set_seed(42)
np.random.seed(42)

pinn = PINN()
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
loss_df, final_mse = pinn.perform_training(min_mse=0.05, track_losses=False, batch_size=100)

# Plot solution
print(f"PINN error: {final_mse}")
u_preds = pinn.get_predictions_shaped()
generate_contour_and_snapshots_plot(u=u_preds.T, savefig_path='scripts/run_Upwind_with_PINN/PINN_solution.jpg')


# Part 2: Use initial data generated by PINN and use them as IC for Upwind solver

n_spatial = 1281
n_temporal = 10**4 + 1

# Generate initial data for Upwind solver
feat = np.column_stack((np.linspace(-1, 1, n_spatial), np.zeros(n_spatial)))
initial_data = pinn.network(feat)[:, 0]

# Run solver
upwind = Upwind(n_spatial=n_spatial, n_temporal=n_temporal, u0=initial_data)
upwind.time_integrate()

# Get error and plot solution
print(f"Upwind error: {upwind.get_l2_error()}")
generate_contour_and_snapshots_plot(u=upwind.u, savefig_path='scripts/run_Upwind_with_PINN/Upwind_solution.jpg')