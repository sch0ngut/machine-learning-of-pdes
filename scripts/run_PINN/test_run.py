from burgers_ml.PINN import PINN
from util.generate_plots import *
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

pinn = PINN(n_coll=1000, loss_obj=tf.keras.losses.MeanAbsoluteError())
pinn.generate_training_data(n_initial=50, n_boundary=25)
loss_df, final_mse = pinn.perform_training(max_n_epochs=3, min_mse=0.005, track_losses=True, batch_size=100)
print(loss_df)
print(final_mse)
u_preds = pinn.get_predictions_shaped()
generate_contour_and_snapshots_plot(u_preds.T)

plot_df = loss_df[['loss_IC', 'loss_BC', 'loss_coll', 'error']]
plot_df.columns = ['loss on initial data', 'loss on boundary data', 'loss on collocation points', 'error']
generate_loss_plot(plot_df)
