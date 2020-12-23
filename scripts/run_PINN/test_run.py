from burgers_ml.PINN import PINN
from util.generate_plots import *
import numpy as np
import tensorflow as tf

tf.random.set_seed(1)
np.random.seed(1)

pinn = PINN(n_coll=10000, loss_obj=tf.keras.losses.MeanAbsoluteError())
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
loss_df, final_mse = pinn.perform_training(max_n_epochs=15000, min_mse=0.0005, track_losses=True, batch_size='full')
print(loss_df)
print(final_mse)
u_preds = pinn.get_predictions_shaped()
generate_contour_and_snapshots_plot(u_preds.T, train_feat=pinn.train_feat)

plot_df = loss_df[['loss_IC', 'loss_BC', 'loss_coll', 'error']]
plot_df.columns = ['loss on initial data', 'loss on boundary data', 'loss on collocation points', 'error']
generate_loss_plot(plot_df)
