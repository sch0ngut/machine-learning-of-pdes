from burgers_ml.PINN import PINN
from util.generate_plots import *
import tensorflow as tf

pinn = PINN(n_coll=10000, loss_obj=tf.keras.losses.MeanAbsoluteError())
pinn.generate_training_data(n_initial=50, n_boundary=25, equidistant=False)
pinn.perform_training(max_n_epochs=3, min_mse=0.0005, track_losses=True, batch_size='full')
print(pinn.loss_df)
print(pinn.mse)
generate_contour_and_snapshots_plot(pinn.u_pred, train_feat=pinn.train_feat)

plot_df = pinn.loss_df[['loss_IC', 'loss_BC', 'loss_coll', 'error']]
plot_df.columns = ['loss on initial data', 'loss on boundary data', 'loss on collocation points', 'error']
generate_loss_plot(plot_df)
