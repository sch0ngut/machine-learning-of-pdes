from burgers_ml.PINN import PINN
import tensorflow as tf
from util.generate_plots import *
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

pinn = PINN()
pinn.generate_training_data(n_initial=10, n_boundary=4)
loss_df = pinn.perform_training(max_n_epochs=3, min_train_loss=0.001, track_losses=True, batch_size=4)
print(loss_df)
u_preds = pinn.get_predictions_shaped()
generate_contour_plot(u_preds.T, H=pinn.n_spatial, K=pinn.n_temporal)
