from burgers_ml.PINN import PINN
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

pinn = PINN()
pinn.generate_training_data(n_initial=10, n_boundary=4)
print(pinn.coll_points)