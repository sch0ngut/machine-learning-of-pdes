from typing import Union

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from tensorflow.python.data import Dataset as tf_Dataset


class PINN:
    def __init__(self, act_fun: str = "tanh", n_nodes: int = 20, n_layers: int = 8, n_coll: int = 10000,
                 loss_obj: tf.losses = tf.keras.losses.MeanAbsoluteError(), H: int = 320, K: int = 100) -> None:

        # Network parameters
        self.act_fun = act_fun
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.n_coll = n_coll
        self.coll_points = tf.random.uniform(shape=[n_coll, 2], minval=[-1, 0], maxval=[1, 1])
        self.loss_obj = loss_obj

        # Network initialisation
        self.network = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.n_nodes, activation=self.act_fun, input_shape=(2,))])
        for i in range(self.n_layers):
            self.network.add(tf.keras.layers.Dense(self.n_nodes, activation=self.act_fun))
        self.network.add(tf.keras.layers.Dense(1))

        # Network evaluation
        exact_data = scipy.io.loadmat(
            f'burgers_exact/solutions/burgers_exact_N_t={K + 1}_N_x={H + 1}.mat')
        u_exact = np.real(exact_data['mysol']).T
        t = np.linspace(0, 1, K + 1)
        x = np.linspace(-1, 1, H + 1)
        X, T = np.meshgrid(x, t)
        self.eval_feat = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.eval_tar = u_exact.flatten()[:, None]

        # Training data initialisation
        self.train_data = pd.DataFrame()
        self.initial_train_data = pd.DataFrame()
        self.boundary_train_data = pd.DataFrame()
        # ToDo: set right types
        # self.train_feat = tf.Tensor()
        # self.train_tar = tf.Tensor()
        # self.initial_train_feat = tf.Tensor()
        # self.initial_train_tar = tf.Tensor()
        # self.boundary_train_feat = tf.Tensor()
        # self.boundary_train_tar = tf.Tensor()

    def generate_training_data(self, n_initial: int, n_boundary: int) -> None:
        # Generate data
        x = np.linspace(-1, 1, n_initial)
        u0 = np.zeros(n_initial)
        u0[1:-1] = -np.sin(np.pi * x[1:-1])
        t = np.linspace(1 / (n_boundary - 1), 1, n_boundary - 1)
        data_t0 = pd.DataFrame({"x": x, "t": np.zeros(n_initial), "u": u0})
        data_boundary1 = pd.DataFrame({"x": np.ones(n_boundary - 1), "t": t, "u": np.zeros(n_boundary - 1)})
        data_boundary2 = pd.DataFrame({"x": -np.ones(n_boundary - 1), "t": t, "u": np.zeros(n_boundary - 1)})

        # Set training data
        self.train_data = pd.concat([data_t0, data_boundary1, data_boundary2])
        self.initial_train_data = self.train_data.loc[self.train_data['t'] == 0]
        self.boundary_train_data = self.train_data.loc[(self.train_data['x'] == 1) | (self.train_data['x'] == -1)]

        # Split training data in features and targets
        self.train_feat, self.train_tar = next(iter(self.batch_and_split_data(self.train_data)))
        self.initial_train_feat, self.initial_train_tar = next(
            iter(self.batch_and_split_data(self.initial_train_data)))
        self.boundary_train_feat, self.boundary_train_tar = next(
            iter(self.batch_and_split_data(self.boundary_train_data)))

    @staticmethod
    def batch_and_split_data(data, batch_size: Union[int, str] = 'full', shuffle: bool = True) -> tf_Dataset:

        # Takes a pd.DataFrame, batches it and splits the batches into features and labels
        if batch_size == 'full':
            batch_size = data.shape[0]

        data_copy = data.copy()
        u = data_copy.pop('u')
        data_set = tf.data.Dataset.from_tensor_slices((data_copy.values, u.values))
        if shuffle:
            data_set = data_set.shuffle(data_copy.shape[0], reshuffle_each_iteration=False)
        return data_set.batch(batch_size, drop_remainder=False)
