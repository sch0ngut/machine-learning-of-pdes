from typing import Union, List
import tensorflow as tf
import numpy as np
import pandas as pd
from util.data_loader import data_loader


class PINN:
    def __init__(self, act_fun: str = "tanh", n_nodes: int = 20, n_layers: int = 8, n_coll: int = 10000,
                 loss_obj: tf.losses = tf.keras.losses.MeanAbsoluteError(), n_spatial: int = 321,
                 n_temporal: int = 101) -> None:
        """
        :param act_fun: Activation function at each node of the neural network
        :param n_nodes: Number of nodes of each hidden layer
        :param n_layers: Number of hidden layers
        :param n_coll: Number of collocation points used to evaluate the regularisation term during model training
        :param loss_obj: The loss function
        :param n_spatial: Number of spatial discretisation points used for model evaluation
        :param n_temporal: Number of temporal discretisation points used for model evaluation
        """

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
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.u_exact = data_loader(self.n_spatial, self.n_temporal).T
        self.t = np.linspace(0, 1, self.n_temporal)
        self.x = np.linspace(-1, 1, self.n_spatial)
        x_mesh, t_mesh = np.meshgrid(self.x, self.t)
        self.eval_feat = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))
        self.eval_tar = self.u_exact.flatten()[:, None]
        print(self.eval_feat.shape)
        print(self.eval_tar.shape)

        # Training data initialisation
        self.train_data = pd.DataFrame()
        self.initial_train_data = pd.DataFrame()
        self.boundary_train_data = pd.DataFrame()
        self.train_feat = tf.zeros([0, 2])
        self.train_tar = tf.zeros([0, 1])
        self.initial_train_feat = tf.zeros([0, 2])
        self.initial_train_tar = tf.zeros([0, 1])
        self.boundary_train_feat = tf.zeros([0, 2])
        self.boundary_train_tar = tf.zeros([0, 1])

    def generate_training_data(self, n_initial: int, n_boundary: int) -> None:
        """
        Generates the training data on the initial and boundary intervals with a uniform distance

        :param n_initial: Number of training points at t=0
        :param n_boundary: Number of training points at x in {-1,1}
        """
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

    def perform_training(self, max_n_epochs=500, min_train_loss=0.01, batch_size='full',
                         optimizer=tf.keras.optimizers.Adam(), track_losses=True) -> pd.DataFrame:
        """
        Trains the network until a maximum given number of epochs or minimum loss on the training data is achieved.

        :param max_n_epochs: Stopping criterion: The maximum number of epochs
        :param min_train_loss: Stopping criterion: The minimum loss on the training data
        :param batch_size: Yhe batch size used during model training
        :param optimizer: The optimizer used for model training
        :param track_losses: Whether to track the losses in each epoch. Setting to False speeds up the computation
        :return: A data frame with the loss on training and collocation points together with the error at the given
        epochs
        """
        # Initialise loss data frame and print losses before model training
        loss_df = pd.DataFrame(
            columns=['epoch', 'loss_IC', 'loss_BC', 'loss_train', 'loss_coll', 'loss_tot', 'error']).set_index('epoch')
        loss_df.loc[0] = self.get_losses()
        print("Epoch {:03d}: loss_tot: {:.3f}, loss_train: {:.3f}, loss_coll: {:.3f}, error: {:.3f}".
              format(0,
                     loss_df.loc[0, 'loss_tot'],
                     loss_df.loc[0, 'loss_train'],
                     loss_df.loc[0, 'loss_coll'],
                     loss_df.loc[0, 'error']))

        epoch = 1
        while epoch <= max_n_epochs and self.loss_obj(self.network(self.train_feat), self.train_tar) > min_train_loss:

            # Perform training
            train_data_batched = self.batch_and_split_data(self.train_data, batch_size)
            for train_feat_batch, train_tar_batch in train_data_batched:
                loss_gradients = self.get_loss_gradients(train_feat_batch, train_tar_batch)
                optimizer.apply_gradients(zip(loss_gradients, self.network.trainable_variables))

            # Track training process
            if track_losses:
                loss_df.loc[epoch] = self.get_losses()
            if epoch % 100 == 0:
                if not track_losses:
                    loss_df.loc[epoch] = self.get_losses()
                print("Epoch {:03d}: loss_tot: {:.3f}, loss_train: {:.3f}, loss_coll: {:.3f}, error: {:.3f}".
                      format(epoch,
                             loss_df.loc[epoch, 'loss_tot'],
                             loss_df.loc[epoch, 'loss_train'],
                             loss_df.loc[epoch, 'loss_coll'],
                             loss_df.loc[epoch, 'error']))

            epoch = epoch + 1

        return loss_df

    def get_losses(self) -> List:
        """
        Computes the loss on the collocation points, the overall training data, the initial training data, the boundary
        training data and the error

        :return: A list containing the losses and the error
        """
        loss_coll = self.get_coll_loss().numpy()
        loss_train = self.loss_obj(self.network(self.train_feat), self.train_tar).numpy()
        loss_initial = self.loss_obj(self.network(self.initial_train_feat), self.initial_train_tar).numpy()
        loss_boundary = self.loss_obj(self.network(self.boundary_train_feat), self.boundary_train_tar).numpy()
        error = self.loss_obj(self.network(self.eval_feat), self.eval_tar).numpy()

        return [loss_initial, loss_boundary, loss_train, loss_coll, loss_train + loss_coll, error]

    def get_predictions_shaped(self) -> np.ndarray:
        """
        Generates the network's solution on the evaluation features
        :return: The predictions as an array
        """
        preds = self.network(self.eval_feat)
        return np.reshape(preds, (self.n_temporal, self.n_spatial))

    def get_coll_loss(self) -> tf.Tensor:
        """
        Computes the regularisation term of the loss function using the collocation points and automatic differentiation

        :return: The loss value as a tensor
        """
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.coll_points)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(self.coll_points)
                # predicted solution on coll_points
                u_coll = self.network(self.coll_points)
                u_coll = tf.gather(u_coll, 0, axis=1)
            # 1st order derivative
            u_coll_grads = t2.gradient(u_coll, self.coll_points)
            u_coll_x = tf.gather(u_coll_grads, 0, axis=1)
            u_coll_t = tf.gather(u_coll_grads, 1, axis=1)
        # 2nd order derivative
        u_coll_grads_2 = t1.gradient(u_coll_grads, self.coll_points)
        u_coll_xx = tf.gather(u_coll_grads_2, 0, axis=1)

        loss_coll_vec = u_coll_t + tf.multiply(u_coll, u_coll_x) - (0.01 / np.pi) * u_coll_xx
        loss_coll = self.loss_obj(loss_coll_vec, tf.zeros((self.n_coll,)))

        del t1, t2

        return loss_coll

    def get_loss_gradients(self, train_feat_batch: tf.Tensor, train_tar_batch: tf.Tensor) -> List[tf.Tensor]:
        """
        Computes the gradient of the overall loss functions, i.e. the loss function containing both the loss on some
        given training data and the loss on the collocation points

        :param train_feat_batch: The features of the training data
        :param train_tar_batch: The targets of the training data

        :return: A list of Tensors where each entry contains the gradients of the weights or the biases of one layer of
        the neural network
        """
        with tf.GradientTape() as tape:
            tape.watch(self.network.trainable_variables)
            loss_coll = self.get_coll_loss()
            loss_train = self.loss_obj(self.network(train_feat_batch), train_tar_batch)
            loss_tot = tf.dtypes.cast(loss_train, tf.float32) + loss_coll
        loss_gradients = tape.gradient(loss_tot, self.network.trainable_variables)
        del tape
        return loss_gradients

    @staticmethod
    def batch_and_split_data(data: pd.DataFrame, batch_size: Union[int, str] = 'full', shuffle: bool = True):
        """
        Takes a data frame, batches it and splits the batches into features and labels

        :param data: The data frame consisting of both features and targets
        :param batch_size: The desired batch size. Should be an integer or alternatively 'full'
        :param shuffle: Whether the data is shuffled before batching
        :return: An iterator containing the the batches. Each batch consists of two tensors. One containing the
        features and one containing the targets
        """
        if batch_size == 'full':
            batch_size = data.shape[0]

        data_copy = data.copy()
        u = data_copy.pop('u')
        data_set = tf.data.Dataset.from_tensor_slices((data_copy.values, u.values))
        if shuffle:
            data_set = data_set.shuffle(data_copy.shape[0], reshuffle_each_iteration=False)
        return data_set.batch(batch_size, drop_remainder=False)
