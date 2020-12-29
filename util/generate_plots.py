import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


def generate_contour_plot(u: np.ndarray, savefig_path="", **kwargs) -> None:
    """
    Generates a contour plot

    :param u: An array containing the solution to plot. Should have the dimensions (n_spatial x n_temporal)
    :param savefig_path: The path were to store the plot. Leave empty if saving of the file is not desired.
    """
    # Define grid
    n_spatial = u.shape[0]
    n_temporal = u.shape[1]
    x = np.linspace(-1, 1, n_spatial)
    t = np.linspace(0, 1, n_temporal)
    x_mesh, t_mesh = np.meshgrid(x, t)

    # Plot
    v = np.linspace(-1, 1, 5, endpoint=True)
    plt.contourf(t_mesh, x_mesh, u.T, v, levels=100, cmap=plt.cm.jet)
    plt.colorbar(ticks=v)
    plt.ylabel(r'$x$')
    plt.xlabel(r'$t$')
    plt.title(r'$u(x,t)$')

    # Add the training data if desired
    if 'train_feat' in kwargs:
        train_feat = kwargs.get('train_feat')
        plt.plot(train_feat[:, 1], train_feat[:, 0], 'kx', label='training data', markersize=4, clip_on=False)

    plt.show()

    # Save
    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)


def generate_snapshots_plot(u: np.ndarray, t_vec: np.array = np.array([0, 0.25, 0.5, 0.75, 1]),
                            savefig_path: str = "") -> None:
    """
    Generates a time snapshots plot

    :param u: An array containing the solution to plot. Should have the dimensions (n_spatial x n_temporal)
    :param t_vec: A vector containing the desired time points to plot
    :param savefig_path: The path were to store the plot. Leave empty if saving of the file is not desired.
    """
    # Define grid
    n_spatial = u.shape[0]
    n_temporal = u.shape[1]
    x = np.linspace(-1, 1, n_spatial)

    # Plot
    for t_val in t_vec:
        j = int(t_val * (n_temporal-1))
        plt.plot(x, u[:, j], label=r'$t={{{}}}$'.format(t_val))
    plt.legend()
    plt.ylabel(r'$u(x,t)$')
    plt.xlabel(r'$x$')

    # Save
    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)

    plt.show()


def generate_contour_and_snapshots_plot(u: np.ndarray, t_vec: np.array = np.array([0, 0.25, 0.5, 0.75, 1]),
                                        savefig_path: str = "", **kwargs) -> None:
    """
    Generates a contour and time snapshots plot in one figure

    :param u: An array containing the solution to plot. Should have the dimensions (n_spatial x n_temporal)
    :param t_vec: A vector containing the desired time points to plot
    :param savefig_path: The path were to store the plot. Leave empty if saving of the file is not desired
    """
    # Define grid
    n_spatial = u.shape[0]
    n_temporal = u.shape[1]
    x = np.linspace(-1, 1, n_spatial)
    t = np.linspace(0, 1, n_temporal)
    x_mesh, t_mesh = np.meshgrid(x, t)

    # Create figure
    plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Contour Plot
    ax0 = plt.subplot(gs[0])
    v = np.linspace(-1, 1, 5, endpoint=True)
    p = ax0.contourf(t_mesh, x_mesh, u.T, v, levels=100, cmap=plt.cm.jet)
    plt.colorbar(p, ax=ax0, ticks=v)
    ax0.set_ylabel(r'$x$')
    ax0.set_xlabel(r'$t$')
    ax0.set_title(r'$u(x,t)$')
    # Add the training data if desired
    if 'train_feat' in kwargs:
        train_feat = kwargs.get('train_feat')
        plt.plot(train_feat[:, 1], train_feat[:, 0], 'kx', label='training data', markersize=4, clip_on=False)

    # Time snapshots plot
    ax1 = plt.subplot(gs[1])
    for t_val in t_vec:
        j = int(t_val * (n_temporal-1))
        # plt.plot(x, u_exact[:, j], label=r'$t={{{}}}$'.format(t_val))
        ax1.plot(x, u[:, j], label=r'$t={{{}}}$'.format(t_val))
    ax1.legend()
    ax1.set_ylabel(r'$u(x,t)$')
    ax1.set_xlabel(r'$x$')

    # Save
    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)

    plt.show()


def generate_loss_plot(loss_df: pd.DataFrame, savefig_path: str = None, **kwargs) -> None:
    """
    Generates a plot of the losses and against the epochs

    :param loss_df: The data frame containing the different losses in the columns and the epochs as indices
    :param color_dict: A dictionary assigning the column names of loss_df a color for plotting
    :param savefig_path: The path were to store the plot. Leave empty if saving of the file is not desired.
    """
    if 'color_dict' in kwargs:
        color_dict = kwargs.get('color_dict')
        loss_df.plot(color=[color_dict.get(x, '#333333') for x in loss_df.columns])
    else:
        loss_df.plot()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    if savefig_path:
        plt.savefig(savefig_path)

    plt.show()
