import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def generate_contour_plot(u, H, K, savefig_path=""):

    # Define axes
    x = np.linspace(-1, 1, H+1)
    t = np.linspace(0, 1, K+1)

    # Create figure
    # plt.figure(figsize=(14, 4))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Contour Plot
    X, T = np.meshgrid(x, t)
    # ax0 = plt.subplot(gs[0])
    v = np.linspace(-1, 1, 5, endpoint=True)

    plt.contourf(T, X, u.T, v, levels=100, cmap=plt.cm.jet)
    plt.colorbar(ticks=v)
    plt.ylabel(r'$x$')
    plt.xlabel(r'$t$')
    plt.title(r'$u(x,t)$')

    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)
    plt.show()


def generate_snapshots_plot(u, H, K, t_vec=np.array([0, 0.25, 0.5, 0.75, 1]), savefig_path=""):

    x = np.linspace(-1, 1, H+1)
    t = np.linspace(0, 1, K+1)

    for t_val in t_vec:
        j = int(t_val * K)
        plt.plot(x, u[:, j], label=r'$t={{{}}}$'.format(t_val))
    plt.legend()
    plt.ylabel(r'$u(x,t)$')
    plt.xlabel(r'$x$')

    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)
    plt.show()


def generate_contour_and_snapshots_plot(u, H, K, t_vec=np.array([0, 0.25, 0.5, 0.75, 1]), savefig_path=""):

    # Define axes
    x = np.linspace(-1, 1, H+1)
    t = np.linspace(0, 1, K+1)

    # Create figure
    plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Contour Plot
    X, T = np.meshgrid(x, t)
    ax0 = plt.subplot(gs[0])
    v = np.linspace(-1, 1, 5, endpoint=True)

    p = ax0.contourf(T, X, u.T, v, levels=100, cmap=plt.cm.jet)
    plt.colorbar(p, ax=ax0, ticks=v)
    ax0.set_ylabel(r'$x$')
    ax0.set_xlabel(r'$t$')
    ax0.set_title(r'$u(x,t)$')

    # Time snapshots plot
    ax1 = plt.subplot(gs[1])
    for t_val in t_vec:
        j = int(t_val * K)
        # plt.plot(x, u_exact[:, j], label=r'$t={{{}}}$'.format(t_val))
        ax1.plot(x, u[:, j], label=r'$t={{{}}}$'.format(t_val))
    ax1.legend()
    ax1.set_ylabel(r'$u(x,t)$')
    ax1.set_xlabel(r'$x$')

    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)
    plt.show()
