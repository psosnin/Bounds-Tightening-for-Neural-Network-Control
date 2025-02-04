import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .helpers import getLogGeometricMean


def plotBounds(bounds):
    """
    Plot different bound types on the same set of axes.
    Parameters:
        bounds: dict of name: bounds or list of bounds
    """
    if not isinstance(bounds, dict):
        bounds = {"": bounds}
    L = len(list(bounds.values())[0])
    f, ax = plt.subplots(ncols=L, sharey=True, figsize=(3 * L, 5))
    for i in range(L):
        ax[i].set_xlabel(f"Layer {i + 1}")
    ax[0].set_ylabel("$f(x) = w^Tx + b$")

    color = sns.color_palette('Dark2', len(bounds))
    for j, (n, bd) in enumerate(bounds.items()):
        plotSingleBound(bd, ax, color[j], n)
    f.legend(bbox_to_anchor=(0.9, 0.9), loc="upper left")
    return f


def plotSingleBound(bound, ax, color, label):
    """
    Plot the a single type bound on the given axis.
    """
    for i, (L, U) in enumerate(bound):
        ax[i].plot(L, color=color, label=label)
        label = None
        ax[i].plot(U, color=color)


def plotMNIST(data, target=None):
    """
    Show a single MNIST image.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.size == 784
    if len(data.shape) == 1:
        data = data.reshape(28, 28)
    plt.figure(figsize=(2, 2))
    plt.imshow(data.squeeze(), cmap='gray')
    if target:
        plt.title(target)
    plt.show()


def plotBoundsTimeVsTightness(bounds, time):
    """
    Plot the time vs tightness of the bounds.
    Parameters:
        bounds: dict of name: bounds
        time: dict of name: computation time
    """
    f = plt.figure(figsize=(6, 4))
    for n, v in bounds.items():
        plt.scatter(np.log(time[n]), getLogGeometricMean(v)[-1], label=n)
    plt.xlabel("Log time taken to compute bounds")
    plt.ylabel("Log geometric mean of final layer bounds")
    plt.legend(bbox_to_anchor=(1.6, 0.5), loc="center right")
    return f


def plotBoundsTightnessVsLayer(bounds):
    """
    Plot the tightness of the bound at each layer of the network.
    Parameters:
        bounds: dict of name: bounds or list of bounds
        time: dict of name: computation time
    """
    if not isinstance(bounds, dict):
        bounds = {"": bounds}
    f = plt.figure(figsize=(6, 4))
    for n, v in bounds.items():
        plt.plot(getLogGeometricMean(v), label=n)
    plt.xlabel("Layer")
    plt.ylabel("Log geometric mean of bounds")
    plt.legend(bbox_to_anchor=(1.6, 0.5), loc="center right")
    return f
