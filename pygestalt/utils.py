import numpy as np
from numpy.typing import ArrayLike

from skimage import feature, filters

from matplotlib import pyplot as plt


def angle(dx, dy):
    return np.mod(np.arctan2(dy, dx), np.pi)

def add_jitter(dx, dy, jitter:float, mode:str='random'):
    """Add jitters to given directions.
    """
    # assert H.ndim==2 and H.shape[1]==2, "Wrong input shape."
    assert len(dx)==len(dy)
    j = jitter/180*np.pi
    if mode == 'random':
        j *= 2*(np.random.rand(len(dx))-0.5)
    a = angle(dx, dy) + j
    return np.asarray([np.cos(a), np.sin(a)])


def image_derivative(X:ArrayLike, method:str='sobel') -> tuple[ArrayLike]:
    """Compute the image derivatives and its direction.

    Args:
        X: input image
        method: scikit-image method for derivatives, {'sobel', 'prewitt', 'scharr', 'farid'}

    Returns:
        horizontal derivative
        vertical derivative
        normal direction, or the angle (in [0,π]) of the derivative image
    """
    if method == 'sobel':
        df_h, df_v = filters.sobel_h, filters.sobel_v
    elif method == 'prewitt':
        df_h, df_v = filters.prewitt_h, filters.prewitt_v
    elif method == 'scharr':
        df_h, df_v = filters.scharr_h, filters.scharr_v
    elif method == 'farid':
        df_h, df_v = filters.farid_h, filters.farid_v
    else:
        raise NameError(f"Unknwon method: {method}")

    dXh, dXv = df_h(X), df_v(X)

    return dXh, dXv, angle(dXh, dXv)


def contour_from_image(X:ArrayLike, *, method='farid', **kwargs):
    """Get contour information of an image.

    Args:
        X: input image
        keyword args: for the method `feature.canny()`

    Returns:
        index of contour points
        tangent direction of contour points

    Notes:
        The numpy convention for the pixel image:
        - row (first dimension) as x axis
        - origin is at upper-left corner
    """
    _, _, A = image_derivative(X, method=method)
    # Contour detection using Canny detector
    E = feature.canny(X, sigma=1)
    # Position of contour points
    P = np.vstack(np.where(E)).T
    # Tangent direction of contour points
    H = np.vstack([np.cos(np.pi/2+A[E]), np.sin(np.pi/2+A[E])]).T

    return P, H


def extend_image(X:ArrayLike, s:float=.5):
    if s<=0.:
        return X.copy()
    else:
        xr, xc = X.shape[0], X.shape[1]
        yr, yc = int(xr*(1+s)), int(xc*(1+s))
        Y = np.zeros((yr, yc, *X.shape[2:]), dtype=X.dtype)
        sr, sc = (yr-xr)//2, (yc-xc)//2
        Y[sr:(sr+xr), sc:(sc+xc)] = X
        return Y


def ball_plot(B, F=None, radius:float=None, thresh:float=None):
    fig = plt.figure(); ax = plt.gca()

    # foreground
    X = B
    xs, ys = X[:,0], X[:,1]
    ax.scatter(xs, ys, marker='.')
    if radius is not None:
        for (x,y) in X:
            circle = plt.Circle((x, y), radius, alpha=0.05)
            ax.add_patch(circle)
    N = len(X)

    # background
    if F is not None:
        X = F
        xs, ys = X[:,0], X[:,1]
        ax.scatter(xs, ys, marker='.')

        if radius is not None:
            for (x,y) in X:
                circle = plt.Circle((x, y), radius, alpha=0.05)
                ax.add_patch(circle)
        N += len(X)

    ax.set_xlim((-0.05,1.05))
    ax.set_ylim((-0.05,1.05))
    ax.set_aspect('equal')

    if thresh is None:
        _ = ax.set_title(f'Total points: {N}')
    else:
        _ = ax.set_title(f'Total points: {N}, Threshold: {thresh:.0e}')

    return fig, ax