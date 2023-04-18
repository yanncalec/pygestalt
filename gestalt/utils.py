import numpy as np
from matplotlib import pyplot as plt

def contour_from_cartoon(X0:np.ndarray, s:float=0.5, **kwargs):
    """Get contour information from a cartoon image.

    The numpy convention for the pixel image:
    - row (first dimension) as x axis
    - origin is at upper-left corner

    Args:
        X0: 2d input image
        s: padding factor
        keyword args: for the method `np.pad()`

    Returns:
        a padded image
        contour coordinates in the padded image
        finite difference at contour points
    """

    if s>0:
        d = np.int32(np.asarray(X0.shape) * s/2, **kwargs)
        X = np.pad(X0, d, **kwargs)
    else:
        X = X0

    # np.gradient returns the derivative along each axis
    dX = np.asarray(np.gradient(X))
    Y = np.linalg.norm(dX, axis=0)
    idx = np.abs(Y) > 0

    # Position of contour points
    Ps = np.vstack(np.where(idx)).T
    # Gradient at contour points
    Gs = dX[:,idx].T

    return X, Ps, Gs


def extend_image(X:np.ndarray, s:float=.5):
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
        _ = ax.set_title(f'Total points: {N}, Threshold={thresh:.0e}')

    return fig, ax