"""Patches.
"""

import numpy as np
from numpy.typing import ArrayLike
from . import utils


def segment(X:ArrayLike, h:ArrayLike, l:float, w:float) -> ArrayLike:
    """Line segment function.

    Args:
        X: coordinates of evaluation, of dimension `?x2`.
        h: tangent direction of the segment, which is perpendicular to the normal direction.
        l: length of the segment.
        w: width of the segment.

    Returns:
        function values at the given coordinates.
    """
    h = h / np.linalg.norm(h)
    g = np.asarray([-h[1], h[0]])
    return (np.abs(X @ g) < w/2) * (np.abs(X @ h) < l/2)

# def gabor(x, *,f:float, σ2:float, θ:float):
#     """Gabor function.
#     """
#     return (1+np.exp(-(x[0]**2+x[1]**2)/(2*σ2))*np.cos(2*np.pi*f*(x[1]*np.cos(θ)-x[0]*np.sin(θ))))/2

# vgabor = np.vectorize(gabor)


def gabor(X:ArrayLike, h:ArrayLike, f:float, σ2:float) -> ArrayLike:
    """Gabor function.

    Args:
        X: coordinates of evaluation, of dimension ?x2.
        h: tangent direction of the segment, which is perpendicular to the normal direction.
        f: frequency of modulation.
        σ2: variance.
    """
    h = h / np.linalg.norm(h)
    g = np.asarray([-h[1], h[0]])
    nX = np.linalg.norm(X, axis=-1)
    return (0 + np.exp(-(nX)/(2*σ2)) * np.cos(2*np.pi*f*X@g)) / 2


def generate_image(Ps:ArrayLike, Hs:ArrayLike=None, *, N:int, pfunc:callable) -> ArrayLike:
    """Generate a pixel image of random oriented patches, with optional tangent directions.

    Args:
        Ps: position of balls in the square `[0,1]x[0,1]`, of shape `?x2`
        Hs: tangent direction at `Ps`.
        N: image resolution in pixels.
        pfunc: patch function. `pfunc(z,g)` is the patch function with orientation `g` evaluated at `z` (of shape `?x2`).

    Returns:
        a pixel image.
    """
    I = np.zeros((N,N), dtype=float)

    # meshgrid on [0,1]x[0,1]
    XYg = np.stack(np.meshgrid(range(N), range(N))).reshape(2,-1).T / N
    # relative coordinates of meshgrid points to balls
    Z = XYg[:,:,None] - Ps.T[None,:,:]

    # iteration over balls is more efficient than over pixels
    for n in range(Z.shape[-1]):
        z = Z[:,:,n]
        if Hs is not None:
            h = Hs[n] #; h /= np.linalg.norm(h)
        else:
            h = np.random.randn(2) #; h /= np.linalg.norm(h)
        I += pfunc(z,h).reshape(N, N)

    return I


def generate_image_foreground(Ps:ArrayLike, Xs:ArrayLike, Hs:ArrayLike, *, N:int, ng:int=1, jitter:float=0., pfunc:callable) -> ArrayLike:
    """Generate a pixel image of random oriented patches, with tangent direction computed from foreground points.

    Args:
        Ps: position of balls in the square `[0,1]x[0,1]`, of shape `?x2`
        Xs: foreground curve points, of shape `?x2`.
        Hs: tangent at `Xs`, of the same shape.
        N: image resolution in pixels.
        ng: number of adjacent points for smoothing the tangent.
        jitter: maximal angular jitter around the tangent direction, in degree.
        pfunc: patch function. `pfunc(z,g)` is the patch function with orientation `g` evaluated at `z` (of shape ?x2).

    Returns:
        a pixel image.

    Notes:
        Unlike `generate_image()`, this method requires the information of a contour's position and tangent to make alignment. These information can be obtained with the method `utils.contour_from_image()`. The points in `Ps` are randomly distributed on the curve randomly, while the foreground points `Xs` are regularly spaced on the curve and are used as reference to compute the smoothed tangent at `Ps` using `ng` nearest neighbours.
    """
    I = np.zeros((N,N), dtype=float)

    # retrieve smoothed gradients using foreground points
    dist = np.linalg.norm(Ps[:,:,None] - Xs.T[None,:,:], axis=1)
    idx = np.argsort(dist, axis=1)[:,:ng]
    H = np.median(Hs[idx], axis=1)
    # H = H + np.random.randn(*H.shape) * jitter
    H = utils.add_jitter(H[:,0], H[:,1], jitter).T

    # meshgrid on [0,1]x[0,1]
    XYg = np.stack(np.meshgrid(range(N), range(N))).reshape(2,-1).T / N
    # relative coordinates of meshgrid points to balls
    Z = XYg[:,:,None] - Ps.T[None,:,:]

    # iteration over balls is more efficient than over pixels
    for n in range(Z.shape[-1]):
        z = Z[:,:,n]
        h = H[n]
        I += pfunc(z,h).reshape(N, N)

    return I