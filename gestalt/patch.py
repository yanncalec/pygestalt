"""Patches.
"""

import numpy as np
from numpy.typing import ArrayLike

def segment(X:ArrayLike, g:ArrayLike, l:float, w:float) -> ArrayLike:
    """Line segment function.

    Args:
        X: coordinates of evaluation, of dimension ?x2.
        g: normal direction of the segment.
        l: length of the segment.
        w: width of the segment.

    Returns:
        function values at the given coordinates.
    """
    g = g / np.linalg.norm(g)
    h = np.asarray([-g[1], g[0]])
    return (np.abs(X @ g) < w/2) * (np.abs(X @ h) < l/2)

# def gabor(x, *,f:float, σ2:float, θ:float):
#     """Gabor function.
#     """
#     return (1+np.exp(-(x[0]**2+x[1]**2)/(2*σ2))*np.cos(2*np.pi*f*(x[1]*np.cos(θ)-x[0]*np.sin(θ))))/2

# vgabor = np.vectorize(gabor)


def gabor(X:ArrayLike, g:ArrayLike, f:float, σ2:float) -> ArrayLike:
    """Gabor function.

    Args:
        X: coordinates of evaluation, of dimension ?x2.
        g: normal direction of the patch.
        f: frequency of modulation.
        σ2: variance.
    """
    g = g / np.linalg.norm(g)
    nX = np.linalg.norm(X, axis=-1)
    return (0 + np.exp(-(nX)/(2*σ2)) * np.cos(2*np.pi*f*X@g)) / 2


def generate_image(P, Xs:ArrayLike=None, Gs:ArrayLike=None, *, N:int, ng:int=10, pfunc:callable) -> ArrayLike:
    """Generate a pixel image of random oriented patches located inside balls.

    Args:
        P: position of balls in [0,1]x[0,1], of shape ?x2
        N: image resolution in pixels.
        Xs: foreground curve points, of shape ?x2.
        Gs: gradients at `Xs`.
        ng: number of adjacent points for smoothing the gradient.
        pfunc: patch function. `pfunc(z,g)` is the patch function with orientation `g` evaluated at `z` (of shape ?x2).

    Returns:
        a pixel image.
    """
    I = np.zeros((N,N), dtype=float)
    # if foreground curve points and gradients are given, retrieve smoothed gradients
    foreground = (Xs is not None) and (Gs is not None)
    if foreground:
        dist = np.linalg.norm(P[:,:,None] - Xs.T[None,:,:], axis=1)
        idx = np.argsort(dist, axis=1)[:,:ng]
        G = np.mean(Gs[idx], axis=1)

    # meshgrid on [0,1]x[0,1]
    XYg = np.stack(np.meshgrid(range(N), range(N))).reshape(2,-1).T / N
    # relative coordinates of meshgrid points to balls
    Z = XYg[:,:,None] - P.T[None,:,:]

    # iteration on balls is more efficient than that on pixels
    for n in range(Z.shape[-1]):
        z = Z[:,:,n]
        if foreground:
            g = G[n]; g /= np.linalg.norm(g)
        else:
            g = np.random.randn(2); g /= np.linalg.norm(g)
        I += pfunc(z,g).reshape(N, N)

    return I