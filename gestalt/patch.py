"""Patches.
"""

import numpy as np

def segment(X, g, l:float, w:float):
    """Line segment.
    """
    g = g / np.linalg.norm(g)
    h = np.asarray([-g[1], g[0]])
    return (np.abs(X @ g) < w/2) * (np.abs(X @ h) < l/2)

# def gabor(x, *,f:float, σ2:float, θ:float):
#     """Gabor function.
#     """
#     return (1+np.exp(-(x[0]**2+x[1]**2)/(2*σ2))*np.cos(2*np.pi*f*(x[1]*np.cos(θ)-x[0]*np.sin(θ))))/2

# vgabor = np.vectorize(gabor)


def gabor(X, g, f:float, σ2:float):
    """Gabor function.
    """
    g = g / np.linalg.norm(g)
    nX = np.linalg.norm(X, axis=-1)
    return (0 + np.exp(-(nX)/(2*σ2)) * np.cos(2*np.pi*f*X@g)) / 2


def generate_image(P, Xs=None, Gs=None, *, N:int, ng:int=10, pfunc:callable):
    I = np.zeros((N,N), dtype=float)
    if (Xs is not None) and (Gs is not None):
        dist = np.linalg.norm(P[:,:,None] - Xs.T[None,:,:], axis=1)
        idx = np.argsort(dist, axis=1)[:,:ng]
        G = np.mean(Gs[idx], axis=1)

    XYg = np.stack(np.meshgrid(range(N), range(N))).reshape(2,-1).T / N
    Z = XYg[:,:,None] - P.T[None,:,:]

    for n in range(Z.shape[-1]):
        z = Z[:,:,n]
        if (Xs is not None) and (Gs is not None):
            g = G[n]; g /= np.linalg.norm(g)
        else:
            g = np.random.randn(2); g /= np.linalg.norm(g)
        I += pfunc(z,g).reshape(N, N)

    return I