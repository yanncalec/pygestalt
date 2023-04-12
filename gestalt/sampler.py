"""Random sampler for stimuli.
"""

import numpy as np
from scipy import stats

def draw_positions(radius:float, sampler, *,
             exclusions:np.ndarray=np.empty((0,2)),
            #  fmax:float=np.inf,
             thresh:float=1e-3):
    """Draw random positions of balls.

    Args
    ----
    radius: of balls
    sampler: a generator of infinite length
    exclusion: a set of coordinates to be excluded
    thresh
    """
    X = np.empty((0,2))
    niter = 0

    for x in sampler:
        # The candidate ball must not touch the exclusion balls
        dist = np.linalg.norm(x-exclusions, axis=-1)
        if not(dist.size==0 or np.min(dist) >= 2*radius):
            continue

        niter += 1
        dist = np.linalg.norm(x-X, axis=-1)
        idx = np.where(2*radius<=dist)[0]

        # The candidate ball must not touch more than one existing ball,
        # and must be close to at least on existing ball.
        # if len(idx)>=len(X)-1 and (dist.size==0 or np.min(dist)<=(2+fmax)*radius):
        if len(idx)>=len(X)-1:
            X = np.vstack([X[idx],x])

        # exit if have generated enough points and current sampling is efficient
        if len(X)/niter <thresh:
            break

    return X


def polygone(P:list, α:float=1.):
    """Draw random samples in a polygone area.
    """
    d = stats.dirichlet(np.ones(len(P))*α)  # symmetric Dirichlet distribution
    while True:
        t = d.rvs()
        yield np.sum(t[:,None]*P, axis=0)


def box(pos:tuple=(0,0), size:tuple=(1,1)):
    """Draw random samples in a box area.
    """
    while True:
        yield np.random.rand(2)*size + pos


# def line(p1:tuple=(0.25,0.25), p2:tuple=(0.75,0.75)):
#     """Draw random samples on a line segment.
#     """
#     p1 = np.asarray(p1)
#     p2 = np.asarray(p2)
#     while True:
#         t = np.random.rand()
#         yield (1-t)*p1 + t*p2


def segments(P:list):
    #     p1:tuple=(0.25,0.25), p2:tuple=(0.75,0.25), p3:tuple=(0.5,0.683)
    while True:
        i = np.random.randint(len(P)-1)
        t = np.random.rand()
        yield (1-t)*P[i] + t*P[i+1]


def circle(pos:tuple=(0.5,0.5), radius:float=0.25, *, inside:bool=True):
    """Draw random samples in/on a circle.
    """
    while True:
#         p = np.random.randn(2); p /= np.linalg.norm(p)
        a = np.random.rand()*2*np.pi
        p = np.asarray([np.cos(a), np.sin(a)])
        r = np.random.rand() if inside else 1.; r *= radius
        yield p*r+pos


def point_set(xs:np.ndarray, pert:float=0):
    """Draw random samples from a given set.
    """
    N = len(xs)
    while True:
        i = np.random.randint(N)
        if pert>0:
            p = np.random.randn(2); p /= np.linalg.norm(p)
            yield xs[i] + p*np.random.rand()*pert
        else:
            yield xs[i]


# Not tested
def curve(xs:np.ndarray, gs:np.ndarray=None, *, step:float=0, axis:int=0, radius:float=0):
    while True:
        try:
            dist = np.linalg.norm(x0-xs, axis=-1)
            idx = np.where(dist>=step & ~visited)[0]
            i = idx[np.argmin(dist[idx])]
            visited[i] = True
        except:
            visited = np.zeros(xs.shape[axis], dtype=bool)
            i = np.random.randint(xs.shape[axis])
            visited[i] = True
        x0 = xs.take(i, axis=axis)

        if radius>0:
            try:
                s = 2*(np.random.rand()>0.5)-1  # get random sign
                x = x0 + s*np.random.rand()*radius*gs[i]
            except:
                s = np.random.randn(2)
                s /= np.linalg.norm(s)
                x = x0 + s*np.random.rand()*radius
        yield x


def constraint(pos:tuple, size:tuple, cfunc:callable):
    while True:
        x = box(pos, size)
        if cfunc(x):
            return x