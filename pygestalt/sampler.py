"""Random samplers for visual stimuli.
"""

from typing import Callable, Iterator, NewType, Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy import stats, special


Point = ArrayLike | Sequence[float]
# Point = tuple[float, float]
# Point = NewType('Point', ArrayLike|tuple[float...])

def distances(P:ArrayLike, Q:ArrayLike) -> ArrayLike:
    """Compute the euclidean distance between every pair of points from two sets.

    Args:
        P: first array, of shape `Np x d`, with `d` the dimension of the space
        Q: second array, of shape `Nq x d`

    Return:
        a matrix of shape `Np x Nq` where the value at `(r,c)` repesents the
        distance from the point `P[r]` to the point `Q[c]`.
    """
    assert P.ndim == Q.ndim == 2
    assert P.shape[1] == Q.shape[1]

    return np.linalg.norm(P[:,None,:]-Q[None,...], axis=-1)


def draw_positions(radius:float, sampler:Iterator[Point], *,
                   exclusions:ArrayLike=np.empty((0,2)),
                   thresh:float=1e-3) -> Sequence[Point]:
    """Draw random positions of balls incrementally until convergence.

    Args:
        radius: radius of balls
        sampler: a generator yielding random points and optionally tangent directions
        exclusions: a set of coordinates to be excluded
        thresh: threshold for the sampling efficiency

    Returns:
        an array of points, optionally an array of tangent directions if provided by `sampler`
    """
    # initiation
    X = np.empty((0,2))  # positions
    H = np.empty((0,2))  # tangents
    niter = 0

    for foo in sampler:
        if isinstance(foo, tuple):
            # if the sample provides also the tangent
            x, h = foo
        else:
            x = foo

        # draw N points
        x = np.atleast_2d(x)
        # The candidate ball must not touch the exclusion balls
        dist = distances(x, exclusions)
        if not(dist.size == 0 or np.min(dist) >= 2.*radius):
            continue

        niter += 1
        # distance of the candidate point to existing points
        dist = distances(x, X)
        idx = np.where(np.all(2.*radius < dist, axis=0))

        # The candidate ball must not touch more than N of existing balls
        if len(idx[0]) >= len(X)-len(x):
            # `>=` behaves differently from `>`
            X = np.vstack([X[idx].reshape(-1,2),x])
            try:
                H = np.vstack([H[idx].reshape(-1,2), h])
            except:
                pass

        # exit if the current sampling becomes inefficient
        if len(X)/niter < thresh:
            break

    return X, H


def as_multiple(N:int=1, radius:float=None):
    """Make a sampler of single output to `N` outputs by keeping the
    minimum distance of `2*radius` between points.
    """
    def decorator(original_sampler):
        def wrapper(*args, **kwargs):
            iterator = original_sampler(*args, **kwargs)
            # P = np.empty((0,2), dtype=float)
            P = []
            n = 0
            for x in iterator:
                if N > 1 and radius is not None:
                    dist = np.linalg.norm(np.asarray(P).reshape(-1,2)-x, axis=-1)
                    if not(dist.size==0 or np.min(dist) >= 2.*radius):
                        continue
                P.append(x)
                n += 1
                if n%N==0:
                    yield np.asarray(P)
                    P = []
            # if len(P)>0:
            #     yield P
        return wrapper
    return decorator


def bernstein_basis(N, k):
    """Bernstein polynomial."""
    return lambda t: np.atleast_1d(special.comb(N,k) * (t**k) * ((1-t)**(N-k)))

def bezier_curve_position(t:float|ArrayLike, Ps:list[Point]):
    """Bezier curve of arbitrary order.

    Args:
        t: curve parameter between 0 and 1
        Ps: list of control points, with the first and the last point being the starting and the ending point.

    Reference:
        https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    """
    # assert 0<=t<=1
    N = len(Ps)-1

    for k, P in enumerate(Ps):
        try:
            B += bernstein_basis(N, k)(t)[:,None] * P[None,:]
        except:
            B = bernstein_basis(N, k)(t)[:,None] * P[None,:]

    return B

def bezier_curve_derivative(t:float|ArrayLike, Ps:list[Point]):
    """Tangent of an Bezier curve."""

    # assert 0<=t<=1
    N = len(Ps)-1

    for k in range(N):
        try:
            B += bernstein_basis(N-1, k)(t)[:,None] * (Ps[k+1] - Ps[k])[None,:]
        except:
            B = bernstein_basis(N-1, k)(t)[:,None] * (Ps[k+1] - Ps[k])[None,:]

    return N*B

def bezier_curve(P:list[Point]) -> Iterator[Point]:
    """Draw random points on a Bezier curve, with the tangents."""
    d = stats.uniform()

    while True:
        t = d.rvs()
        yield (bezier_curve_position(t, P), bezier_curve_derivative(t, P))


def polygone(P:list[Point], alpha:float=1.) -> Iterator[Point]:
    """Draw random samples in a polygone area.

    Args:
        P: list of points defining the polygone
        alpha: hyper-parameter of a Dirichlet distribution

    Yields:
        a random point inside the polygone.
    """
    # symmetric Dirichlet distribution
    d = stats.dirichlet(np.ones(len(P))*alpha)
    while True:
        # draw a random sample
        t = d.rvs()
        yield np.sum(t[:,None]*P, axis=0)


def box(pos:tuple=(0,0), size:tuple=(1,1)) -> Iterator[Point]:
    """Draw random samples in a box area.

    Args:
        pos: lower-left corner of the box.
        size: width and height of the box.

    Yields:
        a random point inside the box.
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


def segments(P:list) -> Iterator[Point]:
    """Draw random samples on line segments, with tangents.

    Args:
        P: list of segments' end points

    Yields:
        a random point on the line segments
    """
    # Note that a triangle can be defined as follows:
    # p1:tuple=(0.25,0.25), p2:tuple=(0.75,0.25), p3:tuple=(0.5,0.683), p4=p1
    while True:
        # first draw a random segment from the given ones
        i = np.random.randint(len(P)-1)
        # next draw a random point on that segment
        t = np.random.rand()
        yield (1-t)*P[i] + t*P[i+1], P[i+1]-P[i]


def circle(pos:tuple=(0.5,0.5), radius:float=0.25, *, inside:bool=True) -> Iterator[Point]:
    """Draw random samples in/on a circle, with tangents.
    """
    while True:
#         p = np.random.randn(2); p /= np.linalg.norm(p)
        a = np.random.rand()*2*np.pi
        p = np.asarray([np.cos(a), np.sin(a)])
        r = np.random.rand() if inside else 1.; r *= radius
        yield p*r+pos, np.asarray([np.sin(a), -np.cos(a)])


def point_set(xs:Sequence[Point], pert:float=0) -> Iterator[Point]:
    """Draw random samples from a given set.

    Args:
        xs: a set of points
        pert: perturbation of the random sample
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
def curve(xs:ArrayLike, gs:ArrayLike=None, *, step:float=0, axis:int=0, radius:float=0) -> Iterator[Point]:
    """Conditional sampling on a curve.
    """
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