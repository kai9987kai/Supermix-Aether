"""Normalized-Laplacian carriers with valid degenerate-space alignment."""

import numpy as np

from .protocol import _integer


def _canonical_columns(space, count, previous=None):
    """Project prior columns, then complete using deterministic coordinate pivots."""
    n = space.shape[0]
    chosen = []

    def residual(vector):
        for _ in range(2):
            for column in chosen:
                vector = vector - column * np.dot(column, vector)
        return vector

    if previous is not None:
        for column in previous.T:
            projected = residual(space @ (space.T @ column))
            norm = np.linalg.norm(projected)
            if norm > 1e-10:
                chosen.append(projected / norm)
            else:
                # A prior carrier can leave this eigenspace after graph change.
                # Complete its slot canonically; never mix eigenvalue groups.
                candidates = [residual(space @ space[index, :]) for index in range(n)]
                norms = np.array([np.linalg.norm(v) for v in candidates])
                best = int(np.argmax(norms))
                chosen.append(candidates[best] / norms[best])
            if len(chosen) == count:
                return np.column_stack(chosen)
    while len(chosen) < count:
        candidates = [residual(space @ space[index, :]) for index in range(n)]
        norms = np.array([np.linalg.norm(v) for v in candidates])
        # Roundoff ties must not select different coordinate pivots on a rerun.
        best = int(np.flatnonzero(norms >= norms.max() - 1e-13)[0])
        column = candidates[best] / norms[best]
        pivot = int(np.argmax(abs(column)))
        if column[pivot] < 0:
            column = -column
        chosen.append(column)
    return np.column_stack(chosen)


def graph_modes(adjacency, mode_count, previous=None):
    """Return sorted eigenvalues and an (n, mode_count) orthonormal basis.

    Directed nonnegative adjacency is symmetrized as (A+A.T)/2. The symmetric
    normalized Laplacian has zero rows/diagonal for isolated vertices. Equal
    eigenvalues are grouped only to floating-point eigensolver tolerance;
    alignment projects prior (n, mode_count) carriers inside each full equal
    eigenspace, including when mode_count cuts through a degenerate group.
    Distinct eigenvalues are never jointly rotated. Without prior carriers,
    projected coordinate pivots and sign canonicalization fix the basis.
    """
    raw = np.asarray(adjacency)
    if np.iscomplexobj(raw):
        raise ValueError('adjacency must be real')
    adjacency = np.asarray(raw, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not len(adjacency):
        raise ValueError('adjacency must be a nonempty square matrix')
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
        raise ValueError('adjacency must be finite and nonnegative')
    mode_count = _integer(mode_count, 'mode_count', 1)
    n = len(adjacency)
    if mode_count > n:
        raise ValueError('mode_count exceeds graph size')
    if previous is not None:
        if np.iscomplexobj(previous):
            raise ValueError('previous carriers must be real')
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (n, mode_count) or not np.isfinite(previous).all():
            raise ValueError('previous must be finite and have shape (n, mode_count)')
        if not np.allclose(previous.T @ previous, np.eye(mode_count), atol=1e-8, rtol=0):
            raise ValueError('previous must have orthonormal columns')
    # Scaling before symmetrization prevents overflow and does not change L.
    maximum = adjacency.max()
    scaled = adjacency / maximum if maximum else adjacency
    symmetric = (scaled + scaled.T) * .5
    degree = symmetric.sum(axis=1)
    inv = np.zeros(n)
    inv[degree > 0] = 1 / np.sqrt(degree[degree > 0])
    laplacian = np.diag((degree > 0).astype(float)) - inv[:, None] * symmetric * inv[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    # A size-scaled loose tolerance would incorrectly rotate resolvably distinct
    # low-frequency carriers of a weakly connected graph.
    tolerance = 8 * np.finfo(float).eps * max(1., np.max(abs(values)))
    result = []
    start = 0
    while start < mode_count:
        end = start + 1
        while end < n and abs(values[end] - values[start]) <= tolerance:
            end += 1
        count = min(end, mode_count) - start
        prior = previous[:, start:start + count] if previous is not None else None
        result.append(_canonical_columns(vectors[:, start:end], count, prior))
        start = end
    return values[:mode_count].copy(), np.column_stack(result)
