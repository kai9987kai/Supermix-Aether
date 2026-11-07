import numpy as np
import pytest

from aether.neurowave import graph_modes


def laplacian(adjacency):
    symmetric = (adjacency + adjacency.T) / 2
    degree = symmetric.sum(axis=1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1 / np.sqrt(degree[degree > 0])
    return np.diag((degree > 0).astype(float)) - inv[:, None] * symmetric * inv[None, :]


def test_directed_graph_uses_symmetric_normalized_laplacian():
    adjacency = np.array([[0., 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    values, basis = graph_modes(adjacency, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(laplacian(adjacency) @ basis, basis * values, atol=1e-12)
    assert np.all(np.diff(values) >= -1e-12)
    assert np.count_nonzero(abs(values) < 1e-12) == 2
    _, again = graph_modes(adjacency, 4)
    np.testing.assert_array_equal(basis, again)


def test_rotated_degenerate_previous_is_preserved_even_at_truncated_boundary():
    adjacency = np.ones((5, 5)) - np.eye(5)
    values, basis = graph_modes(adjacency, 5)
    rotation, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(4, 4)))
    previous = np.column_stack((basis[:, 0], basis[:, 1:] @ rotation))[:, :3]
    values, aligned = graph_modes(adjacency, 3, previous=previous)
    np.testing.assert_allclose(aligned, previous, atol=1e-12)
    np.testing.assert_allclose(laplacian(adjacency) @ aligned, aligned * values, atol=1e-12)


def test_alignment_does_not_mix_distinct_eigenvalues():
    adjacency = np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1)
    values, basis = graph_modes(adjacency, 6)
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(6, 6)))
    _, aligned = graph_modes(adjacency, 6, previous=basis @ rotation)
    np.testing.assert_allclose(laplacian(adjacency) @ aligned, aligned * values, atol=1e-12)
    np.testing.assert_allclose(aligned.T @ aligned, np.eye(6), atol=1e-12)


def test_zero_graph_and_rank_deficient_projection_remain_orthonormal():
    adjacency = np.zeros((5, 5))
    values, basis = graph_modes(adjacency, 3)
    np.testing.assert_array_equal(values, np.zeros(3))
    np.testing.assert_allclose(basis.T @ basis, np.eye(3))
    _, basis = graph_modes(np.diag(np.ones(4), 1), 3, previous=basis)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_nearly_equal_but_distinct_carriers_are_not_rotated_together():
    adjacency = np.array([[0., 1, 1e-14, 0], [1, 0, 0, 0],
                          [1e-14, 0, 0, 1], [0, 0, 1, 0]])
    values, eigenvectors = np.linalg.eigh(laplacian(adjacency))
    previous = eigenvectors[:, :2] @ np.array([[1., -1], [1, 1]]) / np.sqrt(2)
    values, basis = graph_modes(adjacency, 2, previous=previous)
    assert np.linalg.norm(laplacian(adjacency) @ basis - basis * values) < 2e-15


@pytest.mark.parametrize('adjacency,count,previous', [
    (np.ones((2, 3)), 1, None), (np.full((2, 2), np.nan), 1, None),
    (-np.ones((2, 2)), 1, None), (np.eye(2), 0, None),
    (np.eye(2), 3, None), (np.eye(2), True, None),
    (np.eye(2), 1, np.ones((3, 1))), (np.eye(2), 1, np.full((2, 1), np.inf)),
    (np.eye(2), 1, np.array([[1+1j], [0]])),
])
def test_invalid_modes_are_rejected(adjacency, count, previous):
    with pytest.raises((TypeError, ValueError)):
        graph_modes(adjacency, count, previous)
