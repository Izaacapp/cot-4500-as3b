import numpy as np
from src.main.assignment_3 import (
    gaussian_elimination,
    lu_decomposition,
    is_diagonally_dominant,
    is_positive_definite,
)


def test_gaussian_elimination():
    A = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
    b = np.array([6, 0, -3])
    result = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_lu_decomposition_rebuilds_matrix():
    A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
    L, U = lu_decomposition(A)
    reconstructed = L @ U
    np.testing.assert_allclose(reconstructed, A, rtol=1e-5)


def test_lu_determinant_matches_numpy():
    A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
    det_actual = np.linalg.det(A)
    L, U = lu_decomposition(A)
    det_from_lu = np.prod(np.diag(U))
    assert abs(det_actual - det_from_lu) < 1e-10


def test_diagonal_dominance_true():
    A = np.array(
        [
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8],
        ]
    )
    assert is_diagonally_dominant(A) is True


def test_positive_definite_true():
    A = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
    assert is_positive_definite(A) is True
