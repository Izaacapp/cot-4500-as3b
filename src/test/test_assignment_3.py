import sys
import os
import numpy as np

# Add src/main to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main')))

from assignment_3 import (
    gaussian_elimination,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)

def test_gaussian_elimination():
    A = np.array([
        [2.0, -1.0, 1.0],
        [1.0,  3.0, 1.0],
        [-1.0, 5.0, 4.0]
    ])
    b = np.array([6.0, 0.0, -3.0])
    x = gaussian_elimination(A.copy(), b.copy())
    expected = np.array([2.0, -1.0, 1.0])
    assert np.allclose(x, expected)

def test_lu_factorization():
    matrix = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ], dtype=float)
    L, U, det = lu_factorization(matrix)

    expected_det = 39.0
    expected_L = np.array([
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 4, 1, 0],
        [-1, -3, 0, 1]
    ], dtype=float)
    expected_U = np.array([
        [1, 1, 0, 3],
        [0, -1, -1, -5],
        [0, 0, 3, 13],
        [0, 0, 0, -13]
    ], dtype=float)

    assert np.isclose(det, expected_det)
    assert np.allclose(L, expected_L)
    assert np.allclose(U, expected_U)

def test_diagonal_dominance():
    matrix = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    assert is_diagonally_dominant(matrix) == False

def test_positive_definite():
    matrix = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    assert is_positive_definite(matrix) == True
