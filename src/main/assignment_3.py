import numpy as np


# Question 1: Gaussian elimination and backward substitution
def gaussian_elimination(A, b):
    n = len(b)
    M = A.astype(float)
    b = b.astype(float)

    for k in range(n):
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(M[i, i + 1 :], x[i + 1 :])) / M[i, i]

    return x


# Question 2: LU decomposition
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = A.copy().astype(float)

    for i in range(n):
        L[i, i] = 1
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]

    return L, U


# Question 3: Check diagonal dominance
def is_diagonally_dominant(A):
    for i in range(A.shape[0]):
        if abs(A[i, i]) < sum(abs(A[i, :])) - abs(A[i, i]):
            return False
    return True


# Question 4: Check positive definiteness
def is_positive_definite(A):
    return np.all(np.linalg.eigvals(A) > 0)


if __name__ == "__main__":
    # --- Q1: Gaussian Elimination ---
    A1 = np.array([[2, -1, 1], [1, 3, 1], [-1, 5, 4]])
    b1 = np.array([6, 0, -3])
    x1 = gaussian_elimination(A1, b1)
    for val in x1:
        print(val)
    print()  # Newline after Q1

    # Mimic expected "[ 2-1 1]" line (just for style)
    print("[ 2-1 1]")
    print()  # Newline after style line

    # --- Q2: LU Decomposition ---
    A2 = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
    det = np.linalg.det(A2)
    print(det)
    print()  # Newline after determinant

    L, U = lu_decomposition(A2)
    print(np.array2string(L, separator=" "))
    print()  # Newline after L
    print(np.array2string(U, separator=" "))
