import numpy as np



def gaussian_elimination(A, b):
    n = len(b)
    
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j][i:] -= factor * A[i][i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]

    return x


def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = matrix.astype(float).copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            U[j] -= factor * U[i]

    determinant = np.prod(np.diag(U))
    return L, U, determinant



def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False



def main():
    A1 = np.array([
        [2.0, -1.0, 1.0],
        [1.0, 3.0, 1.0],
        [-1.0, 5.0, 4.0]
    ])
    b1 = np.array([6.0, 0.0, -3.0])
    solution = gaussian_elimination(A1.copy(), b1.copy())
    for val in solution:
        print(val)
    print(np.round(solution, 6))

    matrix_q2 = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ], dtype=float)
    L, U, det = lu_factorization(matrix_q2)
    print(det)
    print(L)
    print(U)

    
   
    matrix_dd = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    print(is_diagonally_dominant(matrix_dd))

    
   
    matrix_pd = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    print(is_positive_definite(matrix_pd))


if __name__ == "__main__":
    main()
