import numpy as np

def matrix_transpose(A):
    rows = len(A)
    cols = len(A[0])

    result = np.empty((cols, rows), dtype=np.array(A).dtype)

    for i in range(rows):
        for j in range(cols):
            result[j, i] = A[i][j]

    return result