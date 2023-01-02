import numpy as np


def read_matrix_market(path: str) -> np.ndarray:
    mat = None
    with open(path, 'rb') as f:
        for line in f:
            if line.startswith(b'%'):
                continue
            if mat is None:
                mat = np.zeros((int(line.split()[0]), int(line.split()[1])))
                continue
            mat[int(line.split()[0]) - 1, int(line.split()[1]) - 1] = float(line.split()[2])
    return mat
