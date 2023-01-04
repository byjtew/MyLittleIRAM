import numpy as np


def print_matrix(mat: np.ndarray, label: str = "", precision: int = 4) -> None:
    print(label, mat.shape)
    print(np.round(mat, precision), "\n")


def read_matrix_market(path: str) -> np.ndarray:
    raise NotImplementedError("Matrix Market format is not supported yet.")
    mat = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("%"):
                continue
            if mat is None:
                mat = np.zeros((int(line.split()[0]), int(line.split()[1])))
                continue
            mat[int(line.split()[0]) - 1, int(line.split()[1]) - 1] = float(line.split()[2])
    return mat


def read_raw_matrix(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        # Read the first line to get the shape of the matrix
        line = f.readline()
        mat = np.zeros((int(line.split()[0]), int(line.split()[1])))
        row = 0
        for line in f:
            for col, val in enumerate(line.split()):
                mat[row, col] = float(val)
            row += 1
    return mat
