import sys
import numpy as np
from read_matrix_market import read_matrix_market


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <matrix_path>\n")
        sys.exit(1)
    mat = read_matrix_market(sys.argv[1])
    w, v = np.linalg.eig(mat)
    print("Eigenvalues:", w)
    print("Eigenvectors:", v)


if __name__ == '__main__':
    main()
