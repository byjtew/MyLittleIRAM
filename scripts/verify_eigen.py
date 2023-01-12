import sys
import numpy as np
from _matrix_functions import read_matrix_market, read_raw_matrix, print_matrix

def main():
    if len(sys.argv) < 2:
        print(f"Usage: \n"
              f"\t| python3 {sys.argv[0]} <path_to_matrix_market_file.mm>\n"
              f"\t| python3 {sys.argv[0]} <path_to_other_format_matrix.data>\n")
        sys.exit(1)
    mat = read_matrix_market(sys.argv[1]) if sys.argv[1].endswith(".mm") else read_raw_matrix(sys.argv[1])
    print_matrix(mat, label="Matrix", precision=4)

    w, v = np.linalg.eig(mat)
    #w = w[w[:, 0].argsort()]
    w = np.sort_complex(w)
    for i, e in enumerate(w):
        print(f"{i}: {e}")

if __name__ == '__main__':
    main()
