import sys
import numpy as np
from _matrix_functions import read_matrix_market, read_raw_matrix, print_matrix


def main():
    if len(sys.argv) < 3:
        print(f"Usage: \n"
              f"\t| python3 {sys.argv[0]} <path_to_matrix_market_file.mm>\n"
              f"\t| python3 {sys.argv[0]} <path_to_other_format_matrix.data>\n")
        sys.exit(1)
    mat1 = read_raw_matrix(sys.argv[1])
    mat2 = read_raw_matrix(sys.argv[2])
    print_matrix(mat1, label="Matrix", precision=4)
    print_matrix(mat2, label="Matrix", precision=4)

    print_matrix(mat1.T * mat2.T, label="Q")


if __name__ == '__main__':
    main()
