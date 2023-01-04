#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <lapacke.h>
#include <cblas.h>

#include "mymath.h"

int main(int argc, char **argv)
{
  srand(getpid());

  matrix_t A;

  if (argc == 2) {
    A = matrix_readFromFile("../mat.dat");
  } else if (argc == 3) {
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);
    A = matrix_generateRandom(x, y);
  } else {
    A = matrix_generateRandom(10, 10);
  }

  if (A.data == NULL) {
    printf("Error: matrix is NULL\n");

  }

  printf("Matrix A:\n");
  matrix_print(&A);
  printf("\n\n");

  eigenData_t eigen = IRAM(&A, 3, 10, 1e-12f);

  printf("Eigenvalues:\n");
  printf("Real:\n");
  vector_print(&eigen.eigen_val_r);
  printf("Imaginary:\n");
  vector_print(&eigen.eigen_val_i);
  printf("\n\n");
  printf("Eigenvectors:\n");
  matrix_print(&eigen.eigen_vec);

  matrix_free(&A);
  matrix_free(&eigen.eigen_vec);
  vector_free(&eigen.eigen_val_r);
  vector_free(&eigen.eigen_val_i);

  return 0;
}