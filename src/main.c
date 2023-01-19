#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <lapacke.h>
#include <cblas.h>

#include "mymath.h"


void print_results(eigenData_t eigen) {
  printf("Eigenvalues:\n");
  for (size_t i = 0; i < eigen.eigen_val_r.n; i++) {
    printf("%ld: %f + %fi\n", i, eigen.eigen_val_r.data[i], eigen.eigen_val_i.data[i]);
    printf("\t");
    for (size_t j = 0; j < eigen.eigen_vec.column; j++) {
      printf("%f ", eigen.eigen_vec.data[i * eigen.eigen_vec.column + j]);
    }
    printf("\n");
  }

}

int main(int argc, char **argv)
{
  srand(getpid());

  matrix_t A;

  if (argc == 2) {
    A = matrix_readFromFile(argv[1]);
  } else if (argc == 3) {
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);
    A = matrix_generateRandom(x, y);
  } else {
    A = matrix_generateRandom(50, 50);
  }

  if (A.data == NULL) {
    printf("Error: matrix is NULL\n");
  }

  printf("Matrix A:\n");
  matrix_print(&A);
  printf("\n\n");

  eigenData_t eigen = IRAM(&A, 3, 10, 1e-6);
  print_results(eigen);

  matrix_free(&A);
  matrix_free(&eigen.eigen_vec);
  vector_free(&eigen.eigen_val_r);
  vector_free(&eigen.eigen_val_i);

  /*size_t m = 3;

  matrix_t A = matrix_readFromFile(argv[1]);
  vector_t b = vector_readFromFile(argv[2]);
  
  matrix_t V = matrix_create(A.row, m + 1);
  matrix_t H = matrix_create(m + 1, m);

  matrix_print(&A);
  vector_print(&b);

  arnoldiProjection(1, &A, &b, m, &V, &H);

  matrix_print(&V);
  matrix_print(&H);*/

  return 0;
}