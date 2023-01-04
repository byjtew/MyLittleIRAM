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

  matrix_t A = matrix_readFromFile("../mat.dat");
  // matrix_t A = matrix_readFromFile("../data/mat_verif1.dat");
  // matrix_t A = matrix_generateRandom(10, 10);

  matrix_print(&A);
  printf("\n\n");

  eigenData_t eigen = IRAM(&A, 3, 10, 1e-12f);

  vector_print(&eigen.eigen_val_r);
  vector_print(&eigen.eigen_val_i);
  matrix_print(&eigen.eigen_vec);

  matrix_free(&A);
  matrix_free(&eigen.eigen_vec);
  vector_free(&eigen.eigen_val_r);
  vector_free(&eigen.eigen_val_i);

  return 0;
}