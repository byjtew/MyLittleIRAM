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

  //matrix_t A = matrix_readFromFile("../data/mat.dat");
  matrix_t A = matrix_readFromFile("../data/mat_verif1.dat");
  //matrix_t A = matrix_generateRandom(30, 30);

  matrix_print(&A);
  printf("\n\n");

  eigenData_t eigen = ERAM(&A, 3, 500, 0.0001);

  vector_print(&eigen.eigen_val_r);
  vector_print(&eigen.eigen_val_i);
  matrix_print(&eigen.eigen_vec);

  matrix_free(&A);
  matrix_free(&eigen.eigen_vec);
  vector_free(&eigen.eigen_val_r);
  vector_free(&eigen.eigen_val_i);

  return 0;
}