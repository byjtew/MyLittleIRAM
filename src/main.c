#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <lapacke.h>
#include <cblas.h>

#include "mymath.h"

void print_results(eigenData_t eigen)
{
  printf("Eigenvalues:\n");
  for (size_t i = 0; i < eigen.eigen_val_r.n; i++)
  {
    printf("%ld: %f + %fi\n", i, eigen.eigen_val_r.data[i], eigen.eigen_val_i.data[i]);
    printf("\t");
    for (size_t j = 0; j < eigen.eigen_vec.column; j++)
    {
      printf("%f ", eigen.eigen_vec.data[i * eigen.eigen_vec.column + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  srand(getpid());

  MPI_Init(&argc, &argv);

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  matrix_t A;

  if (rank == 0)
  {

    if (argc == 2)
    {
      A = matrix_readFromFile(argv[1]);
    }
    else if (argc == 3)
    {
      int x = atoi(argv[1]);
      int y = atoi(argv[2]);
      A = matrix_generateRandom(x, y);
    }
    else
    {
      A = matrix_generateRandom(20, 20);
    }

    if (A.data == NULL)
    {
      printf("Error: matrix is NULL\n");
    }

    printf("Matrix A:\n");
    matrix_print_rowmajor(&A);
    printf("\n\n");
  }

  size_t A_row = rank == 0 ? A.row : 0;
  size_t A_column = rank == 0 ? A.column : 0;
  MPI_Bcast(&A_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&A_column, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0)
    A = matrix_create(A_row, A_column);
  MPI_Bcast(A.data, A_row * A_column, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // eigenData_t eigen = IRAM(&A, 3, 10, 1e-6);

  int best_rank = -1;
  eigenData_t eigen = MIRAM(&A, 3, 10, 1e-6, &best_rank);

  MPI_Barrier(MPI_COMM_WORLD);
  if (best_rank == rank)
  {
    printf("\n-- Best rank: %d\n", best_rank);
    print_results(eigen);
  }

  matrix_free(&A);
  matrix_free(&eigen.eigen_vec);
  vector_free(&eigen.eigen_val_r);
  vector_free(&eigen.eigen_val_i);

  MPI_Finalize();
  return 0;
}