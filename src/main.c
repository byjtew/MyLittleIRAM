#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
#include <lapacke.h>
#include <cblas.h>

#include "mymath.h"

#define MAX_ITERATIONS 100
#define MIN_ERROR 1e-6

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
  MPI_Init(&argc, &argv);

  size_t nb_eigen_values; // Default value
  srand(getpid());

  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  matrix_t A;

  int master_failed = 0;
  if (rank == 0)
  {
    if (argc == 4) // exec <taille_sous_espace> --input <fichier_matrice>
    {
      nb_eigen_values = atol(argv[1]);
      A = matrix_readFromFile(argv[3]);
    }
    else if (argc == 3) // exec <taille_sous_espace> <taille_matrice>
    {
      printf("ici\n");
      nb_eigen_values = atol(argv[1]);
      size_t n = atoi(argv[2]);
      A = matrix_generateRandom(n, n);
    }
    else
    {
      master_failed = 1;
      fprintf(stderr, "Usage: \n\t%s <taille_sous_espace> --input <fichier_matrice>\n\t%s <taille_sous_espace> <taille_matrice>\n",
              argv[0], argv[0]);
    }

    if (!master_failed)
    {
      printf("Matrix A:\n");
      matrix_print_rowmajor(&A);
      printf("\n\n");
    }
  }

  MPI_Bcast(&master_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (master_failed)
  {
    MPI_Finalize();
    return 1;
  }

  // Create and sychronize matrix A & nb_eigen_values
  size_t A_row = rank == 0 ? A.row : 0;
  size_t A_column = rank == 0 ? A.column : 0;
  MPI_Bcast(&A_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&A_column, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0)
    A = matrix_create(A_row, A_column);
  MPI_Bcast(A.data, A_row * A_column, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nb_eigen_values, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int best_rank = -1;
  eigenData_t eigen = MIRAM(&A, nb_eigen_values, MAX_ITERATIONS, MIN_ERROR, &best_rank);

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