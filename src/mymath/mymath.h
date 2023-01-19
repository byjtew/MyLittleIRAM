#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <float.h>
#include <string.h>

typedef struct vector_s
{
  size_t n;
  double *restrict data;
} vector_t;

typedef struct matrix_s
{
  size_t row, column;
  double *restrict data;
} matrix_t;

typedef struct eigenData_s
{
  vector_t eigen_val_r;
  vector_t eigen_val_i;
  matrix_t eigen_vec;
} eigenData_t;

matrix_t matrix_create(const size_t n_row, const size_t n_column);
vector_t vector_create(const size_t n);

matrix_t matrix_copy(matrix_t *restrict dest, const matrix_t *restrict src);
vector_t vector_copy(const vector_t *restrict vector);

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column);
vector_t vector_generateRandom(const size_t n);

void matrix_fill(const matrix_t *restrict matrix, const double value);
void vector_fill(const vector_t *restrict vector, const double value);

void matrix_free(matrix_t *restrict matrix);
void vector_free(vector_t *restrict vector);

matrix_t matrix_read(FILE *restrict file);
vector_t vector_read(FILE *restrict file);

matrix_t matrix_readFromFile(const char *restrict filename);
vector_t vector_readFromFile(const char *restrict filename);

matrix_t matrix_write(FILE *restrict file);
vector_t vector_write(FILE *restrict file);

void matrix_print(const matrix_t *restrict matrix);
void vector_print(const vector_t *restrict vector);

/**
 * @brief Arnoldi projection algo
 *
 * @param A Input matrix n*n
 * @param b Input vector n
 * @param subspace Krylov's space
 * @param Q Output matrix subspace+1*n
 * @param h Output matrix subspace*subspace+1
 */
void arnoldiProjection(size_t start_step, const matrix_t *restrict A, const vector_t *restrict f,
                       const size_t m, const matrix_t *restrict V, const matrix_t *restrict H,
                       const vector_t *restrict buffer);

eigenData_t IRAM(const matrix_t *restrict A, const size_t n_eigen, const size_t max_iter, const double max_error);