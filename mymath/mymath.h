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
  double *data;
} vector_t;

typedef struct matrix_s
{
  size_t row, column;
  double *data;
} matrix_t;

typedef struct eigenData_s
{
  vector_t eigen_val_r;
  vector_t eigen_val_i;
  matrix_t eigen_vec;
} eigenData_t;

matrix_t matrix_create(const size_t n_row, const size_t n_column);
vector_t vector_create(const size_t n);

matrix_t matrix_copy(const matrix_t *matrix);
vector_t vector_copy(const vector_t *vector);

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column);
vector_t vector_generateRandom(const size_t n);

void matrix_fill(const matrix_t *matrix, const double value);
void vector_fill(const vector_t *vector, const double value);

void matrix_free(matrix_t *matrix);
void vector_free(vector_t *vector);

matrix_t matrix_read(FILE *file);
vector_t vector_read(FILE *file);

matrix_t matrix_readFromFile(const char *filename);
vector_t vector_readFromFile(const char *filename);

matrix_t matrix_write(FILE *file);
vector_t vector_write(FILE *file);

void matrix_print(const matrix_t *matrix);
void vector_print(const vector_t *vector);

const double vector_dotProduct(const vector_t *x, const vector_t *y);
const double vector_raw_dotProduct(const double *x, const double *y, const size_t n);

vector_t matrix_dotProduct(const matrix_t *matrix, const vector_t *vector);
void matrix_raw_dotProduct(const double *matrix, const double *vector, double *outVector, const size_t m, const size_t n);

const double vector_norme(const vector_t *x);
const double vector_raw_norme(const double *x, const size_t n);

const double matrix_norme(const matrix_t *x);

/**
 * @brief Arnoldi projection algo
 *
 * @param A Input matrix n*n
 * @param b Input vector n
 * @param subspace Krylov's space
 * @param Q Output matrix subspace+1*n
 * @param h Output matrix subspace*subspace+1
 */
void arnoldiProjection(size_t start_step, const matrix_t *A, const vector_t *f,
                       const size_t m, const matrix_t *V, const matrix_t *H) ;

eigenData_t IRAM(const matrix_t *A, const size_t n_eigen, const size_t max_iter, const double max_error);