#include "mymath.h"

matrix_t matrix_create(const size_t n_row, const size_t n_column)
{
  matrix_t res;

  res.data = malloc(sizeof(double) * n_row * n_column);
  assert(res.data);

  res.row = n_row;
  res.colunm = n_column;

  return res;
}

vector_t vector_create(const size_t n)
{
  vector_t res;

  res.data = malloc(sizeof(double) * n);
  assert(res.data);

  res.n = n;

  return res;
}

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column)
{
  matrix_t res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
    res.data[i] = (double)rand() / (double)RAND_MAX;

  return res;
}

vector_t vector_generateRandom(const size_t n)
{
  vector_t res = vector_create(n);

  for (size_t i = 0; i < n; i++)
    res.data[i] = (double)rand() / (double)RAND_MAX;

  return res;
}

void matrix_fill(const matrix_t *matrix, const double value)
{
  assert(matrix);

  assert(matrix->data);
  assert(matrix->row != 0);
  assert(matrix->colunm != 0);

  for (size_t i = 0; i < matrix->row * matrix->colunm; i++)
    matrix->data[i] = value;
}

void vector_fill(const vector_t *vector, const double value)
{
  assert(vector);

  assert(vector->data);
  assert(vector->n != 0);

  for (size_t i = 0; i < vector->n; i++)
    vector->data[i] = value;
}

void matrix_free(matrix_t *matrix)
{
  assert(matrix);

  free(matrix->data);
  matrix->data = NULL;
  matrix->row = matrix->colunm = 0;
}

void vector_free(vector_t *vector)
{
  assert(vector);

  free(vector->data);
  vector->data = NULL;
  vector->n = 0;
}

matrix_t matrix_read(FILE *file)
{
  matrix_t res;
  size_t n_row, n_column;

  fscanf(file, "%lu %lu", &n_row, &n_column);
  res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
    fscanf(file, "%lf", res.data + i);

  return res;
}

vector_t vector_read(FILE *file)
{
  vector_t res;
  unsigned n;
  fscanf(file, "%u", &n);
  res = vector_create(n);

  for (size_t i = 0; i < n; i++)
    fscanf(file, "%lf", res.data + i);

  return res;
}

matrix_t matrix_readFromFile(const char *filename)
{
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  matrix_t res = matrix_read(fp);
  fclose(fp);

  return res;
}

vector_t vector_readFromFile(const char *filename)
{
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  vector_t vec = vector_read(fp);
  fclose(fp);
  return vec;
}

void matrix_print(const matrix_t *matrix)
{
  printf("[%lu, %lu]\n", matrix->row, matrix->colunm);
  for (size_t i = 0; i < matrix->row; i++)
  {
    for (size_t j = 0; j < matrix->colunm; j++)
    {
      printf("%lf ", matrix->data[i * matrix->colunm + j]);
    }
    printf("\n");
  }
}

void vector_print(const vector_t *vector)
{
  printf("[%lu]\n", vector->n);
  for (size_t i = 0; i < vector->n; i++)
    printf("%lf\n", vector->data[i]);
}

const double vector_dotProduct(const vector_t *x, const vector_t *y)
{
  assert(x->n == y->n);

  double res = 0.0;
  for (size_t i = 0; i < x->n; i++)
    res += x->data[i] * y->data[i];

  return res;
}

const double vector_raw_dotProduct(const double *x, const double *y, const size_t n)
{
  assert(x);
  assert(y);

  double res = 0.0;
  for (size_t i = 0; i < n; i++)
    res += x[i] * y[i];

  return res;
}

vector_t matrix_dotProduct(const matrix_t *matrix, const vector_t *vector)
{
  assert(matrix);
  assert(vector);
  assert(matrix->row == matrix->colunm);
  assert(matrix->row == vector->n);

  const size_t size = matrix->row;
  vector_t res = vector_create(size);

  for (size_t i = 0; i < size; i++)
  {
    res.data[i] = 0.0;
    for (size_t j = 0; j < size; j++)
      res.data[i] += matrix->data[i * size + j] * vector->data[j];
  }

  return res;
}

void matrix_raw_dotProduct(const double *matrix, const double *vector, double *outVector, const size_t m, const size_t n)
{
  // matrix (m*n)
  // vector (1*n)

  assert(matrix);
  assert(vector);
  assert(m != 0);
  assert(n != 0);

  const size_t size = m;

  for (size_t i = 0; i < size; i++)
  {
    outVector[i] = 0.0;
    for (size_t j = 0; j < size; j++)
      outVector[i] += matrix[i * size + j] * vector[j];
  }
}

const double vector_norme(const vector_t *x)
{
  double res = 0.0;

  for (size_t i = 0; i < x->n; i++)
  {
    res += x->data[i] * x->data[i];
  }
  return sqrt(res);
}

const double vector_raw_norme(const double *x, const size_t n)
{
  double res = 0.0;

  for (size_t i = 0; i < n; i++)
  {
    res += x[i] * x[i];
  }
  return sqrt(res);
}

const double matrix_norme(const matrix_t *x)
{
  double res = 0.0;

  for (size_t i = 0; i < x->row; i++)
  {
    for (size_t j = 0; j < x->colunm; j++)
    {
      res += x->data[i * x->colunm + j] * x->data[i * x->colunm + j];
    }
  }
  return sqrt(res);
}

void arnoldiProjection(const matrix_t *A, const vector_t *b, const size_t subspace, const matrix_t *Q, const matrix_t *h)
{
  assert(A);
  assert(b);
  assert(Q);
  assert(h);

  assert(A->row == A->colunm);
  assert(A->row == b->n);
  assert(Q->row == (subspace + 1));
  assert(Q->colunm == A->row);
  assert(h->row == subspace);
  assert(h->colunm == (subspace + 1));

  const size_t size = A->row;
  const double epsilon = 1e-12;

  const double norme_b = vector_norme(b);
  for (size_t i = 0; i < size; i++)
    Q->data[i] = b->data[i] / norme_b;

  vector_t v = vector_create(size);
  for (size_t k = 1; k < subspace + 1; k++)
  {
    matrix_raw_dotProduct(A->data, Q->data + ((k - 1) * A->colunm), v.data, A->row, A->colunm);

    for (size_t j = 0; j < k; j++)
    {
      h->data[(k - 1) * h->colunm + j] = vector_raw_dotProduct(Q->data + (j * Q->colunm), v.data, size);

      for (size_t i = 0; i < v.n; i++)
        v.data[i] = v.data[i] - h->data[(k - 1) * h->colunm + j] * Q->data[j * Q->colunm + i];
    }
    h->data[(k - 1) * h->colunm + k] = vector_norme(&v);

    if (h->data[(k - 1) * h->colunm + k] > epsilon)
    {
      for (size_t i = 0; i < A->colunm; i++)
        Q->data[k * Q->colunm + i] = v.data[i] / h->data[(k - 1) * h->colunm + k];
    }
    else
      return;
  }
}

void ERAM_computeEigenSubspace(const matrix_t *h, const vector_t *eigen_values_r, const vector_t *eigen_values_i, const matrix_t *Z)
{
  // T Z eigenValue = computeEigenValue(h)
  LAPACKE_dhseqr(LAPACK_COL_MAJOR, 'S', 'I', h->row, 1, h->row, h->data, h->colunm, eigen_values_r->data, eigen_values_i->data, Z->data, h->row);

  // eigenVectorSubspace = computeEigenVector(h, Z)
  int m = 0;
  LAPACKE_dtrevc(LAPACK_COL_MAJOR, 'R', 'B', NULL, h->row, h->data, h->colunm, NULL, 1, Z->data, Z->row, Z->colunm, &m);
}

const double ERAM_computeError(const matrix_t *eigen_vectors, const double h_factor)
{
  double error = 0.0;
  for (size_t i = 0; i < eigen_vectors->row; i++)
    error += fabs(eigen_vectors->data[i * eigen_vectors->colunm + (eigen_vectors->colunm - 1)]);
  error *= h_factor;
  return error;
}

void ERAM_computeNewInputVector(const vector_t *input, const matrix_t *eigen_vectors)
{
  for (size_t i = 0; i < input->n; i++)
    input->data[i] = 0.0;

  for (size_t i = 0; i < eigen_vectors->row; i++)
    cblas_daxpy(eigen_vectors->colunm, 1.0, eigen_vectors->data + (i * eigen_vectors->colunm), 1, input->data, 1);

  double norm = vector_raw_norme(input->data, input->n);
  for (size_t i = 0; i < input->n; i++)
    input->data[i] *= norm;
}

eigenData_t ERAM(const matrix_t *A, const size_t n_eigen, const size_t max_iter, const double max_error)
{
  // size_t subspace = n_eigen * 2;
  const size_t subspace = n_eigen;

  vector_t b = vector_generateRandom(A->row);

  matrix_t Q = matrix_create(subspace + 1, A->row);
  matrix_t h = matrix_create(subspace, subspace + 1);

  matrix_t T = matrix_create(h.row, h.colunm);
  matrix_t Z = matrix_create(h.row, h.row);

  eigenData_t eigen;
  eigen.eigen_val_r = vector_create(subspace);
  eigen.eigen_val_i = vector_create(subspace);
  eigen.eigen_vec = matrix_create(subspace, A->row);

  double error = DBL_MAX;
  size_t count_iter = 0;
  while (fabs(error) > max_error && count_iter < max_iter)
  {
    arnoldiProjection(A, &b, subspace, &Q, &h);
    const double h_factor = h.data[(h.row * h.colunm) - 1];

    // Compute eigen vectors of h
    ERAM_computeEigenSubspace(&h, &eigen.eigen_val_r, &eigen.eigen_val_i, &Z);

    // Compute eigen vectors of A
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Z.row, Q.colunm, Z.colunm,
                1.0, Z.data, Z.colunm,
                Q.data, Q.colunm,
                0.0, eigen.eigen_vec.data, eigen.eigen_vec.colunm);

    error = ERAM_computeError(&eigen.eigen_vec, h_factor);

    //printf("%ld %lf\n", count_iter, error);
    count_iter++;

    if (fabs(error) > max_error && count_iter < max_iter)
      ERAM_computeNewInputVector(&b, &eigen.eigen_vec);
  }

  vector_free(&b);
  matrix_free(&Q);
  matrix_free(&h);
  matrix_free(&T);
  matrix_free(&Z);

  return eigen;
}