#include "mymath.h"

static inline int MAT_GET_CMAJOR(matrix_t m, int r, int c)
{
  return (int)c * m.row + r;
}

matrix_t matrix_create(const size_t n_row, const size_t n_column)
{
  matrix_t res;

  res.data = malloc(sizeof(double) * n_row * n_column);
  assert(res.data);

  res.row = n_row;
  res.column = n_column;

  matrix_fill(&res, .0);
  return res;
}

vector_t vector_create(const size_t n)
{
  vector_t res;

  res.data = malloc(sizeof(double) * n);
  assert(res.data);

  res.n = n;

  vector_fill(&res, .0);
  return res;
}

matrix_t matrix_copy(matrix_t *restrict dest, const matrix_t *restrict src)
{
  assert(src && dest);
  assert(dest->row == src->row && dest->column == src->column);
  assert(dest->data && src->data);

  memcpy(dest->data, src->data, sizeof(double) * src->row * src->column);

  return *dest;
}

vector_t vector_copy(const vector_t *restrict vector)
{
  assert(vector);

  vector_t res = vector_create(vector->n);
  memcpy(res.data, vector->data, sizeof(double) * vector->n);

  return res;
}

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column)
{
  matrix_t res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
  {
    res.data[i] = (double)rand() / (double)RAND_MAX;
  }

  return res;
}

vector_t vector_generateRandom(const size_t n)
{
  vector_t res = vector_create(n);

  for (size_t i = 0; i < n; i++)
  {
    res.data[i] = (double)rand() / (double)RAND_MAX;
  }

  return res;
}

void matrix_fill(const matrix_t *restrict matrix, const double value)
{
  assert(matrix);

  assert(matrix->data);
  assert(matrix->row != 0);
  assert(matrix->column != 0);

  for (size_t i = 0; i < matrix->row * matrix->column; i++)
  {
    matrix->data[i] = value;
  }
}

void vector_fill(const vector_t *restrict vector, const double value)
{
  assert(vector);

  assert(vector->data);
  assert(vector->n != 0);

  for (size_t i = 0; i < vector->n; i++)
  {
    vector->data[i] = value;
  }
}

void matrix_free(matrix_t *restrict matrix)
{
  assert(matrix);

  free(matrix->data);
  matrix->data = NULL;
  matrix->row = matrix->column = 0;
}

void vector_free(vector_t *restrict vector)
{
  assert(vector);

  free(vector->data);
  vector->data = NULL;
  vector->n = 0;
}

matrix_t matrix_read(FILE *restrict file)
{
  matrix_t res;
  size_t n_row, n_column;

  fscanf(file, "%lu %lu", &n_row, &n_column);
  res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
  {
    fscanf(file, "%lf", res.data + i);
  }

  return res;
}

vector_t vector_read(FILE *restrict file)
{
  vector_t res;
  unsigned n;
  fscanf(file, "%u", &n);
  res = vector_create(n);

  for (size_t i = 0; i < n; i++)
  {
    fscanf(file, "%lf", res.data + i);
  }

  return res;
}

matrix_t matrix_readFromFile(const char *restrict filename)
{
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  matrix_t res = matrix_read(fp);
  fclose(fp);

  return res;
}

vector_t vector_readFromFile(const char *restrict filename)
{
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  vector_t vec = vector_read(fp);
  fclose(fp);
  return vec;
}

void matrix_print(const matrix_t *restrict matrix)
{
  printf("[%lu, %lu]\n", matrix->row, matrix->column);
  for (size_t i = 0; i < matrix->row; i++)
  {
    for (size_t j = 0; j < matrix->column; j++)
    {
      printf("%lf ", matrix->data[i * matrix->column + j]);
    }
    printf("\n");
  }
}

void matrix_print_colmajor(const matrix_t *restrict matrix)
{
  printf("[%lu, %lu]\n", matrix->row, matrix->column);
  for (size_t i = 0; i < matrix->row; i++)
  {
    for (size_t j = 0; j < matrix->column; j++)
      printf("%lf ", matrix->data[MAT_GET_CMAJOR(*matrix, i, j)]);
    printf("\n");
  }
}

void vector_print(const vector_t *restrict vector)
{
  printf("[%lu]\n", vector->n);
  for (size_t i = 0; i < vector->n; i++)
  {
    printf("%lf\n", vector->data[i]);
  }
}

void eigen_sort(const eigenData_t *restrict eigen, const vector_t *restrict buffer)
{
  const size_t size = eigen->eigen_val_r.n;

  for (size_t j = 0; j < size; j++)
  {
    double max = eigen->eigen_val_r.data[j] * eigen->eigen_val_r.data[j] +
                 eigen->eigen_val_i.data[j] * eigen->eigen_val_i.data[j];
    size_t index = j;
    for (size_t i = j; i < size; i++)
    {
      double tmp = eigen->eigen_val_r.data[i] * eigen->eigen_val_r.data[i] +
                   eigen->eigen_val_i.data[i] * eigen->eigen_val_i.data[i];
      if (max < tmp)
      {
        max = tmp;
        index = i;
      }
    }

    double tmp = eigen->eigen_val_r.data[j];
    eigen->eigen_val_r.data[j] = eigen->eigen_val_r.data[index];
    eigen->eigen_val_r.data[index] = tmp;

    tmp = eigen->eigen_val_i.data[j];
    eigen->eigen_val_i.data[j] = eigen->eigen_val_i.data[index];
    eigen->eigen_val_i.data[index] = tmp;

    memcpy(buffer->data, eigen->eigen_vec.data + (size * j),
           size * sizeof(double));
    memcpy(eigen->eigen_vec.data + (size * j),
           eigen->eigen_vec.data + (size * index), size * sizeof(double));
    memcpy(eigen->eigen_vec.data + (size * index),
           eigen->eigen_vec.data + (size * j), size * sizeof(double));
  }
}

void swap(const size_t index1, const size_t index2, const eigenData_t *restrict eigen, const vector_t *restrict buffer)
{
  const size_t size = eigen->eigen_val_r.n;

  double tmp = eigen->eigen_val_r.data[index1];
  eigen->eigen_val_r.data[index1] = eigen->eigen_val_r.data[index2];
  eigen->eigen_val_r.data[index2] = tmp;

  tmp = eigen->eigen_val_i.data[index1];
  eigen->eigen_val_i.data[index1] = eigen->eigen_val_i.data[index2];
  eigen->eigen_val_i.data[index2] = tmp;

  memcpy(buffer->data, eigen->eigen_vec.data + (size * index1), size * sizeof(double));
  memcpy(eigen->eigen_vec.data + (size * index1), eigen->eigen_vec.data + (size * index2), size * sizeof(double));
  memcpy(eigen->eigen_vec.data + (size * index2), buffer->data, size * sizeof(double));
}

size_t partition(const eigenData_t *restrict eigen, const vector_t *restrict buffer, const size_t low, const size_t high)
{
  const double pivot = eigen->eigen_val_r.data[high];
  size_t i = low;

  for (size_t j = low; j <= high - 1; j++)
  {
    if (eigen->eigen_val_r.data[j] >= pivot)
    {
      swap(i, j, eigen, buffer);
      i++;
    }
  }
  swap(i, high, eigen, buffer);
  return i;
}

void eigenQuickSort(const eigenData_t *restrict eigen, const vector_t *restrict buffer, const size_t low, const size_t high)
{
  if (low < high)
  {
    size_t pi = partition(eigen, buffer, low, high);
    eigenQuickSort(eigen, buffer, low, pi - 1);
    eigenQuickSort(eigen, buffer, pi + 1, high);
  }
}

typedef struct bufferIRAM_s
{
  vector_t f;

  // Projection matrix
  matrix_t V;
  // Hessenberg matrix (Projection of A in the subspace)
  matrix_t H;
  // We need a copy of H later on
  matrix_t H_copy;

  matrix_t T;
  matrix_t Z;

  // QR decomposition matrices
  matrix_t Q;
  vector_t tau;

  // Buffers for QR decomposition and dgemm
  matrix_t res;
  matrix_t res_final;

  // Buffer for eigen sort
  vector_t buffer_eigen_sort;

  // Buffer for arnoldi projection
  vector_t buffer_arnoldi;
} bufferIRAM_t;

bufferIRAM_t bufferIRAM_init(const size_t inputSizeA, const size_t subspaceSize)
{
  bufferIRAM_t res;
  res.f = vector_generateRandom(inputSizeA);

  // Projection matrix
  res.V = matrix_create(inputSizeA, subspaceSize + 1);
  // Hessenberg matrix (Projection of A in the subspace)
  res.H = matrix_create(subspaceSize + 1, subspaceSize);
  // We need a copy of H later on
  res.H_copy = matrix_create(subspaceSize + 1, subspaceSize);

  res.T = matrix_create(res.H.row, res.H.column);
  res.Z = matrix_create(subspaceSize, subspaceSize);

  // QR decomposition matrices
  res.Q = matrix_create(subspaceSize, subspaceSize);
  res.tau = vector_create(subspaceSize);

  // Buffers for QR decomposition and dgemm
  res.res = matrix_create(subspaceSize, subspaceSize);
  res.res_final = matrix_create(subspaceSize + 1, subspaceSize);

  // Buffer for eigen sort
  res.buffer_eigen_sort = vector_create(subspaceSize);

  // Buffer for arnoldi projection
  res.buffer_arnoldi = vector_create(inputSizeA);

  return res;
}

void bufferIRAM_free(bufferIRAM_t *restrict buffer)
{
  vector_free(&buffer->f);
  matrix_free(&buffer->V);
  matrix_free(&buffer->H);
  matrix_free(&buffer->H_copy);
  matrix_free(&buffer->T);
  matrix_free(&buffer->Z);
  matrix_free(&buffer->Q);
  vector_free(&buffer->tau);
  matrix_free(&buffer->res);
  matrix_free(&buffer->res_final);
  vector_free(&buffer->buffer_eigen_sort);
  vector_free(&buffer->buffer_arnoldi);
}

void arnoldiProjection(size_t start_step, const matrix_t *restrict A, const vector_t *restrict f,
                       const size_t m, const matrix_t *restrict V, const matrix_t *restrict H, const vector_t *restrict buffer)
{
  assert(A && f && V && H);

  assert(A->row == A->column && A->row == f->n);
  assert(V->row == A->row && V->column == (m + 1));
  assert(H->row == (m + 1) && H->column == m);

  const size_t size = A->row;
  const double kEpsilon = 1e-12;

  if (start_step == 1)
  {
    const double norme_b = 1.0 / cblas_dnrm2(f->n, f->data, 1);
    for (size_t i = 0; i < size; i++)
    {
      V->data[i] = f->data[i] * norme_b;
    }
  }

  for (size_t k = start_step; k < m + 1; k++)
  {
    cblas_dgemv(CblasColMajor, CblasNoTrans, A->row, A->column,
                1.0, A->data, A->row, V->data + ((k - 1) * V->row),
                1, 0.0, buffer->data, 1);

    for (size_t j = 0; j < k; j++)
    {
      H->data[MAT_GET_CMAJOR(*H, j, (k - 1))] = cblas_ddot(size, V->data + (j * V->row), 1, buffer->data, 1);

      for (size_t i = 0; i < buffer->n; i++)
      {
        buffer->data[i] = buffer->data[i] - H->data[MAT_GET_CMAJOR(*H, j, (k - 1))] *
                                                V->data[MAT_GET_CMAJOR(*V, i, j)];
      }
    }
    H->data[MAT_GET_CMAJOR(*H, k, (k - 1))] = cblas_dnrm2(buffer->n, buffer->data, 1);

    if (H->data[MAT_GET_CMAJOR(*H, k, (k - 1))] > kEpsilon)
    {
      const double tmp = 1.0 / H->data[MAT_GET_CMAJOR(*H, k, (k - 1))];
      for (size_t i = 0; i < A->column; i++)
      {
        V->data[MAT_GET_CMAJOR(*V, i, k)] = buffer->data[i] * tmp;
      }
    }
    else
    {
      printf("Arnoldi: breaking because norme < tol, norme = %lf, k = %lu\n",
             H->data[MAT_GET_CMAJOR(*H, k, (k - 1))], k);
      break;
    }
  }
}

void IRAM_computeEigenSubspace(const matrix_t *restrict H,
                               const vector_t *restrict eigen_values_r,
                               const vector_t *restrict eigen_values_i,
                               const matrix_t *restrict Z)
{
  // T Z eigenValue = computeEigenValue(h)
  LAPACKE_dhseqr(LAPACK_COL_MAJOR, 'S', 'I', H->column, 1, H->column, H->data,
                 H->row, eigen_values_r->data, eigen_values_i->data, Z->data,
                 Z->row);

  // eigenVectorSubspace = computeEigenVector(h, Z)
  int m = 0;
  LAPACKE_dtrevc(LAPACK_COL_MAJOR, 'R', 'B', NULL, H->row - 1, H->data, H->row,
                 NULL, 1, Z->data, Z->row, Z->column, &m);
}

double IRAM_computeError(size_t k, const matrix_t *restrict eigen_vectors,
                         const double h_factor)
{
  double error = 0.0;
  for (size_t i = 0; i < k; i++)
  {
    error += fabs(eigen_vectors->data[MAT_GET_CMAJOR(
        *eigen_vectors, (eigen_vectors->column - 1), i)]);
  }
  error *= h_factor;
  return fabs(error);
}


eigenData_t IRAM(const matrix_t *A, const size_t n_eigen, const size_t max_iter,
                 const double max_error)
{

  // Number of wanted eigen values
  const size_t k = n_eigen;
  // Subspace size
  const size_t m = 3 * k;
  // Supplementary dimensions
  // (Difference between wanted n eigen values and subspace size)
  const size_t p = m - k;

  // Allocate buffer for IRAM
  bufferIRAM_t buffer = bufferIRAM_init(A->row, m);

  // Final eigenvalues/vectors returned by the algorithm
  eigenData_t eigen;
  eigen.eigen_val_r = vector_create(m);
  eigen.eigen_val_i = vector_create(m);
  eigen.eigen_vec = matrix_create(A->row, m);

  double residual = DBL_MAX;
  size_t count_iter = 0;

  // Bootstrap the algorithm with a full Arnoldi method
  arnoldiProjection(1, A, &buffer.f, m, &buffer.V, &buffer.H, &buffer.buffer_arnoldi);

  while (1)
  {
    count_iter++;

    // Save H(m, m + 1) for later
    const double h_factor = buffer.H.data[MAT_GET_CMAJOR(buffer.H, buffer.H.row - 1, buffer.H.column - 1)];

    // Compute the eigenvalues/eigenvectors in the subspace, using H
    matrix_copy(&buffer.H_copy, &buffer.H);
    IRAM_computeEigenSubspace(&buffer.H_copy, &eigen.eigen_val_r, &eigen.eigen_val_i, &buffer.Z);

    // Retro-projection of the eigenvectors in the original space by multiplying
    // Z and V
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, buffer.Z.row, buffer.V.column - 1,
                buffer.Z.column, 1.0, buffer.Z.data, buffer.Z.column, buffer.V.data, buffer.V.row, 0.0,
                eigen.eigen_vec.data, eigen.eigen_vec.column);

    // eigen_sort(&eigen, &buffer.buffer_eigen_sort);
    eigen_sort(&eigen, &buffer.buffer_eigen_sort);

    // Compute the residual error of the K first eigenvalues
    residual = IRAM_computeError(k, &eigen.eigen_vec, h_factor);

    if (residual < max_error || count_iter > max_iter)
    {
      printf("H factor when breaking: %lf\n", h_factor);
      break;
    }

    double *mu = eigen.eigen_val_r.data + k;

    // Q is the identity matrix
    matrix_fill(&buffer.Q, 0.0);
    for (size_t i = 0; i < buffer.Q.row; i++)
    {
      buffer.Q.data[MAT_GET_CMAJOR(buffer.Q, i, i)] = 1.0;
    }

    // Perform an implicitly shifted QR decomposition using the unwanted
    // eigenvalues as shifts
    for (size_t i = 0; i < p; i++)
    {

      // Copy H
      matrix_copy(&buffer.H_copy, &buffer.H);

      // H - ujI
      for (size_t j = 0; j < buffer.H.column; j++)
      {
        buffer.H_copy.data[MAT_GET_CMAJOR(buffer.H_copy, j, j)] -= mu[i];
      }

      // QR decomposition
      // Here, tau contains elementary reflectors

      LAPACKE_dgeqrf(LAPACK_COL_MAJOR, buffer.H_copy.row - 1, buffer.H_copy.column,
                     buffer.H_copy.data, buffer.H_copy.row, buffer.tau.data);
      // Compute Qj from elementary reflectors and store it in H_copy
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, buffer.H_copy.row - 1, buffer.H_copy.column,
                     buffer.H_copy.column, buffer.H_copy.data, buffer.H_copy.row, buffer.tau.data);

      // From here on, H_copy contains Qj

      // Make an alias for clarity
      matrix_t Qj;
      Qj.row = buffer.H_copy.row;
      Qj.column = buffer.H_copy.column;
      Qj.data = buffer.H_copy.data;

      // Compute Qj* x H
      cblas_dgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, Qj.row - 1, buffer.H.column,
                  Qj.column, 1.0, Qj.data, Qj.row, buffer.H.data, buffer.H.row, 0.0, buffer.res.data,
                  buffer.res.row);

      //  Compute (Qj* x H) x Qj
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, buffer.res.row, Qj.column,
                  buffer.res.column, 1.0, buffer.res.data, buffer.res.column, Qj.data, Qj.row, 0.0,
                  buffer.res_final.data, buffer.res_final.row);
      // Swap H and res_final
      matrix_t buf = buffer.H;
      buffer.H = buffer.res_final;
      buffer.res_final = buf;

      // Compute Q = Q x Qj
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, buffer.Q.row, Qj.column,
                  buffer.Q.row, 1.0, buffer.Q.data, buffer.Q.row, Qj.data, Qj.row, 0.0, buffer.res.data,
                  buffer.res.row);
      // Swap Q and res
      buf = buffer.Q;
      buffer.Q = buffer.res;
      buffer.res = buf;
    }


    // Update f
    const double kBeta = buffer.H.data[MAT_GET_CMAJOR(buffer.H, k + 1, k)];
    const double kSigma = buffer.Q.data[MAT_GET_CMAJOR(buffer.Q, m, k)];
    for (size_t i = 0; i < buffer.f.n; i++)
    {
      buffer.f.data[i] = buffer.V.data[MAT_GET_CMAJOR(buffer.V, i, buffer.V.column - 1)] * kBeta +
                  buffer.f.data[i] * kSigma;
    }

    matrix_t new_v = matrix_create(buffer.V.row, buffer.V.column);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, buffer.V.row, k,
                buffer.V.column - 1, 1.0, buffer.V.data, buffer.V.row, buffer.Q.data, buffer.Q.row, 0.0,
                new_v.data, new_v.row);
    matrix_free(&buffer.V);
    buffer.V = new_v;

    // Update H
    for (size_t y = k; y < buffer.H.row; y++)
    {
      for (size_t x = k; x < buffer.H.column; x++)
      {
        buffer.H.data[MAT_GET_CMAJOR(buffer.H, y, x)] = .0;
      }
    }
    //  Selection des shifts
    //  Decomposition QR k fois
    //  Update de V et A, et f (vecteur d'entrée)
    arnoldiProjection(k, A, &buffer.f, m, &buffer.V, &buffer.H, &buffer.buffer_arnoldi);
    // Restart
  }

  bufferIRAM_free(&buffer);

  printf("itération : %ld / max_iter: %ld\nerror : %lf / max error: %lf\n",
         count_iter, max_iter, fabs(residual), max_error);

  return eigen;
}
