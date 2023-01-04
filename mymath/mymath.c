#include "mymath.h"

matrix_t matrix_create(const size_t n_row, const size_t n_column) {
  matrix_t res;

  res.data = malloc(sizeof(double) * n_row * n_column);
  assert(res.data);

  res.row = n_row;
  res.column = n_column;

  matrix_fill(&res, .0);
  return res;
}

vector_t vector_create(const size_t n) {
  vector_t res;

  res.data = malloc(sizeof(double) * n);
  assert(res.data);

  res.n = n;

  vector_fill(&res, .0);
  return res;
}

matrix_t matrix_copy(const matrix_t *matrix) {
  assert(matrix);

  matrix_t res = matrix_create(matrix->row, matrix->column);
  memcpy(res.data, matrix->data, sizeof(double) * matrix->row * matrix->column);

  return res;
}

vector_t vector_copy(const vector_t *vector) {
  assert(vector);

  vector_t res = vector_create(vector->n);
  memcpy(res.data, vector->data, sizeof(double) * vector->n);

  return res;
}

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column) {
  matrix_t res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
    res.data[i] = (double)rand() / (double)RAND_MAX;

  return res;
}

vector_t vector_generateRandom(const size_t n) {
  vector_t res = vector_create(n);

  for (size_t i = 0; i < n; i++)
    res.data[i] = (double)rand() / (double)RAND_MAX;

  return res;
}

void matrix_fill(const matrix_t *matrix, const double value) {
  assert(matrix);

  assert(matrix->data);
  assert(matrix->row != 0);
  assert(matrix->column != 0);

  for (size_t i = 0; i < matrix->row * matrix->column; i++)
    matrix->data[i] = value;
}

void vector_fill(const vector_t *vector, const double value) {
  assert(vector);

  assert(vector->data);
  assert(vector->n != 0);

  for (size_t i = 0; i < vector->n; i++)
    vector->data[i] = value;
}

void matrix_free(matrix_t *matrix) {
  assert(matrix);

  free(matrix->data);
  matrix->data = NULL;
  matrix->row = matrix->column = 0;
}

void vector_free(vector_t *vector) {
  assert(vector);

  free(vector->data);
  vector->data = NULL;
  vector->n = 0;
}

matrix_t matrix_read(FILE *file) {
  matrix_t res;
  size_t n_row, n_column;

  fscanf(file, "%lu %lu", &n_row, &n_column);
  res = matrix_create(n_row, n_column);

  for (size_t i = 0; i < n_row * n_column; i++)
    fscanf(file, "%lf", res.data + i);

  return res;
}

vector_t vector_read(FILE *file) {
  vector_t res;
  unsigned n;
  fscanf(file, "%u", &n);
  res = vector_create(n);

  for (size_t i = 0; i < n; i++)
    fscanf(file, "%lf", res.data + i);

  return res;
}

matrix_t matrix_readFromFile(const char *filename) {
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  matrix_t res = matrix_read(fp);
  fclose(fp);

  return res;
}

vector_t vector_readFromFile(const char *filename) {
  FILE *fp = fopen(filename, "r");
  assert(fp != NULL);

  vector_t vec = vector_read(fp);
  fclose(fp);
  return vec;
}

void matrix_print(const matrix_t *matrix) {
  printf("[%lu, %lu]\n", matrix->row, matrix->column);
  for (size_t i = 0; i < matrix->row; i++) {
    for (size_t j = 0; j < matrix->column; j++) {
      printf("%lf ", matrix->data[i * matrix->column + j]);
    }
    printf("\n");
  }
}

void vector_print(const vector_t *vector) {
  printf("[%lu]\n", vector->n);
  for (size_t i = 0; i < vector->n; i++)
    printf("%lf\n", vector->data[i]);
}

const double vector_dotProduct(const vector_t *x, const vector_t *y) {
  assert(x->n == y->n);

  double res = 0.0;
  for (size_t i = 0; i < x->n; i++)
    res += x->data[i] * y->data[i];

  return res;
}

const double vector_raw_dotProduct(const double *x, const double *y,
                                   const size_t n) {
  assert(x);
  assert(y);

  double res = 0.0;
  for (size_t i = 0; i < n; i++)
    res += x[i] * y[i];

  return res;
}

vector_t matrix_dotProduct(const matrix_t *matrix, const vector_t *vector) {
  assert(matrix);
  assert(vector);
  assert(matrix->row == matrix->column);
  assert(matrix->row == vector->n);

  const size_t size = matrix->row;
  vector_t res = vector_create(size);

  for (size_t i = 0; i < size; i++) {
    res.data[i] = 0.0;
    for (size_t j = 0; j < size; j++)
      res.data[i] += matrix->data[i * size + j] * vector->data[j];
  }

  return res;
}

void matrix_raw_dotProduct(const double *matrix, const double *vector,
                           double *outVector, const size_t m, const size_t n) {
  // matrix (m*n)
  // vector (1*n)

  assert(matrix);
  assert(vector);
  assert(m != 0);
  assert(n != 0);

  const size_t size = m;

  for (size_t i = 0; i < size; i++) {
    outVector[i] = 0.0;
    for (size_t j = 0; j < size; j++)
      outVector[i] += matrix[i * size + j] * vector[j];
  }
}

const double vector_norme(const vector_t *x) {
  double res = 0.0;

  for (size_t i = 0; i < x->n; i++) {
    res += x->data[i] * x->data[i];
  }
  return sqrt(res);
}

const double vector_raw_norme(const double *x, const size_t n) {
  double res = 0.0;

  for (size_t i = 0; i < n; i++) {
    res += x[i] * x[i];
  }
  return sqrt(res);
}

const double matrix_norme(const matrix_t *x) {
  double res = 0.0;

  for (size_t i = 0; i < x->row; i++) {
    for (size_t j = 0; j < x->column; j++) {
      res += x->data[i * x->column + j] * x->data[i * x->column + j];
    }
  }
  return sqrt(res);
}

void arnoldiProjection(size_t start_step, const matrix_t *A, const vector_t *f,
                       const size_t m, const matrix_t *V, const matrix_t *H) {
  assert(A && f && V && H);

  assert(A->row == A->column && A->row == f->n);
  assert(V->row == (m + 1) && V->column == A->row);
  assert(H->row == m && H->column == (m + 1));

  const size_t size = A->row;
  const double kEpsilon = 1e-12;

  const double norme_b = 1.0/vector_norme(f);
  for (size_t i = 0; i < size; i++)
    V->data[i] = f->data[i] * norme_b;

  vector_t buf = vector_create(size);
  for (size_t k = start_step; k < m + 1; k++) {
    matrix_raw_dotProduct(A->data, V->data + ((k - 1) * A->column), buf.data,
                          A->row, A->column);

    for (size_t j = 0; j < k; j++) {
      H->data[(k - 1) * H->column + j] =
          vector_raw_dotProduct(V->data + (j * V->column), buf.data, size);

      for (size_t i = 0; i < buf.n; i++)
        buf.data[i] = buf.data[i] - H->data[(k - 1) * H->column + j] *
                                        V->data[j * V->column + i];
    }
    H->data[(k - 1) * H->column + k] = vector_norme(&buf);

    if (H->data[(k - 1) * H->column + k] > kEpsilon) {
      for (size_t i = 0; i < A->column; i++)
        V->data[k * V->column + i] =
            buf.data[i] / H->data[(k - 1) * H->column + k];
    } else
      return;
  }
}

void ERAM_computeEigenSubspace(const matrix_t *H,
                               const vector_t *eigen_values_r,
                               const vector_t *eigen_values_i,
                               const matrix_t *Z) {
  // T Z eigenValue = computeEigenValue(h)
  LAPACKE_dhseqr(LAPACK_COL_MAJOR, 'S', 'I', H->row, 1, H->row, H->data,
                 H->column, eigen_values_r->data, eigen_values_i->data, Z->data,
                 H->row);

  // eigenVectorSubspace = computeEigenVector(h, Z)
  int m = 0;
  LAPACKE_dtrevc(LAPACK_COL_MAJOR, 'R', 'B', NULL, H->row, H->data, H->column,
                 NULL, 1, Z->data, Z->row, Z->column, &m);
}

const double ERAM_computeError(const matrix_t *eigen_vectors,
                               const double h_factor) {
  double error = 0.0;
  for (size_t i = 0; i < eigen_vectors->row; i++)
    error += fabs(eigen_vectors->data[i * eigen_vectors->column +
                                      (eigen_vectors->column - 1)]);
  error *= h_factor;
  return error;
}

void ERAM_computeNewInputVector(const vector_t *input,
                                const matrix_t *eigen_vectors) {
  for (size_t i = 0; i < input->n; i++)
    input->data[i] = 0.0;

  for (size_t i = 0; i < eigen_vectors->row; i++)
    cblas_daxpy(eigen_vectors->column, 1.0,
                eigen_vectors->data + (i * eigen_vectors->column), 1,
                input->data, 1);

  double norm = vector_raw_norme(input->data, input->n);
  for (size_t i = 0; i < input->n; i++)
    input->data[i] *= norm;
}

eigenData_t IRAM(const matrix_t *A, const size_t n_eigen, const size_t max_iter,
                 const double max_error) {
  // size_t subspace = n_eigen * 2;
  // m = 2 * k = k + p
  // => p = k
  const size_t m = 3 * n_eigen;
  const size_t k = m - n_eigen;

  vector_t f = vector_generateRandom(A->row);

  matrix_t V = matrix_create(m + 1, A->row);
  matrix_t H = matrix_create(m, m + 1);

  matrix_t T = matrix_create(H.row, H.column);
  matrix_t Z = matrix_create(H.row, H.row);

  eigenData_t eigen;
  eigen.eigen_val_r = vector_create(m);
  eigen.eigen_val_i = vector_create(m);
  eigen.eigen_vec = matrix_create(m, A->row);

  double error = DBL_MAX;
  size_t count_iter = 0;

  // Do a complete arnoldi projection
  arnoldiProjection(1, A, &f, m, &V, &H);

  while (1) {
    // Save H(m, m + 1) for later
    const double h_factor = H.data[(H.row * H.column) - 1];

    // Compute eigen vectors of H
    // Do not overwrite A !
    ERAM_computeEigenSubspace(&H, &eigen.eigen_val_r, &eigen.eigen_val_i, &Z);

    // Compute eigen vectors of A
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Z.row, V.column,
                Z.column, 1.0, Z.data, Z.column, V.data, V.column, 0.0,
                eigen.eigen_vec.data, eigen.eigen_vec.column);

    error = ERAM_computeError(&eigen.eigen_vec, h_factor);

    count_iter++;
    if (fabs(error) < max_error || count_iter > max_iter) {
      printf("itération : %ld\nerror : %lf\n", count_iter, fabs(error));
      break;
    }

    double *mu = eigen.eigen_val_r.data + n_eigen;

    // Create Identity matrix
    matrix_t Q = matrix_create(m, m);
    matrix_fill(&Q, 0.0);
    for (size_t i = 0; i < Q.row; i++)
      Q.data[i * Q.column + i] = 1.0;

    for (size_t i = 0; i < k; i++) {
      // Copy H
      // H - ujI

      matrix_t H_copy = matrix_copy(&H);


      for (size_t j = 0; j < H.row; j++)
        H_copy.data[j * H_copy.column + j] -= mu[i];

      matrix_t R = matrix_create(H_copy.row, H_copy.column - 1);
      matrix_print(&H_copy);
      LAPACKE_dgeqrf(LAPACK_COL_MAJOR, H_copy.row, H_copy.column - 1, H_copy.data, H_copy.column, R.data);
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, H_copy.row, H_copy.column - 1, H_copy.column - 1, H_copy.data, H_copy.column, R.data);

      matrix_t Qj;
      Qj.row = H_copy.row;
      Qj.column = H_copy.column;
      Qj.data = H_copy.data;

      // H = Qj(conjugué) * H * Qj;
      printf("ICI 1: \n");
      printf("Qj: \n");
      matrix_print(&Qj);
      printf("H: \n");
      matrix_print(&H);

      matrix_t res = matrix_create(Qj.row, H.column - 1);

      cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, 
                  Qj.row, H.column - 1, Qj.column - 1, 
                  1.0, 
                  Qj.data, Qj.column, 
                  H.data, H.column,   
                  0.0,
                  res.data, res.column);
      printf("Qj * H = : \n");
      matrix_print(&res);

      matrix_t res_final = matrix_create(Qj.row, H.column);
      
      
      printf("ICI 2: \n");
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                  res.row, Qj.column - 1, res.column, 
                  1.0, 
                  res.data, res.column, 
                  Qj.data, Qj.column, 
                  0.0,
                  res_final.data, res_final.column);
      
      H = res_final;

      // Q = Q * Qj
      matrix_print(&Q);
      matrix_print(&Qj);
      printf("ICI 3: \n");
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                  Q.row, Qj.column - 1, Q.column, 
                  1.0, 
                  Q.data, Q.column,
                  Qj.data, Qj.column,
                  0.0, 
                  res.data, res.column);
      Q = res;
      matrix_print(&Q);
    }

    printf("ma petite partie crash pas\n");

    // Update f
    const double kBeta = H.data[(H.row * H.column) - 1];
    const double kSigma = Q.data[(Q.row * Q.column) - 1];
    for (size_t i = 0; i < f.n; i++)
      f.data[i] =
          V.data[i * V.column + V.column - 1] * kBeta + f.data[i] * kSigma;

    // Update V
    printf("ICI 4: \n");
    matrix_print(&V);
    matrix_print(&Q);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                V.row, Q.column, V.column - 1, 
                1.0, 
                V.data, V.column, 
                Q.data, Q.column, 
                0.0,
                V.data, V.column);
    //exit(1);
    // Selection des shifts
    // Decomposition QR k fois
    // Update de V et A, et f (vecteur d'entrée)
    arnoldiProjection(n_eigen, A, &f, m, &V, &H);
    // Restart
  }

  vector_free(&f);
  matrix_free(&V);
  matrix_free(&H);
  matrix_free(&T);
  matrix_free(&Z);

  return eigen;
}