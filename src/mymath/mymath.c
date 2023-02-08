#include "mymath.h"

static inline int MAT_GET_CMAJOR(matrix_t m, int r, int c) {
    return (int) c * m.row + r;
}

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

matrix_t matrix_copy(matrix_t *dest, const matrix_t *src) {
    assert(src && dest);
    assert(dest->row == src->row && dest->column == src->column);
    assert(dest->data && src->data);

    memcpy(dest->data, src->data, sizeof(double) * src->row * src->column);

    return *dest;
}

vector_t vector_copy(const vector_t *vector) {
    assert(vector);

    vector_t res = vector_create(vector->n);
    memcpy(res.data, vector->data, sizeof(double) * vector->n);

    return res;
}

matrix_t matrix_generateRandom(const size_t n_row, const size_t n_column) {
    matrix_t res = matrix_create(n_row, n_column);

    for (size_t i = 0; i < n_row * n_column; i++) {
        res.data[i] = (double) rand() / (double) RAND_MAX;
    }

    return res;
}

vector_t vector_generateRandom(const size_t n) {
    vector_t res = vector_create(n);

    for (size_t i = 0; i < n; i++) {
        res.data[i] = (double) rand() / (double) RAND_MAX;
    }

    return res;
}

void matrix_fill(const matrix_t *matrix, const double value) {
    assert(matrix);

    assert(matrix->data);
    assert(matrix->row != 0);
    assert(matrix->column != 0);

    for (size_t i = 0; i < matrix->row * matrix->column; i++) {
        matrix->data[i] = value;
    }
}

void vector_fill(const vector_t *vector, const double value) {
    assert(vector);

    assert(vector->data);
    assert(vector->n != 0);

    for (size_t i = 0; i < vector->n; i++) {
        vector->data[i] = value;
    }
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

    for (size_t i = 0; i < n_row * n_column; i++) {
        fscanf(file, "%lf", res.data + i);
    }

    return res;
}

vector_t vector_read(FILE *file) {
    vector_t res;
    unsigned n;
    fscanf(file, "%u", &n);
    res = vector_create(n);

    for (size_t i = 0; i < n; i++) {
        fscanf(file, "%lf", res.data + i);
    }

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


void matrix_print_rowmajor(const matrix_t *restrict matrix) {
    printf("Matrix [%lu, %lu]\n", matrix->row, matrix->column);
    if (matrix->row > 20 || matrix->column > 20) {
        printf("[\n\t...\n]\n");
        return;
    }

    for (size_t i = 0; i < matrix->row; i++) {
        for (size_t j = 0; j < matrix->column; j++)
            printf("%lf ", matrix->data[i * matrix->column + j]);

        printf("\n");
    }
}

void matrix_print_colmajor(const matrix_t *restrict matrix) {
    printf("Matrix [%lu, %lu]\n", matrix->row, matrix->column);
    if (matrix->row > 20 || matrix->column > 20) {
        printf("[\n\t...\n]\n");
        return;
    }

    for (size_t i = 0; i < matrix->row; i++) {
        for (size_t j = 0; j < matrix->column; j++)
            printf("%lf ", matrix->data[MAT_GET_CMAJOR(*matrix, i, j)]);
        printf("\n");
    }
}

void vector_print(const vector_t *vector) {
    printf("[%lu]\n", vector->n);
    for (size_t i = 0; i < vector->n; i++) {
        printf("%lf\n", vector->data[i]);
    }
}

double vector_dotProduct(const vector_t *x, const vector_t *y) {
    assert(x->n == y->n);

    double res = 0.0;
    for (size_t i = 0; i < x->n; i++) {
        res += x->data[i] * y->data[i];
    }

    return res;
}

double vector_raw_dotProduct(const double *x, const double *y, const size_t n) {
    assert(x);
    assert(y);

    double res = 0.0;
    for (size_t i = 0; i < n; i++) {
        res += x[i] * y[i];
    }

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
        for (size_t j = 0; j < size; j++) {
            res.data[i] += matrix->data[i * size + j] * vector->data[j];
        }
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
        for (size_t j = 0; j < size; j++) {
            outVector[i] += matrix[i * size + j] * vector[j];
        }
    }
}

double vector_norme(const vector_t *x) {
    double res = 0.0;

    for (size_t i = 0; i < x->n; i++) {
        res += x->data[i] * x->data[i];
    }
    return sqrt(res);
}

double vector_raw_norme(const double *x, const size_t n) {
    double res = 0.0;

    for (size_t i = 0; i < n; i++) {
        res += x[i] * x[i];
    }
    return sqrt(res);
}

double matrix_norme(const matrix_t *x) {
    double res = 0.0;

    for (size_t i = 0; i < x->row; i++) {
        for (size_t j = 0; j < x->column; j++) {
            res += x->data[i * x->column + j] * x->data[i * x->column + j];
        }
    }
    return sqrt(res);
}

void eigen_sort(const eigenData_t *eigen) {
    const size_t size = eigen->eigen_val_r.n;

    vector_t buffer = vector_create(size);

    for (size_t j = 0; j < size; j++) {
        double max = eigen->eigen_val_r.data[j] * eigen->eigen_val_r.data[j] +
                     eigen->eigen_val_i.data[j] * eigen->eigen_val_i.data[j];
        size_t index = j;
        for (size_t i = j; i < size; i++) {
            double tmp = eigen->eigen_val_r.data[i] * eigen->eigen_val_r.data[i] +
                         eigen->eigen_val_i.data[i] * eigen->eigen_val_i.data[i];
            if (max < tmp) {
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

        memcpy(buffer.data, eigen->eigen_vec.data + (size * j),
               size * sizeof(double));
        memcpy(eigen->eigen_vec.data + (size * j),
               eigen->eigen_vec.data + (size * index), size * sizeof(double));
        memcpy(eigen->eigen_vec.data + (size * index),
               eigen->eigen_vec.data + (size * j), size * sizeof(double));
    }
    vector_free(&buffer);
}

void arnoldiProjection(size_t start_step, const matrix_t *A, const vector_t *f,
                       const size_t m, const matrix_t *V, const matrix_t *H) {
    assert(A && f && V && H);

    assert(A->row == A->column && A->row == f->n);
    assert(V->row == A->row && V->column == (m + 1));
    assert(H->row == (m + 1) && H->column == m);

    const size_t size = A->row;
    const double kEpsilon = 1e-12;

    if (start_step == 1) {
        const double norme_b = 1.0 / vector_norme(f);
        for (size_t i = 0; i < size; i++) {
            V->data[i] = f->data[i] * norme_b;
        }
    }

    vector_t buf = vector_create(size);

    for (size_t k = start_step; k < m + 1; k++) {
        matrix_raw_dotProduct(A->data, V->data + ((k - 1) * V->row), buf.data,
                              A->row, A->column);

        for (size_t j = 0; j < k; j++) {
            H->data[MAT_GET_CMAJOR(*H, j, (k - 1))] =
                    vector_raw_dotProduct(V->data + (j * V->row), buf.data, size);

            for (size_t i = 0; i < buf.n; i++) {
                buf.data[i] = buf.data[i] - H->data[MAT_GET_CMAJOR(*H, j, (k - 1))] *
                                            V->data[MAT_GET_CMAJOR(*V, i, j)];
            }
        }
        H->data[MAT_GET_CMAJOR(*H, k, (k - 1))] = vector_norme(&buf);

        if (H->data[MAT_GET_CMAJOR(*H, k, (k - 1))] > kEpsilon) {
            for (size_t i = 0; i < A->column; i++) {
                V->data[MAT_GET_CMAJOR(*V, i, k)] =
                        buf.data[i] / H->data[MAT_GET_CMAJOR(*H, k, (k - 1))];
            }
        } else {
            printf("Arnoldi: breaking because norme < tol, norme = %lf, k = %i\n",
                   H->data[MAT_GET_CMAJOR(*H, k, (k - 1))], k);
            break;
        }
    }
    vector_free(&buf);
}

void ERAM_computeEigenSubspace(const matrix_t *H,
                               const vector_t *eigen_values_r,
                               const vector_t *eigen_values_i,
                               const matrix_t *Z) {
    // T Z eigenValue = computeEigenValue(h)
    matrix_print_colmajor(H);
    LAPACKE_dhseqr(LAPACK_COL_MAJOR, 'S', 'I', H->column, 1, H->column, H->data,
                   H->row, eigen_values_r->data, eigen_values_i->data, Z->data,
                   Z->row);

    // eigenVectorSubspace = computeEigenVector(h, Z)
    int m = 0;
    LAPACKE_dtrevc(LAPACK_COL_MAJOR, 'R', 'B', NULL, H->row - 1, H->data, H->row,
                   NULL, 1, Z->data, Z->row, Z->column, &m);
}

double ERAM_computeError(size_t k, const matrix_t *eigen_vectors,
                         const double h_factor) {
    double error = 0.0;
    for (size_t i = 0; i < k; i++) {
        error += fabs(eigen_vectors->data[MAT_GET_CMAJOR(
                *eigen_vectors, (eigen_vectors->column - 1), i)]);
    }
    error *= h_factor;
    return fabs(error);
}

void ERAM_computeNewInputVector(const vector_t *input,
                                const matrix_t *eigen_vectors) {
    for (size_t i = 0; i < input->n; i++) {
        input->data[i] = 0.0;
    }

    for (size_t i = 0; i < eigen_vectors->row; i++) {
        cblas_daxpy(eigen_vectors->column, 1.0,
                    eigen_vectors->data + (i * eigen_vectors->column), 1,
                    input->data, 1);
    }

    double norm = vector_raw_norme(input->data, input->n);
    for (size_t i = 0; i < input->n; i++) {
        input->data[i] *= norm;
    }
}

eigenData_t IRAM(const matrix_t *A, const size_t n_eigen, const size_t max_iter,
                 const double max_error) {

    // Number of wanted eigen values
    const size_t k = n_eigen;
    // Subspace size
    const size_t m = 3 * k;
    // Supplementary dimensions
    // (Difference between wanted n eigen values and subspace size)
    const size_t p = m - k;

    // Allocate all matrices to avoid memory allocation during the algorithm
    // Create a random starting vector
    vector_t f = vector_generateRandom(A->row);

    // Projection matrix
    matrix_t V = matrix_create(A->row, m + 1);
    // Hessenberg matrix (Projection of A in the subspace)
    matrix_t H = matrix_create(m + 1, m);
    // We need a copy of H later on
    matrix_t H_copy = matrix_create(m + 1, m);

    matrix_t T = matrix_create(H.row, H.column);
    matrix_t Z = matrix_create(m, m);

    // QR decomposition matrices
    matrix_t Q = matrix_create(m, m);
    double *tau = (double *) malloc(m * sizeof(double));

    // Buffers for QR decomposition and dgemm
    matrix_t res = matrix_create(m, m);
    matrix_t res_final = matrix_create(m + 1, m);

    // Final eigenvalues/vectors returned by the algorithm
    eigenData_t eigen;
    eigen.eigen_val_r = vector_create(m);
    eigen.eigen_val_i = vector_create(m);
    eigen.eigen_vec = matrix_create(A->row, m);

    double residual = DBL_MAX;
    size_t count_iter = 0;

    // Bootstrap the algorithm with a full Arnoldi method
    arnoldiProjection(1, A, &f, m, &V, &H);

    while (1) {
        count_iter++;

        // Save H(m, m + 1) for later
        const double h_factor = H.data[MAT_GET_CMAJOR(H, H.row - 1, H.column - 1)];

        // Compute the eigenvalues/eigenvectors in the subspace, using H
        ERAM_computeEigenSubspace(&H, &eigen.eigen_val_r, &eigen.eigen_val_i, &Z);

        // Retro-projection of the eigenvectors in the original space by multiplying
        // Z and V
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Z.row, V.column - 1,
                    Z.column, 1.0, Z.data, Z.column, V.data, V.row, 0.0,
                    eigen.eigen_vec.data, eigen.eigen_vec.column);

        eigen_sort(&eigen);

        // Compute the residual error of the K first eigenvalues
        residual = ERAM_computeError(k, &eigen.eigen_vec, h_factor);

        if (residual < max_error || count_iter > max_iter) {
            matrix_print_colmajor(&H);
            printf("H factor when breaking: %lf\n", h_factor);
            break;
        }

        double *mu = eigen.eigen_val_r.data + k;

        // Q is the identity matrix
        matrix_fill(&Q, 0.0);
        for (size_t i = 0; i < Q.row; i++) {
            Q.data[MAT_GET_CMAJOR(Q, i, i)] = 1.0;
        }

        // Perform an implicitly shifted QR decomposition using the unwanted
        // eigenvalues as shifts
        for (size_t i = 0; i < p; i++) {

            // Copy H
            matrix_copy(&H_copy, &H);

            // H - ujI
            for (size_t j = 0; j < H.column; j++) {
                H_copy.data[MAT_GET_CMAJOR(H_copy, j, j)] -= mu[i];
            }

            // QR decomposition
            // Here, tau contains elementary reflectors
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, H_copy.row - 1, H_copy.column,
                           H_copy.data, H_copy.row, tau);
            // Compute Qj from elementary reflectors and store it in H_copy
            LAPACKE_dorgqr(LAPACK_COL_MAJOR, H_copy.row - 1, H_copy.column,
                           H_copy.column, H_copy.data, H_copy.row, tau);
            // From here on, H_copy contains Qj

            // Make an alias for clarity
            matrix_t Qj;
            Qj.row = H_copy.row;
            Qj.column = H_copy.column;
            Qj.data = H_copy.data;

            // Compute Qj* x H
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Qj.row - 1, H.column,
                        Qj.column, 1.0, Qj.data, Qj.row, H.data, H.row, 0.0, res.data,
                        res.row);

            //  Compute (Qj* x H) x Qj
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, res.row, Qj.column,
                        res.column, 1.0, res.data, res.column, Qj.data, Qj.row, 0.0,
                        res_final.data, res_final.row);
            // Swap H and res_final
            matrix_t buf = H;
            H = res_final;
            res_final = buf;

            // Compute Q = Q x Qj
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Q.row, Qj.column,
                        Q.row, 1.0, Q.data, Q.row, Qj.data, Qj.row, 0.0, res.data,
                        res.row);
            // Swap Q and res
            buf = Q;
            Q = res;
            res = buf;
        }

        // Update f
        const double kBeta = H.data[MAT_GET_CMAJOR(H, k + 1, k)];
        const double kSigma = Q.data[MAT_GET_CMAJOR(Q, m, k)];
        for (size_t i = 0; i < f.n; i++) {
            f.data[i] = V.data[MAT_GET_CMAJOR(V, V.column - 1, i)] * kBeta +
                        f.data[i] * kSigma;
        }

        // Update V
        matrix_t new_v = matrix_create(V.row, V.column);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, V.row, k,
                    V.column - 1, 1.0, V.data, V.row, Q.data, Q.row, 0.0,
                    new_v.data, new_v.row);
        matrix_free(&V);
        V = new_v;

        // Update H
        for (size_t y = k; y < H.row; y++) {
            for (size_t x = k; x < H.column; x++) {
                H.data[MAT_GET_CMAJOR(H, x, y)] = .0;
            }
        }
        //  Selection des shifts
        //  Decomposition QR k fois
        //  Update de V et A, et f (vecteur d'entrée)
        arnoldiProjection(k, A, &f, m, &V, &H);
        // Restart
    }

    vector_free(&f);
    matrix_free(&V);
    matrix_free(&H);
    matrix_free(&T);
    matrix_free(&Z);

    matrix_free(&H_copy);
    matrix_free(&Q);
    free(tau);
    matrix_free(&res);
    matrix_free(&res_final);

    printf("Done !\n");
    printf("H: \n");
    matrix_print_colmajor(&H);
    printf("Itération : %ld / max_iter: %ld\nerror : %lf / max error: %lf\n",
           count_iter, max_iter, fabs(residual), max_error);

    return eigen;
}
