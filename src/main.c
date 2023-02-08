#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "mymath.h"

#define MAX_ITERATIONS 100
#define MIN_ERROR 1e-6

void print_results(eigenData_t eigen) {
    printf("Eigenvalues:\n");
    for (size_t i = 0; i < eigen.eigen_val_r.n; i++) {
        printf("%ld: %f + %fi\n", i, eigen.eigen_val_r.data[i], eigen.eigen_val_i.data[i]);
        printf("\t");
        for (size_t j = 0; j < eigen.eigen_vec.column; j++) {
            printf("%f ", eigen.eigen_vec.data[i * eigen.eigen_vec.column + j]);
        }
        printf("\n");
    }

}

int main(int argc, char **argv) {
    srand(getpid());

    size_t nb_eigen_values;
    matrix_t A;

    if (argc == 4) // exec <taille_sous_espace> --input <fichier_matrice>
    {
        nb_eigen_values = atol(argv[1]);
        A = matrix_readFromFile(argv[3]);
    } else if (argc == 3) // exec <taille_sous_espace> <taille_matrice>
    {
        printf("ici\n");
        nb_eigen_values = atol(argv[1]);
        size_t n = atoi(argv[2]);
        A = matrix_generateRandom(n, n);
    } else {
        fprintf(stderr,
                "Usage: \n\t%s <taille_sous_espace> --input <fichier_matrice>\n\t%s <taille_sous_espace> <taille_matrice>\n",
                argv[0], argv[0]);
        exit(1);
    }



    if (A.data == NULL) {
        printf("Error: matrix is NULL\n");
    }

    printf("Matrix A:\n");
    matrix_print_rowmajor(&A);
    printf("\n");

    eigenData_t eigen = IRAM(&A, nb_eigen_values, MAX_ITERATIONS, MIN_ERROR);
    print_results(eigen);

    matrix_free(&A);
    matrix_free(&eigen.eigen_vec);
    vector_free(&eigen.eigen_val_r);
    vector_free(&eigen.eigen_val_i);

    return 0;
}