# MIRAM

Dans le cadre de nos études d'informatique de Master 2 CHPS, nous avons dû implémenter l'algorithme Multi Restarted Arnoldi Method (MIRAM) en C++.

## Prérequis

- CMake >= 3.16
- Intel MKL
- MPI


## Compilation

Pour compiler notre projet, nous utilisons le système de build CMake. Pour compiler utiliser la commande suivante:

```bash
  cmake -D_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc ..
  make
```

## Utilisation

L'utilisation du programme se fait grâce à la commande suivante:

```bash
  mpirun -n <nombre de processus> ./bin/main <nombre de valeurs/vecteurs propres> --input <fichier_matrice>
  # ou
  mpirun -n <nombre de processus> ./bin/main <nombre de valeurs/vecteurs propres> <taille_matrice>
```

- \<nombre de processus> : nombre d'instance d'IRAM.
- <nombre de valeurs/vecteurs propres> : nombre de valeurs/vecteurs propres recherch
- <fichier_matrice> : fichier source d'une matrice au format suivant:

Pour une matrice 4x4:
```
4 4
1 2 3 4
5 6 7 8
9 1 2 3
```

#### Autres variables:
Pour modifier le nombre maximal d'itérations: modifier `#define MAX_ITERATIONS` (*main.c*)
Pour modifier l'erreur minimale: modifier `#define MIN_ERROR` (*main.c*)



