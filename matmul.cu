/*
* Rayhana ZIARA
* produit matrice matrice
*/

#include <stdlib.h>
#include <stdio.h>

/*
* DESCRIPTION : kernel concernant le produit matrice matrice
* PARAMETRES : matrice A, nb ligne de A, nb colonne de A, matrice B, nb ligne de B, nb colonne de B, matrice C, nb ligne de C et nb colonne de C
* RETOUR : /
*/
__global__ void matMul(float *A, int l_A, int c_A, float *B, int l_B, int c_B, float *C, int l_C, int c_C) 
{
  float resultat = 0.0;
  int ligne = blockDim.x * blockIdx.x + threadIdx.x;
  int colonne = blockDim.y * blockIdx.y + threadIdx.y;

  if(ligne > l_A || colonne > c_B)
  {
    printf("ERREUR - Soit ligne > m soit colonne > m\n");
    return;
  }

  for(int i = 0; i < c_A; i++)
    resultat += A[ligne * c_A + i] * B[i * c_B + colonne];

  C[ligne * c_C + colonne] = resultat;
}

/*
* DESCRIPTION : fonction d'affichage de matrice et de vecteur
* PARAMETRES : matrice à afficher, nb ligne et nb colonne de A, 
* RETOUR : /
*/
void affichage(float *M, int ligne, int colonne)
{
  for(int i = 0; i < ligne; i++)
  {
    for(int j = 0; j < colonne; j++)
      fprintf(stdout, "%lf\t", M[i * ligne + j]);
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");
}

int main(int argc, char **argv)
{
  // declaration des variables du produit matrice matrice
  // variables du hote
  float *A, *B, *C;
  int n, m; // taille des matrices A et B

  // variables du device
  float *d_A, *d_B, *d_C;

  if(argc != 3)
  {
    fprintf(stderr, "ERREUR - Veuillez entrez la taille de A et la taille de B en parametre d'execution. Merci'\n./exam_rz n m\n");
    return -1;
  }

  n = atoi(argv[1]); // taille de la matrice A(n * n)
  m = atoi(argv[2]); // taille de la matrice B(m * m)

  // allocation memoire dans le hote pour les matrices A, B et C
  A = (float*)malloc(m * n * sizeof(float));
  B = (float*)malloc(n * m * sizeof(float));
  C = (float*)malloc(m * m * sizeof(float));

  // initialisation des matrices (matrice stockée en 1D)
  for(int i = 0; i < m; i++)
  {
    for(int j = 0; j < n; j++)
      A[i * m + j] = i * m + j;
  }

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
      B[i * n + j] = i * n + j;
  }

  // allocation memoire dans le device pour les equivalents des matrices A, B et C
  cudaMalloc(&d_A, m * n * sizeof(float));
  cudaMalloc(&d_B, n * m * sizeof(float));
  cudaMalloc(&d_C, m * m * sizeof(float));

  // copie des matrices A et B dans le device
  cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n * m * sizeof(float), cudaMemcpyHostToDevice);

  // appel du kernel
  dim3 threads(32, 16); // 32*16=512
  dim3 blocks;
  blocks.x = (m + threads.x - 1) / threads.x;
  blocks.y = (m + threads.y - 1) / threads.y;
  matMul<<<blocks, threads>>>(d_A, m, n, d_B, n, m, d_C, m, m);

  // attente de tous les threads
  cudaThreadSynchronize();

  // copie de la matrice equivalente C dans le hote
  cudaMemcpy(C, d_C, m * m * sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stdout, "Matrice A\n");
  affichage(A, m, n);
  fprintf(stdout, "Matrice B\n");
  affichage(B, n, m);
  fprintf(stdout, "Matrice C\n");
  affichage(C, m, m);

  // liberation memoire hote
  free(A);
  free(B);
  free(C);

  // liberation memoire device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
