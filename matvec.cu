/*
* Rayhana ZIARA
* produit matrice vecteur
*/

#include <stdlib.h>
#include <stdio.h>

/*
* DESCRIPTION : kernel concernant le produit matrice vecteur
* PARAMETRES : matrice A, vecteur v, vecteur r et taille des vecteurs
* RETOUR : /
*/
__global__ void matVect(float *A, float *v, float *r, int size) 
{
  float resultat = 0.0;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if(index > size)
  {
    printf("ERREUR - Index > size\n");
    return;
  }

  for(int i = 0; i < size; i++)
    resultat += A[i * size + index] * v[i];

  r[index] = resultat;
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
  // declaration des variables du produit matrice vecteur
  // variables du hote
  float *A, *v, *r; 
  int n; // taille de la matrice et du vecteur

  // variables du device
  float *d_A, *d_v, *d_r;

  if(argc != 2)
  {
    fprintf(stderr, "ERREUR - Veuillez entrez la taille de A en parametre d'execution. Merci'\n./exam_rz n \n");
    return -1;
  }

  n = atoi(argv[1]); // taille de la matrice A(n * n) et du vecteur v (n)

  // allocation memoire dans le hote pour la matrice A et le vecteur d
  A = (float*)malloc(n * n * sizeof(float));
  v = (float*)malloc(n * sizeof(float));
  r = (float*)malloc(n * sizeof(float));

  // initialisation de la matrice A (matrice stockée en 1D) et du vecteur v
  for(int i = 0; i < n; i++)
  {
    v[i] = i * n;
    for(int j = 0; j < n; j++)
      A[i * n + j] = i * n + j;
  }

  // allocation memoire dans le device pour les equivalents de matrice A et du vecteur v
  cudaMalloc((void**)&d_A, n * n * sizeof(float));
  cudaMalloc((void**)&d_v, n * sizeof(float));
  cudaMalloc((void**)&d_r, n * sizeof(float));

  // copie de la matrice A et du vecteur v dans le device
  cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, n * sizeof(float), cudaMemcpyHostToDevice);

  // appel du kernel
  dim3 threads(4, 4); // 32*16
  dim3 blocks;
  blocks.x = (n + threads.x - 1) / threads.x;
  blocks.y = (n + threads.y - 1) / threads.y;
  matVect<<<blocks, threads>>>(d_A, d_v, d_r, n);

  // attente de tous les threads
  cudaThreadSynchronize();

  // copie de la matrice equivalente C dans le hote
  cudaMemcpy(r, d_r, n * sizeof(float), cudaMemcpyDeviceToHost);

  fprintf(stdout, "Matrice A\n");
  affichage(A, n, n);
  fprintf(stdout, "Vecteur v\n");
  affichage(v, 1, n);
  fprintf(stdout, "Vecteur r\n");
  affichage(r, 1, n);

  // liberation memoire hote
  free(A);
  free(v);
  free(r);

  // liberation memoire device
  cudaFree(d_A);
  cudaFree(d_v);
  cudaFree(d_r);

  return 0;
}
