#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <limits.h>

#define min 50
#define max 1000

double *generar_matriz_distancias(int n)
{
	double f;
	double *d = (double *) malloc(n * n *sizeof(double));

	srand(time(NULL) + getpid());
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			f = (double) rand() / ((double) RAND_MAX + 1);
			*(d + (i*n + j)) = (j != i) ? (min + f*(max - min)) : 0.0;
		}
	}
/* Ejemplo
*(d) = 0; *(d+1) = 2; *(d+2) = INT_MAX; *(d+3) = 12; *(d+4) = 5;
*(d+5) = 2; *(d+6) = 0; *(d+7) = 4; *(d+8) = 8; *(d+9) = INT_MAX;
*(d+10) = INT_MAX; *(d+11) = 4; *(d+12) = 0; *(d+13) = 3; *(d+14) = 3;
*(d+15) = 12; *(d+16) = 8; *(d+17) = 3; *(d+18) = 0; *(d+19) = 10;
*(d+20) = 5; *(d+21) = INT_MAX; *(d+22) = 3; *(d+23) = 10; *(d+24) = 0;
*/
	return d;
}

void print_matrix(double *d, int n)
{
	for(int i = 0; i < n*n; i++) {
		if ((i % n) != 0) { printf("%.0lf ", *(d + i)); }
		else { printf("\n%.0lf ", *(d + i)); }
	}
	printf("\n\n");
}

void print_solution(int n, const int *solucion, double valor)
{
	printf("\nSolution: ");
	for(int i = 0; i < n; i++) { printf("%d ", solucion[i]); }
	printf("\nDistance: %.0lf\n", valor);
}
