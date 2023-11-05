#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <mpi.h>

#include "../include/io.h"

extern double aplicar_ga(int name, int p, const double *, int, int, int, double, int, int, int *);


static double mseconds() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1000 + t.tv_usec/1000;
}

int main(int argc, char **argv)
{
	int name, p, tag = 0;
	MPI_Status status;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&name);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	
	if(name == 0){
	//	Check Number of Input Args
		if(argc < 6) {
			fprintf(stderr,"Ayuda:\n"); 
			fprintf(stderr,"  ./programa n nGen tamPob mRate NGM NEM\n");
			MPI_Finalize();
			return(EXIT_FAILURE);
		}
	}
	
	int n = atoi(argv[1]);
	int n_gen = atoi(argv[2]);
	int tam_pob = atoi(argv[3]);
	double m_rate = atof(argv[4]);
	int ngm = atoi(argv[5]);
	int nem = atoi(argv[6]);
	
	double *d = NULL;
	if(name == 0){
	// 	Proceso 0 crea la matriz de distancias
	//	Generate matrix D with distance values among elements
		d = generar_matriz_distancias(n);
		
		#ifdef DEBUG
			//print_matrix(d, n);
		#endif
		
		// Y la envia al resto
		for(int dest = 1; dest < p; dest++){
			tag = 1;
			MPI_Send(d, n*n, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
		}
		
	}
	else{
		// El resto reserva memoria para la matriz 
		tag = 1;
		d = (double *)malloc(n * n * sizeof(double));
		// Y la recibe
		MPI_Recv(d, n*n, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
	}

	
//	Allocate memory for output data
	int *sol = (int *) malloc(n * sizeof(int));
	
	double ti, tf;
	if (name == 0){
		#ifdef TIME
			ti = mseconds();
		#endif
	}
	
//	Call Metaheuristic
	double value = aplicar_ga(name, p, d, n, n_gen, tam_pob, m_rate, ngm, nem, sol);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if (name == 0){
		#ifdef TIME
			tf = mseconds();
			printf("Execution Time: %.2lf sec\n", (tf - ti)/1000);
		#endif
		
		
		#ifdef DEBUG
			//print_solution(n, sol, value);
		#endif
	}
	// Free Allocated Memory	
	free(sol);

	free(d);
	
	MPI_Finalize();
	return(EXIT_SUCCESS);
}
