#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include <sys/time.h>

#include "../include/ga.h"

#define PRINT 1

int aleatorio(int n) {
	return (rand() % n);  // genera un numero aleatorio entre 0 y n-1
}


int search_element(int *array, int end, int element)
{
	int i=0;
	int found=0;
	
	// comprueba que un elemento no está incluido en el individuo (el cual no admite enteros repetidos)
	while((i < end) && ! found) {
		if(array[i] == element) {
			found = 1;
		}
		i++;
	}
        return found;
}

int find_element(int *array, int end, int element)
{
        int pos = 0;
	for(int i = 0; i < end; i++) {
             if(array[i] == element) {
                 pos = i;
                 break;
             }
        }
        return pos; // Posición del elemento encontrado
}

int *crear_individuo(int n)
{
        // El primer elemento del individuo siempre será el 0, por ejemplo.
	int i=1, value;
	int *individuo = (int *) malloc(n * sizeof(int));
	
	// inicializa array de elementos
	memset(individuo, 0, n * sizeof(int));
	
	while(i < n) {
		value = aleatorio(n);
		// si el nuevo elemento no está en el array...
		if(!search_element(individuo, i, value)) {
			individuo[i] = value;  // lo incluimos
			i++;
		}
	}
	return individuo;
}

int comp_fitness(const void *a, const void *b) {
	/* qsort pasa un puntero al elemento que está ordenando */
	return (*(Individuo **)a)->fitness - (*(Individuo **)b)->fitness;
}

double aplicar_ga(const double *d, int n, int n_gen, int tam_pob, double m_rate, int *sol)
{
	int i, g, mutation_start;
	
	// crea poblacion inicial (array de individuos)
	Individuo **poblacion = (Individuo **) malloc(tam_pob * sizeof(Individuo *));
	assert(poblacion);
	
	
	for(i = 0; i < tam_pob; i++) {
		poblacion[i] = (Individuo *) malloc(sizeof(Individuo));
		poblacion[i]->array_int = crear_individuo(n);
			
		// calcula el fitness del individuo
		fitness(d, poblacion[i], n);
	}
	// ordena individuos segun la funcion de bondad (menor "fitness" --> mas aptos)
	qsort(poblacion, tam_pob, sizeof(Individuo *), comp_fitness);
	
	// evoluciona la poblacion durante un numero de generaciones
	for(g = 0; g < n_gen; g++)
	{
		// los hijos de los ascendientes mas aptos sustituyen a la ultima mitad de los individuos menos aptos
		
		for(i = 0; i < (tam_pob/2) - 1; i += 2) {
			cruzar(poblacion[i], poblacion[i+1], poblacion[tam_pob/2 + i], poblacion[tam_pob/2 + i + 1], n);
		}
		
		// por ejemplo, inicia la mutacion a partir de 1/4 de la poblacion.
                // puede haber otras opciones pero dejar el primer individuo sin modificar siempre
		mutation_start = tam_pob/4;
		
		// muta 3/4 partes de la poblacion
		for(i = mutation_start; i < tam_pob; i++) {
			mutar(poblacion[i], n, m_rate);
		}
		
		// recalcula el fitness del individuo
		for(i = 0; i < tam_pob; i++) {
			fitness(d, poblacion[i], n);
		}
		
		// ordena individuos segun la funcion de bondad (menor "fitness" --> mas aptos)
		qsort(poblacion, tam_pob, sizeof(Individuo *), comp_fitness);
		if (PRINT) {
			//printf("Generacion %d - ", g);
			//printf("Fitness = %.0lf\n", (poblacion[0]->fitness));
		}
		
	
	}
	memmove(sol, poblacion[0]->array_int, n*sizeof(int));
	
	// almacena el mejor valor obtenido para el fitness
	double value = (poblacion[0]->fitness);
	
	// se libera la memoria reservada
	for(i = 0; i < tam_pob; i++) {
		free(poblacion[i]->array_int);
		free(poblacion[i]);
	}
	free(poblacion);
	
	// devuelve el valor obtenido para el fitness
	return value;
}

void cruzar(Individuo *padre1, Individuo *padre2, Individuo *hijo1, Individuo *hijo2, int n)
{
	// Elegir un punto (o puntos) de corte aleatorio a partir del que se realiza el intercambio de los genes. 
    	// Genera un número aleatorio entre 0 y n
    	int num = aleatorio(n+1);
    	int i;  
   	
	// Entonces, por ejemplo, los primeros genes del padre1 van al hijo1, y los primeros del padre2 al hijo2.
	// Si pos es 0 en este paso no se hará nada e hijo1=padre2 e hijo2=padre1
	// Si pos es n, hijo1=padre1, hijo2=padre2
		
        memmove(hijo1->array_int, padre1->array_int, num*sizeof(int));
                  
        memmove(hijo2->array_int, padre2->array_int, num*sizeof(int));
                  
        // Y los restantes genes de cada hijo son del otro padre, respectivamente. 
                 
       	memmove(hijo1->array_int+num, padre2->array_int+num, (n-num)*sizeof(int));
	   
        memmove(hijo2->array_int+num, padre1->array_int+num, (n-num)*sizeof(int));
        
        // Se debe evitar en cada paso la introduccion de duplicados en los hijos.
	// Comprobamos en hijo1 de manera similar a la funcion crear_individuo.
        i = num;
	while(i < n) {
	// si el elemento no está en el array...
		if(!search_element(hijo1->array_int, i, hijo1->array_int[i])) {
			i++;
		}
		// Si esta, lo cambiamos
		else{
			hijo1->array_int[i] = aleatorio(n);
		}
				
	}
              
        //Hijo2
	i = num;
	while(i < n) {
	// si el elemento no está en el array...
		if(!search_element(hijo2->array_int, i, hijo2->array_int[i])) {
			i++;
		}
		// Si esta, lo cambiamos
		else{
			hijo2->array_int[i] = aleatorio(n);
		}
	}
            

        // Otra opción podría ser factibilizar a posteriori, despues de generar los descendientes: eliminar posibles 
        // repetidos de ambos hijos. Si encuentro algún elemento repetido en el hijo, lo cambio por otro que no este el array
}

void invertir(int *a, int k)
{
        int t;
	// Uno por uno invierte los elementos de a[0..k-1]
	for (int i = 0; i<k/2; i++){
		t = a[k-1-i];
		a[k-1-i] = a[i];
		a[i] = t;
	}
}

void mutamos(Individuo *actual, int n)
{
	// Implementación recomendada (aunque puede ser cualquier otra que se considere adecuada para este problema): 
	// Reverse Sequence Mutation (RSM), donde elegimos una secuencia S limitada por dos posiciones i, j
	// elegidas aleatoriamente con i<j, e i>0 para no modificar nunca el 1er elemento. El orden de los elementos en 
	// esta secuencia será invertido, por ejemplo con i=1, j=4: (1,2,3,4,5,6) --> (1,5,4,3,2,6).
	// Elegimos aleatoriamente i y j.
	int i = 0;
	int j = 0;
	// Mientras que i y j sean iguales o uno de los dos sea cero segumos buscando aleatoriamente.
	// así aseguramos acabar con dos números distintos y que no sean cero y si son distintos uno es mayor que el otro.
	while(i==j || j==0 || i==0){
		i = aleatorio(n);
		j = aleatorio(n);
	}
	// ordenamos para que i sea el menor y j el mayor
	if(i>j) {
		int aux;
		aux = i;
		i = j;
		j = aux;
	}
	// printf("\nLos valores de corte son i:%d, j: %d\n", i, j);
	// Llamamos a la funcion invertir sumandole a la direccion del array el valor i y pasando como límite el número
	// de elemntos entre i y j.
	
	invertir(actual->array_int+i, j-i+1);
}


void mutar(Individuo *actual, int n, double m_rate)
{
    	// Para cada individuo se ejecutaran x iteraciones en las que en cada una tendrá
    	// una probabilidad de m_rate para mutar. El número de iteraciones será m_rate * número de nodos.
    	int iteraciones = n*m_rate;
    	for (int i = 0; i< iteraciones; i++){
		// Generar un número decimal aleatorio entre 0 y 1
		
		double numeroAleatorio = (double)rand() / RAND_MAX;
		// Si este decimal es menor o igual a m_rate mutamos.
		if (numeroAleatorio <= m_rate)
			mutamos(actual, n);
        // Usar la variable m_rate para establecer la intensidad (iteraciones) de la mutación, teniendo en cuenta que
	// si el valor es demasiado pequeño la convergencia es muy pequeña y si es demasiado puede diverger.
	}
}

double distancia_ij(const double *d, int i, int j, int n)
{
	double dist = 0.0;
	// Devuelve la distancia entre dos elementos de la matriz 'd'
	// La distancia de i a j viene dada por la posición de la matriz d[i][j]
	//printf("La distancia entre %d y %d es de: %lf \n",i, j, *(d + i*n + j));
	dist = *(d + i*n + j);
	return dist;
}

void fitness(const double *d, Individuo *individuo, int n)
{
	// Determina la calidad del individuo calculando la suma de la distancia entre cada par de ciudades consecutivas en el array
	double fitness = 0.0;
	for (int i = 0; i<n; i++){
		if (i == (n-1)){
			fitness += distancia_ij(d,individuo->array_int[i],individuo->array_int[0], n);
		}
		else{
			fitness += distancia_ij(d,individuo->array_int[i],individuo->array_int[i+1], n);
		}
	}
	//printf("El valor del inividuo es de: %lf\n", fitness);
	// Asignamos el valor al individuo
	individuo->fitness = fitness;
}








