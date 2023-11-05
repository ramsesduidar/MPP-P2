#ifndef _GA
#define _GA
	
	typedef struct {
		int *array_int;
		double fitness;
	} Individuo;
	
	void cruzar(Individuo *, Individuo *, Individuo *, Individuo *, int);
	void fitness(const double *, Individuo *, int);
	void fitness_atomic(const double *, Individuo *, int);
	void fitness_critical(const double *, Individuo *, int);
	void fitness_reduction(const double *, Individuo *, int);
	void mutar(Individuo *, int, double);
	void mergeSort(Individuo **poblacion, int izq, int der);
#endif
