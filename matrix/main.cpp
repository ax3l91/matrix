#include <stdio.h>
#include <random>
#include <time.h>
#include <math.h>

#include "../matrix/definitions.h"
#include "../matrix/matrixKernel.cuh"

int main(){

	srand(time(0));
	auto A = new int[DIM][DIM];
	auto B = new int[DIM][DIM];
	auto C = new int[DIM][DIM];

	for (int i = 0; i<DIM; i++){
		for (int j = 0; j < DIM; j++){
			A[i][j] = rand() % 100;
			B[i][j] = rand() % 100;
		}
	}
	matrixmulti(A,B,C);

}