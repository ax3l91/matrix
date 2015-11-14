#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "../matrix/definitions.h"

__global__ void multi(int *A, int *B, int *C)
{
	int cvalue = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;


	if (row > DIM || col > DIM) return;

	for (int e = 0; e < DIM; ++e){
		cvalue += A[row*DIM + e] * B[e*DIM + col];
	}
	C[row*DIM + col] = cvalue;
}

void matrixmulti(int A[][DIM],int B[][DIM],int C[][DIM]){
	int *dev_a, *dev_b, *dev_c;

	//allocate memory on global memory of gpu
	cudaError_t err = cudaMalloc((void**)&dev_a, ((DIM)*(DIM))*sizeof(int));
	printf("Cuda malloc A:%s \n", cudaGetErrorString(err));
	err = cudaMalloc((void**)&dev_b, ((DIM)*(DIM))*sizeof(int));
	printf("Cuda malloc B:%s \n", cudaGetErrorString(err));
	err = cudaMalloc((void**)&dev_c, ((DIM)*(DIM))*sizeof(int));
	printf("Cuda malloc C:%s \n", cudaGetErrorString(err));


	//Copy array A and B on device allocated memory
	err = cudaMemcpy(dev_a, A, ((DIM*DIM))*sizeof(int), cudaMemcpyHostToDevice);
	printf("Cuda memcpy to device A:%s \n", cudaGetErrorString(err));
	err = cudaMemcpy(dev_b, B, ((DIM*DIM))*sizeof(int), cudaMemcpyHostToDevice);
	printf("Cuda memcpy to device B:%s \n", cudaGetErrorString(err));

	//two dimension threads
	dim3 dimBlock(BlockSize, BlockSize);
	dim3 dimGrid((DIM + dimBlock.x - 1) / dimBlock.x, (DIM + dimBlock.y - 1) / dimBlock.y);

	//call the kernel function multi
	multi << < dimGrid, dimBlock >> >(dev_a, dev_b, dev_c);

	//retrieve array C from device memory
	err = cudaMemcpy(C, dev_c, ((DIM*DIM))*sizeof(int), cudaMemcpyDeviceToHost);
	printf("Cuda memcpy to HOST C:%s \n", cudaGetErrorString(err));

	for (int i = 0; i < DIM; i++){
		for (int j = 0; j < DIM; j++){
			printf("C(%d,%d) = %d \n", i, j, C[i][j]);
		}
	}

	//free the memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}