#include "cuda_runtime.h"
#include <stdio.h>

__global__ void computePositionParallel(float *agentsX, float *agentsY, float *destX, float *destY, float *destR, int n, int *reached) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i = index; i < n; i += stride) {
		// if there is no destination to go to
		if (destX[i] == -1 || destY[i] == -1) {
			continue;
		}

		// compute and update next position
		double diffX = destX[i] - agentsX[i];
		double diffY = destY[i] - agentsY[i];
		double length = sqrtf(diffX * diffX + diffY * diffY);
		agentsX[i] = (float)llrintf(agentsX[i] + diffX / length);
		agentsY[i] = (float)llrintf(agentsY[i] + diffY / length);

		// check if next position is inside the destination radius
		diffX = destX[i] - agentsX[i];
		diffY = destY[i] - agentsY[i];
		length = sqrtf(diffX * diffX + diffY * diffY);

		if (length < destR[i]) {
			reached[i] = 1;
		}
	}
}

float *d_agentsX, *d_agentsY, *d_destX, *d_destY, *d_destR;
int *d_reached;

void cudaSetup(int n, float agentsX[], float agentsY[], float destX[], float destY[], float destR[]) {
	cudaMalloc((void **)&d_agentsX, sizeof(float) * n);
	cudaMalloc((void **)&d_agentsY, sizeof(float) * n);
	cudaMalloc((void **)&d_destX, sizeof(float) * n);
	cudaMalloc((void **)&d_destY, sizeof(float) * n);
	cudaMalloc((void **)&d_destR, sizeof(float) * n);
	cudaMalloc((void **)&d_reached, sizeof(int) * n);
}

void cudaComputePosition(float agentsX[], float agentsY[], float desiredAgentsX[], float desiredAgentsY[], float destX[], float destY[], float destR[], int n, int reached[]) {
	int blockSize = 1024;
	int numBlocks = (n + blockSize - 1) / blockSize;

	cudaMemcpy((void *)d_agentsX, (void*)agentsX, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_agentsY, (void*)agentsY, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_destX, (void*)destX, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_destY, (void*)destY, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_destR, (void*)destR, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_reached, (void*)reached, sizeof(int) * n, cudaMemcpyHostToDevice);

	computePositionParallel<<<numBlocks, blockSize>>>(d_agentsX, d_agentsY, d_destX, d_destY, d_destR, n, d_reached);

	cudaMemcpy((void *)desiredAgentsX, (void*)d_agentsX, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)desiredAgentsY, (void*)d_agentsY, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)reached, (void*)d_reached, sizeof(int) * n, cudaMemcpyDeviceToHost);
}