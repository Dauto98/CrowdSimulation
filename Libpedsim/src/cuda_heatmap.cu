#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "ped_model.h"

#define WEIGHTSUM 273
#define BLOCK_SIZE 16

int * heatmap;
size_t heatmap_pitch;

int * scaled_heatmap;
size_t scaled_heatmap_pitch;

int * blurred_heatmap;
size_t blurred_heatmap_pitch;

float* d_desiredPositionX;
float* d_desiredPositionY;

#define cudaCheckError(ans) { cudaCheckErrorAssert((ans), __FILE__, __LINE__); }
inline void cudaCheckErrorAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
cudaStream_t stream;

void cudaSetupHeatmap(int n, int** cuda_blurred_heatmap) {
	cudaCheckError(cudaMallocPitch(&heatmap, &heatmap_pitch, SIZE * sizeof(int), SIZE));

	cudaCheckError(cudaMallocPitch(&scaled_heatmap, &scaled_heatmap_pitch, SCALED_SIZE * sizeof(int), SCALED_SIZE));

	cudaCheckError(cudaMallocHost(cuda_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int)));

	cudaCheckError(cudaMemset2D(heatmap, heatmap_pitch, 0, SIZE * sizeof(int), SIZE));

	cudaCheckError(cudaMallocPitch(&blurred_heatmap, &blurred_heatmap_pitch, SCALED_SIZE * sizeof(int), SCALED_SIZE));

	cudaCheckError(cudaMalloc(&d_desiredPositionX, n * sizeof(float)));

	cudaCheckError(cudaMalloc(&d_desiredPositionY, n * sizeof(float)));

	cudaCheckError(cudaStreamCreate(&stream));

	// because cudaMemset2D is asynchronous with respect to the host
	cudaCheckError(cudaDeviceSynchronize());
}

__global__ void computeHeatmap(float* desiredAgentsX, float* desiredAgentsY, int n, int* heatmap, size_t heatmap_pitch, int* scaled_heatmap, size_t scaled_heatmap_pitch) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Thread row and column block
	int row = threadIdx.y;
	int col = threadIdx.x;

	// x, y coordinate
	int x = blockCol * blockDim.x + col;
	int y = blockRow * blockDim.y + row;

	// fade heatmap
	int* heatPoint = (int*)((char*)heatmap + y * heatmap_pitch) + x;
	*heatPoint = (int)round((*heatPoint) * 0.80);

	// pull desiredAgentxX and Y array from global to shared memory, only 1 thread will do it
	extern __shared__ float desiredPosition[];
	
	if (row == 0 && col == 0) {
		for (int i = 0; i < n; i++) {
			desiredPosition[i] = desiredAgentsX[i];
			desiredPosition[i + n] = desiredAgentsY[i];
		}
	}

	__syncthreads();

	// Count how many agents want to go to each location
	for (int i = 0; i < n; i++) {
		int desiredX = (int)desiredPosition[i];
		int desiredY = (int)desiredPosition[i + n];

		if (x == desiredX && y == desiredY) {
			// intensify heat for better color results
			if ((*heatPoint) + 40 <= 255) {
				*heatPoint += 40;
			}
		}
	}
}

__global__ void computeScaledHeatmap(int* heatmap, size_t heatmap_pitch, int* scaled_heatmap, size_t scaled_heatmap_pitch) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Thread row and column block
	int row = threadIdx.y;
	int col = threadIdx.x;

	// x, y coordinate
	int x = blockCol * blockDim.x + col;
	int y = blockRow * blockDim.y + row;

	// Scale the data for visual representation
	int value = *((int*)((char*)heatmap + y * heatmap_pitch) + x);
	for (int r = 0; r < CELLSIZE; r++) {
		int* row = (int*)((char*)scaled_heatmap + (r + y * CELLSIZE) * scaled_heatmap_pitch);
		for (int c = 0; c < CELLSIZE; c++) {
			row[x * CELLSIZE + c] = value;
		}
	}
}

__global__ void blurfilterHeatmap(int* blurred_heatmap, size_t blurred_heatmap_pitch, int* scaled_heatmap, size_t scaled_heatmap_pitch) {
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Thread row and column block
	int row = threadIdx.y;
	int col = threadIdx.x;

	// x, y coordinate
	int x = blockCol * blockDim.x + col;
	int y = blockRow * blockDim.y + row;

	// pull subsection of scaled_heatmap from global memory to shared memory
	__shared__ int shared_scaled_heatmap[BLOCK_SIZE + 4][BLOCK_SIZE + 4];
	int* scaled_heatmap_row = (int*)((char*)scaled_heatmap + y * scaled_heatmap_pitch);
	shared_scaled_heatmap[row + 2][col + 2] = scaled_heatmap_row[x];

	// Apply gaussian blurfilter
	if (x > 1 && x < SCALED_SIZE - 2 && y > 1 && y < SCALED_SIZE - 2) {
		// pull missing data to shared memory
		if (row == 0) {
			shared_scaled_heatmap[1][col + 2] = *((int*)((char*)scaled_heatmap + (y - 1) * scaled_heatmap_pitch) + x);
			shared_scaled_heatmap[0][col + 2] = *((int*)((char*)scaled_heatmap + (y - 2) * scaled_heatmap_pitch) + x);
		}
		if (row == blockDim.y - 1) {
			shared_scaled_heatmap[BLOCK_SIZE + 2][col + 2] = *((int*)((char*)scaled_heatmap + (y + 1) * scaled_heatmap_pitch) + x);
			shared_scaled_heatmap[BLOCK_SIZE + 3][col + 2] = *((int*)((char*)scaled_heatmap + (y + 2) * scaled_heatmap_pitch) + x);
		}
		if (col == 0) {
			shared_scaled_heatmap[row + 2][1] = *((int*)((char*)scaled_heatmap + y * scaled_heatmap_pitch) + x - 1);
			shared_scaled_heatmap[row + 2][0] = *((int*)((char*)scaled_heatmap + y * scaled_heatmap_pitch) + x - 2);
		}
		if (col == blockDim.x - 1) {
			shared_scaled_heatmap[row + 2][BLOCK_SIZE + 2] = *((int*)((char*)scaled_heatmap + y * scaled_heatmap_pitch) + x + 1);
			shared_scaled_heatmap[row + 2][BLOCK_SIZE + 3] = *((int*)((char*)scaled_heatmap + y * scaled_heatmap_pitch) + x + 2);
		}
		if (row == 0 && col == 0) {
			shared_scaled_heatmap[0][0] = *((int*)((char*)scaled_heatmap + (y - 2) * scaled_heatmap_pitch) + x - 2);
			shared_scaled_heatmap[0][1] = *((int*)((char*)scaled_heatmap + (y - 2) * scaled_heatmap_pitch) + x - 1);
			shared_scaled_heatmap[1][0] = *((int*)((char*)scaled_heatmap + (y - 1) * scaled_heatmap_pitch) + x - 2);
			shared_scaled_heatmap[1][1] = *((int*)((char*)scaled_heatmap + (y - 1) * scaled_heatmap_pitch) + x - 1);
		}
		if (row == 0 && col == blockDim.x - 1) {
			shared_scaled_heatmap[0][BLOCK_SIZE + 2] = *((int*)((char*)scaled_heatmap + (y - 2) * scaled_heatmap_pitch) + x + 1);
			shared_scaled_heatmap[0][BLOCK_SIZE + 3] = *((int*)((char*)scaled_heatmap + (y - 2) * scaled_heatmap_pitch) + x + 2);
			shared_scaled_heatmap[1][BLOCK_SIZE + 2] = *((int*)((char*)scaled_heatmap + (y - 1) * scaled_heatmap_pitch) + x + 1);
			shared_scaled_heatmap[1][BLOCK_SIZE + 3] = *((int*)((char*)scaled_heatmap + (y - 1) * scaled_heatmap_pitch) + x + 2);
		}
		if (row == blockDim.y - 1 && col == 0) {
			shared_scaled_heatmap[BLOCK_SIZE + 2][0] = *((int*)((char*)scaled_heatmap + (y + 1) * scaled_heatmap_pitch) + x - 2);
			shared_scaled_heatmap[BLOCK_SIZE + 2][1] = *((int*)((char*)scaled_heatmap + (y + 1) * scaled_heatmap_pitch) + x - 1);
			shared_scaled_heatmap[BLOCK_SIZE + 3][0] = *((int*)((char*)scaled_heatmap + (y + 2) * scaled_heatmap_pitch) + x - 2);
			shared_scaled_heatmap[BLOCK_SIZE + 3][1] = *((int*)((char*)scaled_heatmap + (y + 2) * scaled_heatmap_pitch) + x - 1);
		}
		if (row == blockDim.y - 1 && col == blockDim.x - 1) {
			shared_scaled_heatmap[BLOCK_SIZE + 2][BLOCK_SIZE + 2] = *((int*)((char*)scaled_heatmap + (y + 1) * scaled_heatmap_pitch) + x + 1);
			shared_scaled_heatmap[BLOCK_SIZE + 2][BLOCK_SIZE + 3] = *((int*)((char*)scaled_heatmap + (y + 1) * scaled_heatmap_pitch) + x + 2);
			shared_scaled_heatmap[BLOCK_SIZE + 3][BLOCK_SIZE + 2] = *((int*)((char*)scaled_heatmap + (y + 2) * scaled_heatmap_pitch) + x + 1);
			shared_scaled_heatmap[BLOCK_SIZE + 3][BLOCK_SIZE + 3] = *((int*)((char*)scaled_heatmap + (y + 2) * scaled_heatmap_pitch) + x + 2);
		}

		__syncthreads();

		int sum = 0;
		for (int r = -2; r < 3; r++) {
			for (int c = -2; c < 3; c++) {
				sum += w[r + 2][c + 2] * shared_scaled_heatmap[row + 2 + r][col + 2 + c];
			}
		}
		int value = sum / WEIGHTSUM;
		int* row = (int*)((char*)blurred_heatmap + y * blurred_heatmap_pitch);
		row[x] = 0x00FF0000 | value << 24;
	} else {
		int* row = (int*)((char*)blurred_heatmap + y * blurred_heatmap_pitch);
		row[x] = 0x00FF0000;
	}
}

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data) {
	printf("%s", (char*)data);
}

void cudaUpdateHeatmap(float * desiredPositionX, float * desiredPositionY, int n, int* cuda_blurred_heatmap) {
	cudaEvent_t start, stop;
	cudaCheckError(cudaEventCreate(&start));
	cudaCheckError(cudaEventCreate(&stop));	float elapsedTime = 0.0;

	printf("start heatmap\n");

	/**
	* Copy desired position from host to device
	*/
	//cudaCheckError(cudaEventRecord(start, stream));
	cudaCheckError(cudaMemcpyAsync(d_desiredPositionX, desiredPositionX, n * sizeof(float), cudaMemcpyHostToDevice, stream));
	cudaCheckError(cudaMemcpyAsync(d_desiredPositionY, desiredPositionY, n * sizeof(float), cudaMemcpyHostToDevice, stream));
	cudaCheckError(cudaStreamAddCallback(stream, MyCallback, "Finish copy host to device\n", 0));
	//cudaCheckError(cudaEventRecord(stop, stream));
	//cudaCheckError(cudaEventSynchronize(stop));
	//cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));

	//printf("Time for memcpy host to device: %f\n", elapsedTime);

	/**
	* generate heatmap in kernel
	*/
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
	//cudaCheckError(cudaEventRecord(start, stream));
	computeHeatmap<<<dimGrid, dimBlock, n * 2 * sizeof(float), stream>>>(d_desiredPositionX, d_desiredPositionY, n, heatmap, heatmap_pitch, scaled_heatmap, scaled_heatmap_pitch);
	cudaCheckError(cudaStreamAddCallback(stream, MyCallback, "Finish generating heatmap\n", 0));
	//cudaCheckError(cudaEventRecord(stop, stream));
	//cudaCheckError(cudaEventSynchronize(stop));
	//cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Time for compute heatmap: %f\n", elapsedTime);
	//cudaCheckError(cudaPeekAtLastError());
	//cudaCheckError(cudaDeviceSynchronize());

	/**
	* generate scaled heatmap in kernel
	*/
	//cudaCheckError(cudaEventRecord(start, stream));
	computeScaledHeatmap<<<dimGrid, dimBlock, 0, stream>>>(heatmap, heatmap_pitch, scaled_heatmap, scaled_heatmap_pitch);
	cudaCheckError(cudaStreamAddCallback(stream, MyCallback, "Finish scaling heatmap\n", 0));
	//cudaCheckError(cudaEventRecord(stop, stream));
	//cudaCheckError(cudaEventSynchronize(stop));
	//cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Time for compute scaled heatmap: %f\n", elapsedTime);

	/**
	* Blur the scaled heatmap in kernel
	*/
	dim3 dimGrid2(SCALED_SIZE / BLOCK_SIZE, SCALED_SIZE / BLOCK_SIZE);
	//cudaCheckError(cudaEventRecord(start, stream));
	blurfilterHeatmap<<<dimGrid2, dimBlock, 0, stream>>>(blurred_heatmap, blurred_heatmap_pitch, scaled_heatmap, scaled_heatmap_pitch);
	cudaCheckError(cudaStreamAddCallback(stream, MyCallback, "Finish blurring heatmap\n", 0));
	//cudaCheckError(cudaEventRecord(stop, stream));
	//cudaCheckError(cudaEventSynchronize(stop));
	//cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Time for blur heatmap: %f\n", elapsedTime);
	//cudaCheckError(cudaPeekAtLastError());
	//cudaCheckError(cudaDeviceSynchronize());

	/**
	* Copy blurred heatmap from device to host
	*/
	//cudaCheckError(cudaEventRecord(start, stream));
	cudaCheckError(cudaMemcpy2DAsync(cuda_blurred_heatmap, SCALED_SIZE * sizeof(int), blurred_heatmap, blurred_heatmap_pitch, SCALED_SIZE * sizeof(int), SCALED_SIZE, cudaMemcpyDeviceToHost, stream));
	//cudaCheckError(cudaEventRecord(stop, stream));
	//cudaCheckError(cudaEventSynchronize(stop));
	//cudaCheckError(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Time for memcpy device to host: %f\n", elapsedTime);

	cudaCheckError(cudaStreamAddCallback(stream, MyCallback, "Finish stream operation\n", 0));
	printf("Finish Async call\n");
}

void heatmapSynchronize() {
	printf("start synchronize\n");
	cudaCheckError(cudaStreamSynchronize(stream));
	printf("finish synchronize\n");
}