#include "cuda_runtime.h"
#include <stdio.h>

float *d_agentsX, *d_agentsY;
int *d_currentDest, *d_numWaypoint;
cudaPitchedPtr d_waypoint;

int blockSize;
int numBlocks;

__global__ void computePositionParallel(int n, float *agentsX, float *agentsY, cudaPitchedPtr waypoints, int* currentDest, int* numWaypoint) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	char * devPtr = (char *)waypoints.ptr;
	size_t pitch = waypoints.pitch;
	size_t slidePitch = pitch * waypoints.ysize;

	for (int i = index; i < n; i += stride) {
		// extract the current destination for this agents
		float * waypointCoordinate = (float *) ((devPtr + i * slidePitch) + currentDest[i] * pitch);

		// if there is no destination to go to
		if (waypointCoordinate[0] == -1 || waypointCoordinate[1] == -1) {
			continue;
		}

		// compute and update next position
		double diffX = waypointCoordinate[0] - agentsX[i];
		double diffY = waypointCoordinate[1] - agentsY[i];
		double length = sqrtf(diffX * diffX + diffY * diffY);
		agentsX[i] = (float)llrintf(agentsX[i] + diffX / length);
		agentsY[i] = (float)llrintf(agentsY[i] + diffY / length);

		// check if next position is inside the destination radius
		diffX = waypointCoordinate[0] - agentsX[i];
		diffY = waypointCoordinate[1] - agentsY[i];
		length = sqrtf(diffX * diffX + diffY * diffY);

		if (length < waypointCoordinate[2]) {
			if (currentDest[i] + 1 == numWaypoint[i]) {
				currentDest[i] = 0;
			} else {
				currentDest[i] += 1;
			}
		}
	}
}

void cudaSetup(int n, float agentsX[], float agentsY[], float*** waypoints, int numWaypoint[], int numWaypointMax) {
	blockSize = 1024;
	numBlocks = (n + blockSize - 1) / blockSize;
	
	// copy agents coordinate to GPU
	cudaMalloc((void **)&d_agentsX, sizeof(float) * n);
	cudaMalloc((void **)&d_agentsY, sizeof(float) * n);
	cudaMemcpy((void *)d_agentsX, (void*)agentsX, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)d_agentsY, (void*)agentsY, sizeof(float) * n, cudaMemcpyHostToDevice);

	// init array that keep track of current waypoint
	cudaMalloc((void **)&d_currentDest, sizeof(int) * n);
	int* currentDest = new int[n];
	for (int i = 0; i < n; i++) {
		currentDest[i] = 0;
	}
	cudaMemcpy((void *)d_currentDest, (void *)currentDest, sizeof(int) * n, cudaMemcpyHostToDevice);

	// array that hold the number of waypoint each agent has
	cudaMalloc((void **)&d_numWaypoint, sizeof(int) * n);
	cudaMemcpy((void *)d_numWaypoint, (void *)waypoints, sizeof(int) * n, cudaMemcpyHostToDevice);

	// init and copy the waypoint to GPU
	cudaExtent extent = make_cudaExtent(3 * sizeof(float), numWaypointMax, n);
	cudaMalloc3D(&d_waypoint, extent);

	cudaMemcpy3DParms myParms = { 0 };
	myParms.srcPtr.ptr = waypoints;
	myParms.srcPtr.pitch = 3 * sizeof(float);
	myParms.srcPtr.xsize = 3;
	myParms.srcPtr.ysize = numWaypointMax;
	myParms.dstPtr.ptr = d_waypoint.ptr;
	myParms.dstPtr.pitch = d_waypoint.pitch;
	myParms.dstPtr.xsize = 3;
	myParms.dstPtr.ysize = numWaypointMax;
	myParms.extent.width = 3 * sizeof(float);
	myParms.extent.height = numWaypointMax;
	myParms.extent.depth = n;
	myParms.kind = cudaMemcpyHostToDevice;
	
	cudaMemcpy3D(&myParms);
}

void cudaComputePosition(float agentsX[], float agentsY[], int n) {
	computePositionParallel<<<numBlocks, blockSize>>>(n, d_agentsX, d_agentsY, d_waypoint, d_currentDest, d_numWaypoint);

	cudaMemcpy((void *)agentsX, (void*)d_agentsX, sizeof(float) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)agentsY, (void*)d_agentsY, sizeof(float) * n, cudaMemcpyDeviceToHost);
}