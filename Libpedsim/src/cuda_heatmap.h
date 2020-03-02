void cudaSetupHeatmap(int n, int* &cuda_blurred_heatmap);

void cudaUpdateHeatmap(float * desiredPositionX, float * desiredPositionY, int n, int* cuda_blurred_heatmap);

void heatmapSynchronize();