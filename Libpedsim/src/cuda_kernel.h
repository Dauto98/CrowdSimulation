#pragma once

void cudaSetup(int n, float agentsX[], float agentsY[], float destX[], float destY[], float destR[]);
void cudaComputePosition(float agentsX[], float agentsY[], float destX[], float destY[], float destR[], int n, int reached[]);