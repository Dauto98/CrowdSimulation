#pragma once

void cudaSetup(int n, float agentsX[], float agentsY[], float*** waypoints, int numWaypoint[], int numWaypointMax);
void cudaComputePosition(float agentsX[], float agentsY[], int n);