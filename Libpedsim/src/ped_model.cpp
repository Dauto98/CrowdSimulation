//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include <Windows.h>

#include <atomic>
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include "ped_region.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include "cuda_kernel.h"
#include <chrono>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#define NUM_THREAD 8
#define ZERO_DURATION std::chrono::duration<double, std::nano>(baseTime - baseTime).count()

static std::chrono::high_resolution_clock::time_point baseTime;

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation, IMPLEMENTATION moveImp)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// pull out all agents coordinate, current destination coordinate and the Twaypoints vector into 5 vectors
	for (int i = 0; i < agents.size(); ++i) {
		agentsX.push_back((float) agents[i]->getX());
		agentsY.push_back((float) agents[i]->getY());

		desiredAgentsX.push_back((float)agents[i]->getX());
		desiredAgentsY.push_back((float)agents[i]->getY());

		waypoints.push_back(agents[i]->getWaypoints());

		if (agents[i]->getWaypoints().size() == 0) {
			destX.push_back(-1);
			destY.push_back(-1);
			destR.push_back(-1);
		} else {
			destX.push_back((float)agents[i]->getWaypoints().front()->getx());
			destY.push_back((float)agents[i]->getWaypoints().front()->gety());
			destR.push_back((float)agents[i]->getWaypoints().front()->getr());
		}
	}

	// for VECTOR implementation, if agents size is not a multiple of 4, we pad the array until it is a multiple of 4
	if (implementation == VECTOR && agents.size() % 4 != 0) {
		for (int i = 0, length = 4 - agents.size() % 4; i < length; i++) {
			agentsX.push_back(0);
			agentsY.push_back(0);

			destX.push_back(0);
			destY.push_back(0);
			destR.push_back(0);
		}
	}

	if (implementation == CUDA) {
		reached = new int[agentsX.size()];
		for (int i = 0; i < agentsX.size(); i++) {
			reached[i] = 0;
		}
		cudaSetup(agentsX.size(), &agentsX[0], &agentsY[0], &destX[0], &destY[0], &destR[0]);
	}

	// setup regions for parallelizing move function
	for (int i = 0; i < 4; i++) {
		regionAgentList.push_back(std::vector<int>());
	}

	regionList.push_back(Ped::Region(std::make_pair(80, 60), 0, 10));
	regionList.push_back(Ped::Region(std::make_pair(80, 60), 10, 25));
	regionList.push_back(Ped::Region(std::make_pair(80, 60), 25, 50));
	regionList.push_back(Ped::Region(std::make_pair(80, 60), 50, 100));

	// init agentsIsBeingProcessed and split agents into regions
	agentsIsBeingProcessed = std::vector<std::atomic<double>>(agentsX.size());
	for (int i = 0; i < agentsX.size(); i++) {
		agentsIsBeingProcessed[i].store(ZERO_DURATION);

		for (int j = 0; j < regionList.size(); j++) {
			if (regionList[j].isInside(agentsX[i], agentsY[i])) {
				regionAgentList[j].push_back(i);
				break;
			}
		}
	}

	baseTime = std::chrono::high_resolution_clock::now();

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implementation. Standard in the given code is SEQ
	this->implementation = implementation;
	this->moveImp = moveImp;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}

void Ped::Model::computeNextPosition(int start, int end) {
	for (int i = start; i < end; i++) {
		// if there is no destination to go to
		if (destX[i] == -1 || destY[i] == -1) {
			continue;
		}

		// compute and update next position		
		double diffX = destX[i] - agentsX[i];
		double diffY = destY[i] - agentsY[i];
		double length = sqrt(diffX * diffX + diffY * diffY);

		desiredAgentsX[i] = (float)round(agentsX[i] + diffX / length);
		desiredAgentsY[i] = (float)round(agentsY[i] + diffY / length);
	}
}

void Ped::Model::tick() {
	if (implementation == SEQ) {
		// sequential mode
		computeNextPosition(0, agentsX.size());
		/*for (int i = 0; i < agentsX.size(); i++) {
			agentsX[i] = desiredAgentsX[i];
			agentsY[i] = desiredAgentsY[i];
		}*/
		move();
	} else if (implementation == PTHREAD) {
		int chunkSize = agentsX.size() / NUM_THREAD;
		int numBiggerList = agentsX.size() % NUM_THREAD;

		// create thread
		std::thread threads[NUM_THREAD];
		for (int i = 0; i < NUM_THREAD; i++) {
			if (i < numBiggerList) {
				threads[i] = std::thread(&Ped::Model::computeNextPosition, this, i * (chunkSize + 1), (i + 1) * (chunkSize + 1));
			} else {
				threads[i] = std::thread(&Ped::Model::computeNextPosition, this, i * chunkSize + numBiggerList, (i + 1) * chunkSize + numBiggerList);
			}
		}

		for (int i = 0; i < NUM_THREAD; i++) {
			threads[i].join();
		}

		move();
	}
	else if (implementation == OMP) {
		omp_set_dynamic(0);

		#pragma omp parallel for num_threads(NUM_THREAD)
		for (int i = 0; i < agentsX.size(); i++) {
			// if there is no destination to go to
			if (destX[i] == -1 || destY[i] == -1) {
				continue;
			}

			// compute and update next position		
			double diffX = destX[i] - agentsX[i];
			double diffY = destY[i] - agentsY[i];
			double length = sqrt(diffX * diffX + diffY * diffY);

			desiredAgentsX[i] = (float)round(agentsX[i] + diffX / length);
			desiredAgentsY[i] = (float)round(agentsY[i] + diffY / length);
		}

		move();
	}
	else if (implementation == VECTOR) {
		for (int i = 0; i < agentsX.size(); i += 4) {
			__m128 vDestX, vDestY, vDestR, vAgentsX, vAgentsY, vDiffX, vDiffY, vLength;
			// load value to vector register
			vDestX = _mm_load_ps(&destX[i]);
			vDestY = _mm_load_ps(&destY[i]);
			vDestR = _mm_load_ps(&destR[i]);
			vAgentsX = _mm_load_ps(&agentsX[i]);
			vAgentsY = _mm_load_ps(&agentsY[i]);

			// if destX, Y, R = -1 -> no destination -> set mask bit to zero, otherwise all 1
			__m128 noDestMask = _mm_cmpge_ps(vDestX, _mm_setzero_ps());
			
			// calculate distance
			vDiffX = _mm_sub_ps(vDestX, vAgentsX);
			vDiffY = _mm_sub_ps(vDestY, vAgentsY);
			vLength = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(vDiffX, vDiffX), _mm_mul_ps(vDiffY, vDiffY)));
			// get new position
			// here we AND the diff/length with mask to zero all the element that don't have destination
			vAgentsX = _mm_round_ps(_mm_add_ps(vAgentsX, _mm_and_ps(_mm_div_ps(vDiffX, vLength), noDestMask)), _MM_FROUND_TO_NEAREST_INT);
			vAgentsY = _mm_round_ps(_mm_add_ps(vAgentsY, _mm_and_ps(_mm_div_ps(vDiffY, vLength), noDestMask)), _MM_FROUND_TO_NEAREST_INT);
			_mm_store_ps(&desiredAgentsX[i], vAgentsX);
			_mm_store_ps(&desiredAgentsY[i], vAgentsY);
		}

		if (moveImp == SEQ) {
			moveSeq();
		} else {
			move();
		}
	} else if (implementation == CUDA) {
		cudaComputePosition(&agentsX[0], &agentsY[0], &desiredAgentsX[0], &desiredAgentsY[0], &destX[0], &destY[0], &destR[0], agentsX.size(), reached);

		move();
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

int getRegion(float x, float y, std::vector<Ped::Region> &regionList) {
	for (int i = 0; i < regionList.size(); i++) {
		if (regionList[i].isInside(x, y)) {
			return i;
		}
	}
	return -1;
}

float calcDistance(float x1, float y1, float x2, float y2) {
	return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

float getSplitRadius(std::vector<int>& agentList, std::vector<float>& agentsX, std::vector<float>& agentsY) {
	std::vector<float> distances = std::vector<float>();

	for (int i = 0; i < agentList.size(); i++) {
		distances.push_back(calcDistance(agentsX[agentList[i]], agentsY[agentList[i]], 80, 60));
	}

	std::sort(distances.begin(), distances.end());

	return distances[(int)(distances.size() / 2)];
}

void Ped::Model::moveSeq() {
	for (int i = 0; i < agentsX.size(); i++) {
		// if there is no destination to go to
		if (destX[i] == -1 || destY[i] == -1) {
			continue;
		}

		// compute and update next position
		double diffX = destX[i] - agentsX[i];
		double diffY = destY[i] - agentsY[i];
		double length = sqrt(diffX * diffX + diffY * diffY);

		std::vector<std::pair<float, float> > prioritizedAlternatives;
		std::pair<float, float> pDesired((float)round(agentsX[i] + diffX / length), (float)round(agentsY[i] + diffY / length));
		prioritizedAlternatives.push_back(pDesired);

		std::pair<float, float> p1, p2;
		if (pDesired.first == agentsX[i] || pDesired.second == agentsY[i]) {
			// Agent wants to walk straight to North, South, West or East
			p1 = std::make_pair((float)round(pDesired.first + diffY / length), (float)round(pDesired.second + diffX / length));
			p2 = std::make_pair((float)round(pDesired.first - diffY / length), (float)round(pDesired.second - diffX / length));

			prioritizedAlternatives.push_back(p1);
			prioritizedAlternatives.push_back(p2);
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] + diffY / length), (float)round(agentsY[i] + diffX / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffY / length), (float)round(agentsY[i] - diffX / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] + (diffY - diffX) / length), (float)round(agentsY[i] + (diffX - diffY) / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - (diffY - diffX) / length), (float)round(agentsY[i] - (diffX - diffY) / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), (float)round(agentsY[i] - diffY / length)));
		}
		else {
			// Agent wants to walk diagonally
			p1 = std::make_pair(pDesired.first, agentsY[i]);
			p2 = std::make_pair(agentsX[i], pDesired.second);

			prioritizedAlternatives.push_back(p1);
			prioritizedAlternatives.push_back(p2);
			prioritizedAlternatives.push_back(std::make_pair(pDesired.first, (float)round(agentsY[i] - diffY / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), pDesired.second));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), agentsY[i]));
			prioritizedAlternatives.push_back(std::make_pair(agentsX[i], (float)round(agentsY[i] - diffY / length)));
			prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), (float)round(agentsY[i] - diffY / length)));
		}

		// Find the first empty alternative position
		for (std::vector<pair<float, float> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
			int empty = 1;
			for (int j = 0; j < agentsX.size(); j++) {
				if (i != j && it->first == agentsX[j] && it->second == agentsY[j]) {
					empty = 0;
					break;
				}
			}
			if (empty == 1) {
				agentsX[i] = it->first;
				agentsY[i] = it->second;
				break;
			}
		}

		// check if next position is inside the destination radius
		diffX = destX[i] - agentsX[i];
		diffY = destY[i] - agentsY[i];
		length = sqrt(diffX * diffX + diffY * diffY);

		if (length < destR[i]) {
			// take pop the current destination and append it to the end
			Ped::Twaypoint* destination = waypoints[i].front();
			waypoints[i].push_back(destination);
			waypoints[i].pop_front();

			// update the new destination
			destination = waypoints[i].front();
			destX[i] = (float)destination->getx();
			destY[i] = (float)destination->gety();
			destR[i] = (float)destination->getr();
		}
	}
}

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move() {
	#pragma omp parallel for num_threads(regionAgentList.size())
	for (int j = 0; j < regionAgentList.size(); j++)	{
		for (int k = 0; k < regionAgentList[j].size(); k++) {
			// get the index of the agent in the region
			int i = regionAgentList[j][k];
			// if there is no destination to go to
			if (destX[i] == -1 || destY[i] == -1) {
				continue;
			}

			std::vector<std::pair<float, float>> prioritizedAlternatives = computeAlternative(i);

			setPositionNoCollision(prioritizedAlternatives, i, j);
		}
	}

	// check if any agents have move to another region
	for (int i = 0; i < regionAgentList.size(); i++) {
		for (std::vector<int>::iterator j = regionAgentList[i].begin(); j != regionAgentList[i].end();) {
			int nextRegion = getRegion(agentsX[*j], agentsY[*j], regionList);
			if (nextRegion != i) {
				regionAgentList[nextRegion].push_back(*j);
				j = regionAgentList[i].erase(j);
			} else {
				j++;
			}
		}
	}

	// merge the regions
	int baseNumAgents = agentsX.size() / 6;
	for (int i = 0, size = regionList.size(); i < size; i++) {
		if (regionAgentList[i].size() < baseNumAgents - 20) {
			int regionToMergeWith;
			if (i == 0) {
				regionToMergeWith = 1;
			} else if (i == regionList.size() - 1) {
				regionToMergeWith = i - 1;
			} else if (regionAgentList[i - 1].size() < regionAgentList[i + 1].size()) {
				regionToMergeWith = i - 1;
			} else {
				regionToMergeWith = i + 1;
			}
			// merge region agent list
			regionAgentList[i].insert(regionAgentList[i].end(), regionAgentList[regionToMergeWith].begin(), regionAgentList[regionToMergeWith].end());
			regionAgentList.erase(regionAgentList.begin() + regionToMergeWith);
			
			// create new region and insert it to the regionList
			float innerRadius = i < regionToMergeWith ? regionList[i].getInnerRadius() : regionList[regionToMergeWith].getInnerRadius();
			float outerRadius = i < regionToMergeWith ? regionList[regionToMergeWith].getOuterRadius() : regionList[i].getOuterRadius();
			Ped::Region newRegion = Ped::Region(std::make_pair(80, 60), innerRadius, outerRadius);
			if (regionToMergeWith < i) {
				regionList.insert(regionList.begin() + regionToMergeWith, newRegion);
				regionList.erase(regionList.begin() + regionToMergeWith + 1);
				regionList.erase(regionList.begin() + regionToMergeWith + 1);
				i -= 2;
			} else {
				regionList.insert(regionList.begin() + i, newRegion);
				regionList.erase(regionList.begin() + i + 1);
				regionList.erase(regionList.begin() + i + 1);
				i--;
			}
			size--;
		}
	}

	// split the regions
	for (int i = 0, size = regionList.size(); i < size; i++) {
		if (regionAgentList[i].size() > baseNumAgents + 20) {
			float splitRadius = getSplitRadius(regionAgentList[i], agentsX, agentsY);
			Ped::Region newRegion1 = Ped::Region(std::make_pair(80, 60), regionList[i].getInnerRadius(), splitRadius);
			Ped::Region newRegion2 = Ped::Region(std::make_pair(80, 60), splitRadius, regionList[i].getOuterRadius());
			std::vector<int> newAgentList1 = std::vector<int>();
			std::vector<int> newAgentList2 = std::vector<int>();

			for (int j = 0; j < regionAgentList[i].size(); j++) {
				if (newRegion1.isInside(agentsX[regionAgentList[i][j]], agentsY[regionAgentList[i][j]]) == 1) {
					newAgentList1.push_back(regionAgentList[i][j]);
				} else {
					newAgentList2.push_back(regionAgentList[i][j]);
				}
			}

			regionList.insert(regionList.begin() + i, newRegion1);
			regionList.insert(regionList.begin() + i + 1, newRegion2);
			regionList.erase(regionList.begin() + i + 2);

			regionAgentList.insert(regionAgentList.begin() + i, newAgentList1);
			regionAgentList.insert(regionAgentList.begin() + i + 1, newAgentList2);
			regionAgentList.erase(regionAgentList.begin() + i + 2);

			size++;
			i++;
		}
	}
}

std::vector<std::pair<float, float>> Ped::Model::computeAlternative(int i) {
	// get the length between the agent and destination
	double diffX = destX[i] - agentsX[i];
	double diffY = destY[i] - agentsY[i];
	double length = sqrt(diffX * diffX + diffY * diffY);

	std::vector<std::pair<float, float>> prioritizedAlternatives;
	std::pair<float, float> pDesired(desiredAgentsX[i], desiredAgentsY[i]);
	prioritizedAlternatives.push_back(pDesired);

	std::pair<float, float> p1, p2;
	if (pDesired.first == agentsX[i] || pDesired.second == agentsY[i]) {
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair((float)round(pDesired.first + diffY / length), (float)round(pDesired.second + diffX / length));
		p2 = std::make_pair((float)round(pDesired.first - diffY / length), (float)round(pDesired.second - diffX / length));

		prioritizedAlternatives.push_back(p1);
		prioritizedAlternatives.push_back(p2);
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] + diffY / length), (float)round(agentsY[i] + diffX / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffY / length), (float)round(agentsY[i] - diffX / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] + (diffY - diffX) / length), (float)round(agentsY[i] + (diffX - diffY) / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - (diffY - diffX) / length), (float)round(agentsY[i] - (diffX - diffY) / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), (float)round(agentsY[i] - diffY / length)));
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agentsY[i]);
		p2 = std::make_pair(agentsX[i], pDesired.second);

		prioritizedAlternatives.push_back(p1);
		prioritizedAlternatives.push_back(p2);
		prioritizedAlternatives.push_back(std::make_pair(pDesired.first, (float)round(agentsY[i] - diffY / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), pDesired.second));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), agentsY[i]));
		prioritizedAlternatives.push_back(std::make_pair(agentsX[i], (float)round(agentsY[i] - diffY / length)));
		prioritizedAlternatives.push_back(std::make_pair((float)round(agentsX[i] - diffX / length), (float)round(agentsY[i] - diffY / length)));
	}

	return prioritizedAlternatives;
}

void Ped::Model::setPositionNoCollision(std::vector<std::pair<float, float> > prioritizedAlternatives, int i, int j) {
	// Find the first empty alternative position
	agentsIsBeingProcessed[i].store(std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now() - baseTime).count());
	for (std::vector<pair<float, float> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {
		int empty = 1;
		for (int l = 0; l < agentsX.size(); l++) {
			// skip agents that are too far away
			if (i != l && it->first - agentsX[l] < 3 && it->first - agentsX[l] > -3 && it->second - agentsY[l] < 3 && it->second - agentsY[l] > -3) {
				// check that if the agent l is being processed by another thread. If yes, and it is started earlier than the current thread,
				// the current thread wait until it is done
				double otherAgent = agentsIsBeingProcessed[l].load();
				double thisAgent = agentsIsBeingProcessed[i].load();
				while (otherAgent != ZERO_DURATION && (otherAgent < thisAgent || (otherAgent == thisAgent && l < i))) {
					otherAgent = agentsIsBeingProcessed[l].load();
				}

				if ((int)it->first == (int)agentsX[l] && (int)it->second == (int)agentsY[l]) {
					empty = 0;
					break;
				}
			}
		}
		if (empty == 1) {
			agentsX[i] = it->first;
			agentsY[i] = it->second;
		
			// check if next position is inside the destination radius
			float diffX = destX[i] - agentsX[i];
			float diffY = destY[i] - agentsY[i];
			float length = sqrt(diffX * diffX + diffY * diffY);

			if (length < destR[i]) {
				// take pop the current destination and append it to the end
				Ped::Twaypoint* destination = waypoints[i].front();
				waypoints[i].push_back(destination);
				waypoints[i].pop_front();

				// update the new destination
				destination = waypoints[i].front();
				destX[i] = (float)destination->getx();
				destY[i] = (float)destination->gety();
				destR[i] = (float)destination->getr();
			}

			break;
		}
	}
	agentsIsBeingProcessed[i].store(ZERO_DURATION);
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
