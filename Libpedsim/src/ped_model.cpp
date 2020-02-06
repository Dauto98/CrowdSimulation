//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include <Windows.h>

#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>

// Memory leak check with msvc++
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#define NUM_THREAD 8

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// pull out all agents coordinate, current destination coordinate and the Twaypoints vector into 5 vectors
	for (int i = 0; i < agents.size(); ++i) {
		agentsX.push_back((float) agents[i]->getX());
		agentsY.push_back((float) agents[i]->getY());

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

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implementation. Standard in the given code is SEQ
	this->implementation = implementation;

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
		agentsX[i] = (float)round(agentsX[i] + diffX / length);
		agentsY[i] = (float)round(agentsY[i] + diffY / length);

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

void Ped::Model::tick() {
	if (implementation == SEQ) {
		// sequential mode
		computeNextPosition(0, agentsX.size());
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
			agentsX[i] = (float)round(agentsX[i] + diffX / length);
			agentsY[i] = (float)round(agentsY[i] + diffY / length);

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
			_mm_store_ps(&agentsX[i], vAgentsX);
			_mm_store_ps(&agentsY[i], vAgentsY);

			// check if next position is near the destination radius
			vDiffX = _mm_sub_ps(vDestX, vAgentsX);
			vDiffY = _mm_sub_ps(vDestY, vAgentsY);
			vLength = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(vDiffX, vDiffX), _mm_mul_ps(vDiffY, vDiffY)));

			__m128 bitMask = _mm_cmplt_ps(vLength, vDestR);
			int mask = _mm_movemask_ps(bitMask);

			if (mask & 1) {
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
			if (mask & 2) {
				// take pop the current destination and append it to the end
				Ped::Twaypoint* destination = waypoints[i + 1].front();
				waypoints[i + 1].push_back(destination);
				waypoints[i + 1].pop_front();

				// update the new destination
				destination = waypoints[i + 1].front();
				destX[i + 1] = (float)destination->getx();
				destY[i + 1] = (float)destination->gety();
				destR[i + 1] = (float)destination->getr();
			}
			if (mask & 4) {
				// take pop the current destination and append it to the end
				Ped::Twaypoint* destination = waypoints[i + 2].front();
				waypoints[i + 2].push_back(destination);
				waypoints[i + 2].pop_front();

				// update the new destination
				destination = waypoints[i + 2].front();
				destX[i + 2] = (float)destination->getx();
				destY[i + 2] = (float)destination->gety();
				destR[i + 2] = (float)destination->getr();
			}
			if (mask & 8) {
				// take pop the current destination and append it to the end
				Ped::Twaypoint* destination = waypoints[i + 3].front();
				waypoints[i + 3].push_back(destination);
				waypoints[i + 3].pop_front();

				// update the new destination
				destination = waypoints[i + 3].front();
				destX[i + 3] = (float)destination->getx();
				destY[i + 3] = (float)destination->gety();
				destR[i + 3] = (float)destination->getr();
			}
		}
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
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
