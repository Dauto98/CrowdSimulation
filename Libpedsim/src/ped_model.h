//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>

#include "ped_agent.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// compute and update next positions for the next 4 agents starting from "index"
		void computeNextPosition(int start, int end);

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Return the agents' X coordinate vector reference
		const std::vector<float>& getAgentsX() const { return agentsX; }

		// Return the agents' Y coordinate vector reference
		const std::vector<float>& getAgentsY() const { return agentsY; }

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;
		
		// store agents coordinates into 2 vectors
		std::vector<float> agentsX;

		std::vector<float> agentsY;

		// store the waypoints for each agent into a 2D vector
		std::vector<std::deque<Twaypoint*>> waypoints;

		// store the current destination coordinate and the radius into 3 vectors
		std::vector<float> destX;

		std::vector<float> destY;

		std::vector<float> destR;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
	};
}
#endif
