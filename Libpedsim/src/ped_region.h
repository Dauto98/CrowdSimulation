#ifndef _ped_region_h_
#define _ped_region_h_

#include <utility>

namespace Ped {
	class Region {
	public:
		int isInside(float x, float y);

		std::pair<std::pair<int, int>, std::pair<int, int>> getBorder();

		Region(std::pair<int, int> upperLeft, std::pair<int, int> lowerRight);

	private:
		std::pair<int, int> upperLeftBorderPoint;

		std::pair<int, int> lowerRightBorderPoint;
	};
}

#endif
