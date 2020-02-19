#include "ped_region.h"

Ped::Region::Region(std::pair<int, int> upperLeft, std::pair<int, int> lowerRight) : upperLeftBorderPoint(upperLeft), lowerRightBorderPoint(lowerRight) {};

int Ped::Region::isInside(float x, float y) {
	if (x >= upperLeftBorderPoint.first && x < lowerRightBorderPoint.first && y >= upperLeftBorderPoint.second && y < lowerRightBorderPoint.second) {
		return 1;
	}
	return 0;
}

std::pair<std::pair<int, int>, std::pair<int, int>> Ped::Region::getBorder() {
	return std::make_pair(upperLeftBorderPoint, lowerRightBorderPoint);
}