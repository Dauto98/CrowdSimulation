#include <math.h>

#include "ped_region.h"

Ped::Region::Region(std::pair<int, int> center, float innerRadius, float outerRadius) : center(center), innerRadius(innerRadius), outerRadius(outerRadius) {};

int Ped::Region::isInside(float x, float y) {
	float distance = sqrt((x - center.first)*(x - center.first) + (y - center.second)*(y - center.second));
	if (distance >= innerRadius && distance < outerRadius) {
		return 1;
	}
	return 0;
}

std::pair<int, int> Ped::Region::getCenter() {
	return center;
}

float Ped::Region::getInnerRadius() {
	return innerRadius;
}

float Ped::Region::getOuterRadius() {
	return outerRadius;
}