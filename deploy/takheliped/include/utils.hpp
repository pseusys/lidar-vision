#pragma once

#include "geometry_msgs/Point.h"

#include "takheliped/point.hpp"


geometry_msgs::Point cartesianToPolar(const geometry_msgs::Point point);
geometry_msgs::Point polarToCartesian(const geometry_msgs::Point point);

takheliped::Point geometryToPoint(const geometry_msgs::Point point);
geometry_msgs::Point pointToGeometry(const takheliped::Point point);
