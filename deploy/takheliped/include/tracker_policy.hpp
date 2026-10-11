#pragma once

#include <string>
#include <vector>

#include "takheliped/point.hpp"


enum TrackerPolicy {
    First,
    Tracked,
    Closest,
    None
};

const std::string getStringFromTrackerPolicy(const TrackerPolicy policy);
const TrackerPolicy getTrackerPolicyFromString(const std::string string);

const takheliped::Point track(const TrackerPolicy policy, const std::vector<takheliped::Point>& points);
const takheliped::Point track(const TrackerPolicy policy, const takheliped::Point previous, const std::vector<takheliped::Point>& points);
