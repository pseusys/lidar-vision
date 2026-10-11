#include <stdexcept>

#include "takheliped/misc.hpp"

#include "tracker_policy.hpp"


const std::string getStringFromTrackerPolicy(const TrackerPolicy policy) {
    switch (policy) {
        case TrackerPolicy::First: return "first";
        case TrackerPolicy::Tracked: return "tracked";
        case TrackerPolicy::Closest: return "closest";
        case TrackerPolicy::None: return "none";
        default: throw std::runtime_error("Bad tracker policy value!");
    }
}

const TrackerPolicy getTrackerPolicyFromString(const std::string string) {
    if (string == "first") return TrackerPolicy::First;
    if (string == "tracked") return TrackerPolicy::Tracked;
    if (string == "closest") return TrackerPolicy::Closest;
    if (string == "none") return TrackerPolicy::None;
    throw std::runtime_error("Bad tracker policy name '" + string + "'!");
}


const takheliped::Point trackFirst(const std::vector<takheliped::Point>& points) {
    return points[0];
}

const takheliped::Point trackTracked(const std::vector<takheliped::Point>& points, const takheliped::Point previous) {
    double minDist = previous.distanceTo(points[0]);
    takheliped::Point tracked = points[0];
    for (takheliped::Point point: points) if (previous.distanceTo(point) < minDist) {
        minDist = previous.distanceTo(point);
        tracked = point;
    }
    return tracked;
}

const takheliped::Point trackClosest(const std::vector<takheliped::Point>& points) {
    return trackTracked(points, takheliped::Point(0, 0));
}

const takheliped::Point track(const TrackerPolicy policy, const std::vector<takheliped::Point>& points) {
    takheliped::Point zero = takheliped::Point(0, 0);
    return track(policy, zero, points);
}

const takheliped::Point track(const TrackerPolicy policy, const takheliped::Point previous, const std::vector<takheliped::Point>& points) {
    if (points.size() < 1) throw std::runtime_error("Points vector is empty!");
    switch (policy) {
        case TrackerPolicy::First: return trackFirst(points);
        case TrackerPolicy::Tracked: return trackTracked(points, previous);
        case TrackerPolicy::Closest: return trackClosest(points);
        case TrackerPolicy::None: return takheliped::Point(0, 0);
        default: throw std::runtime_error("Bad tracker policy value!");
    }
}
