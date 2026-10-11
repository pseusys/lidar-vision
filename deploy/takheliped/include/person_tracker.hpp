#pragma once

#include "ros/ros.h"

#include "geometry_msgs/Point.h"
#include "takheliped/detection.h"

#include "tracker_policy.hpp"


class PersonTracker {
    private:
        ros::NodeHandle handle;
        ros::Subscriber detection;
        ros::Publisher personPosition;

        TrackerPolicy policy;
        bool detectionInitialized = false;

        std::string inputTopic;
        std::vector<takheliped::Point> latestDetection;
        takheliped::Point person;
        bool personTracked = false;

        void detectionCallback(const takheliped::detection::ConstPtr& data);
        void update();

    public:
        PersonTracker(TrackerPolicy tracker, std::string input);
};
