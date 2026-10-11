#pragma once

#include "ros/ros.h"

#include "geometry_msgs/Point.h"
#include "sensor_msgs/LaserScan.h"
#include "nav_msgs/Odometry.h"

#include "takheliped/raw_data.h"
#include "takheliped/detector.hpp"

#include "tracker_policy.hpp"


class AlgorithmicDetector {
    private:
        takheliped::AlgorithmicDetector detector;

        ros::NodeHandle handle;
        ros::Subscriber rawData;
        ros::Publisher detectionData;

        std::vector<takheliped::Point> latestBottomScan, latestTopScan;
        takheliped::Point latestOdometry;

        bool rawDataInitialized = false;
        void rawDataCallback(const takheliped::raw_data::ConstPtr& data);

        void update();
    
    public:
        AlgorithmicDetector(int frequencyInit, int frequencyMax, float uncertaintyMax, float uncertaintyMin, float uncertaintyInc, float clusterThreshold, float distanceLevel, float legSizeMin, float legSizeMax, float chestSizeMin, float chestSizeMax, float legsDistanceMin, float legsDistanceMax, bool logging);
};
