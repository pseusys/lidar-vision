#include "algorithmic_detector.hpp"

#include "takheliped/detector.hpp"

#include "takheliped/detection.h"

#include "main.hpp"
#include "utils.hpp"


void AlgorithmicDetector::rawDataCallback(const takheliped::raw_data::ConstPtr& data) {
    latestBottomScan = std::vector<takheliped::Point>(data->bottom_lidar.size());
    for (int i = 0; i < data->bottom_lidar.size(); i++)
        latestBottomScan[i] = geometryToPoint(polarToCartesian(data->bottom_lidar[i]));

    latestTopScan = std::vector<takheliped::Point>(data->top_lidar.size());
    for (int i = 0; i < data->top_lidar.size(); i++)
        latestTopScan[i] = geometryToPoint(polarToCartesian(data->top_lidar[i]));
    
    latestOdometry = geometryToPoint(data->odometry);

    rawDataInitialized = true;
}

void AlgorithmicDetector::update() {
    if (!rawDataInitialized) return;
    std::vector<takheliped::Point> points = detector.forward(latestBottomScan, latestTopScan, latestOdometry);
    std::vector<geometry_msgs::Point> geometries = std::vector<geometry_msgs::Point>(points.size());
    for (int i = 0; i < points.size(); i++) geometries[i] = pointToGeometry(points[i]);

    takheliped::detection detection;
    detection.detection = geometries;
    detectionData.publish(detection);
}

AlgorithmicDetector::AlgorithmicDetector(int frequencyInit, int frequencyMax, float uncertaintyMax, float uncertaintyMin, float uncertaintyInc, float clusterThreshold, float distanceLevel, float legSizeMin, float legSizeMax, float chestSizeMin, float chestSizeMax, float legsDistanceMin, float legsDistanceMax, bool logging): detector(takheliped::AlgorithmicDetector(logging, frequencyInit, frequencyMax, uncertaintyMax, uncertaintyMin, uncertaintyInc, clusterThreshold, distanceLevel, legSizeMin, legSizeMax, chestSizeMin, chestSizeMax, legsDistanceMin, legsDistanceMax)) {
    rawData = handle.subscribe(RAW_DATA_TOPIC, 1, &AlgorithmicDetector::rawDataCallback, this);
    detectionData = handle.advertise<takheliped::detection>(ALGORITHMIC_DETECTOR_TOPIC, 1);

    ros::Rate rate(HEARTBEAT_RATE);
    while (ros::ok()) {
        ros::spinOnce();
        update();
        rate.sleep();
    }
}


int main(int argc, char **argv) {
    loadArgumentsForNode(argc, argv, ALGORITHMIC_DETECTOR);
    AlgorithmicDetector bsObject(FREQUENCY_INIT, FREQUENCY_MAX, UNCERTAINTY_MAX, UNCERTAINTY_MIN, UNCERTAINTY_INC, CLUSTER_THRESHOLD, DISTANCE_LEVEL, LEG_SIZE_MIN, LEG_SIZE_MAX, CHEST_SIZE_MIN, CHEST_SIZE_MAX, LEGS_DISTANCE_MIN, LEGS_DISTANCE_MAX, ALGORITHMIC_DETECOR_VERBOSE);
    ros::spin();
    return 0;
}
