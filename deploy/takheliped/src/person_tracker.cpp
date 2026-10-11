#include "person_tracker.hpp"

#include "geometry_msgs/Point.h"

#include "takheliped/detection.h"

#include "main.hpp"

void PersonTracker::detectionCallback(const takheliped::detection::ConstPtr& data) {
    latestDetection = std::vector(data->detection.size(), takheliped::Point());
    for (int i = 0; i < latestDetection.size(); i++)
        latestDetection[i] = takheliped::Point(data->detection[i].x, data->detection[i].y);
    detectionInitialized = true;
}

void PersonTracker::update() {
    if (!detectionInitialized) return;

    if (personTracked) {
        if (latestDetection.size() > 0) person = track(policy, person, latestDetection);
        else personTracked = false;
    } else {
        if (latestDetection.size() > 0) {
            person = track(policy, latestDetection);
            personTracked = true;
        }
    }

    geometry_msgs::Point tracked;
    tracked.x = person.x;
    tracked.y = person.y;
    personPosition.publish(tracked);
}

PersonTracker::PersonTracker(TrackerPolicy tracker, std::string input): policy(tracker), inputTopic(input) {
    detection = handle.subscribe(inputTopic, 1, &PersonTracker::detectionCallback, this);
    personPosition = handle.advertise<geometry_msgs::Point>(FOLLOW_ME_BEHAVIOR_TOPIC, 1);

    ros::Rate rate(HEARTBEAT_RATE);
    while (ros::ok()) {
        ros::spinOnce();
        update();
        rate.sleep();
    }
}


int main(int argc, char **argv) {
    loadArgumentsForNode(argc, argv, PERSON_TRACKER);
    PersonTracker bsObject(TRACKING_POLICY, DETECTION_INPUT_TOPIC);
    ros::spin();
    return 0;
}
