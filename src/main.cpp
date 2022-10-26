#include <ros/ros.h>

#include "tensorrt_yolo/TensorrtYoloNode.h"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "tensorrt_yolo_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    ros::Rate rate(30); // ROS Rate at 30Hz

    TensorrtYoloNode tensorrt_yolo(nh, nh_private);

    while (ros::ok()) {
        tensorrt_yolo.Cycle();
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}