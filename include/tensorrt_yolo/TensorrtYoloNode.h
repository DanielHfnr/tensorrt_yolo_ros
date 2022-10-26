#ifndef TENSORRTYOLONODE_H
#define TENSORRTYOLONODE_H

#include <ros/ros.h>

#include "tensorrt_yolo/TensorrtYolo.h"


class TensorrtYoloNode
{
public:
    TensorrtYoloNode(ros::NodeHandle nh, ros::NodeHandle nh_private);
    ~TensorrtYoloNode();

    void Cycle();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    TensorrtYolo neural_net_;
};

#endif