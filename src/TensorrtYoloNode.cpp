#include "tensorrt_yolo/TensorrtYoloNode.h"

#include "tensorrt_yolo/BoundingBox2D.h"


TensorrtYoloNode::TensorrtYoloNode(ros::NodeHandle nh, ros::NodeHandle nh_private)
    : nh_(nh)
    , nh_private_(nh_private)
{

}


TensorrtYoloNode::~TensorrtYoloNode()
{

}


void TensorrtYoloNode::Cycle()
{

}