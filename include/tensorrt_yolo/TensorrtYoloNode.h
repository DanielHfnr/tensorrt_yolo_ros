#ifndef TENSORRTYOLONODE_H
#define TENSORRTYOLONODE_H

#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "tensorrt_yolo/TensorrtYolo.h"

class TensorrtYoloNode
{
public:
    TensorrtYoloNode(ros::NodeHandle nh, ros::NodeHandle nh_private);
    TensorrtYoloNode() = delete;
    ~TensorrtYoloNode();

    void Cycle();

private:
    void Init();
    void InitRosSubscribers();
    void InitRosPublishers();

    void ImageCallback(const sensor_msgs::ImageConstPtr& image);

private:
    ros::NodeHandle nh_{};
    ros::NodeHandle nh_private_{};
    image_transport::ImageTransport it_;

    std::string image_topic_{};
    std::string bounding_boxes_topic_{};

    image_transport::Subscriber image_sub_{};
    ros::Publisher bboxes_pub_{};

    cv::Mat image_{};
    std::string frame_id_{};

    TensorrtYolo neural_net_{}; //!< Tensorrt Neural Net implementation of Yolo
};

#endif