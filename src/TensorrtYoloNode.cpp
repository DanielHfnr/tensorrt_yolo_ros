#include "tensorrt_yolo/TensorrtYoloNode.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ros/console.h>
#include <vision_msgs/BoundingBox2DArray.h>

TensorrtYoloNode::TensorrtYoloNode(ros::NodeHandle nh, ros::NodeHandle nh_private)
    : nh_(nh)
    , nh_private_(nh_private)
    , it_(nh)
{
    Init();
}

TensorrtYoloNode::~TensorrtYoloNode() {}

void TensorrtYoloNode::Init()
{
    ROS_INFO("Intializing ROS params...");

    std::string yolo_onnx_model_filepath{};
    std::string class_labels_filepath{};

    std::string precision{};
    std::string device{};
    bool allow_gpu_fallback{false};

    // Init ROS params
    nh_private_.getParam("yolo_onnx_model", yolo_onnx_model_filepath);
    nh_private_.getParam("class_labels_file", class_labels_filepath);

    nh_private_.getParam("precision", precision);
    nh_private_.getParam("device", device);
    nh_private_.getParam("allow_gpu_fallback", allow_gpu_fallback);

    nh_private_.getParam("image_topic_in", image_topic_in_);
    nh_private_.getParam("image_topic_out", image_topic_out_);
    nh_private_.getParam("bounding_boxes_topic_out", bounding_boxes_topic_);
    nh_private_.getParam("show_output_image", show_output_image_);

    InitRosSubscribers();
    InitRosPublishers();

    ROS_INFO("Intializing neural network...");

    neural_net_.Init(yolo_onnx_model_filepath, class_labels_filepath, precisionTypeFromStr(precision),
        deviceTypeFromStr(device), true);
}

void TensorrtYoloNode::InitRosPublishers()
{
    ROS_INFO("Intializing ROS publishers...");
    bboxes_pub_ = nh_private_.advertise<vision_msgs::BoundingBox2DArray>(bounding_boxes_topic_, 10);
    image_pub_ = it_.advertise(image_topic_out_, 1);
}

void TensorrtYoloNode::InitRosSubscribers()
{
    ROS_INFO("Intializing ROS subscribers...");
    image_sub_ = it_.subscribe(image_topic_in_, 1, &TensorrtYoloNode::ImageCallback, this);
}

void TensorrtYoloNode::ImageCallback(const sensor_msgs::ImageConstPtr& image)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        frame_id_ = image->header.frame_id;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    image_ = cv_ptr->image;
}

void TensorrtYoloNode::Cycle()
{
    cv::Mat output_image = image_.clone();

    bool inference_success = neural_net_.Detect(image_);

    if (!inference_success)
    {
        ROS_INFO_STREAM("Failed to run inference...");
    }

    uint32_t num_detections = neural_net_.GetNumDetections();
    auto detections = neural_net_.GetDetections();
    ROS_INFO_STREAM("Num detections: " << std::to_string(num_detections));

    vision_msgs::BoundingBox2DArray bounding_boxes;
    bounding_boxes.header.stamp = ros::Time::now();
    bounding_boxes.header.frame_id = frame_id_;

    // Loop through all detection and convert to ROS message
    for (uint32_t i = 0; i < num_detections; ++i)
    {
        // Prepare output bouding box and convert to ROS format
        vision_msgs::BoundingBox2D box_2d;

        float center_x, center_y;
        detections[i].Center(&center_x, &center_y);
        float width = detections[i].Width();
        float height = detections[i].Height();

        box_2d.center.x = center_x;
        box_2d.center.y = center_y;
        box_2d.size_x = width;
        box_2d.size_y = height;

        bounding_boxes.boxes.push_back(box_2d);

        // Render bounding boxes in image
        cv::rectangle(output_image, cv::Rect(detections[i].Left, detections[i].Top, width, height), (0, 0, 255), 2);
    }

    // Debug purposes only. Show rendered image in opencv window
    if (show_output_image_ && !output_image.empty())
    {
        cv::imshow("object_detection", output_image);
        cv::waitKey(3);
    }

    // Publish the bounding boxes
    bboxes_pub_.publish(bounding_boxes);

    // Publish rendere image with bounding boxes
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_image).toImageMsg();
    image_pub_.publish(image_msg);
}