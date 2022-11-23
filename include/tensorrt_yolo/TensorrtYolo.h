#ifndef TENSORRTYOLO_H
#define TENSORRTYOLO_H

#include "ObjectBoundingBox.h"

#include <tensorrt_base/TensorrtBase.h>

#include <math.h>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

class TensorrtYolo : public TensorrtBase
{
public:
    TensorrtYolo();
    ~TensorrtYolo();

    bool Init(std::string onnx_model_path, std::string class_labels_path, PrecisionType precision, DeviceType device,
        bool allow_gpu_fallback);

    bool Detect(cv::Mat image);

    uint32_t GetNumDetections() const;

    std::shared_ptr<ObjectBoundingBox[]> GetDetections() const;

private:
    bool PreprocessInputs(cv::Mat& image);
    bool PostporcessOutputs(const uint32_t in_image_width, const uint32_t in_image_height);

    uint32_t GetNetworkInputWidth();
    uint32_t GetNetworkInputHeight();

private:
    std::string yolo_onnx_model_filepath_{}; //!< Filepath to ONNX model
    std::string class_labels_filepath_{};    //!< Filepath to txt file mapping class names to integers

    std::shared_ptr<ObjectBoundingBox[]> bounding_boxes_{}; //!< list of 2d bounding box detections
    uint32_t num_detections_{0};                            //!< current number of detections
    uint32_t max_detections_{0}; //!< maximum number of bounding boxes that can be detected per frame
};

#endif