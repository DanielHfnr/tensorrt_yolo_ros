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

    //!
    //! \brief Initialized the TensorRT engine.
    //!
    //! \param onnx_model_path Filepath to the onnx input model
    //! \param precision Precsion for the TRT engine (FP32, FP16 or INT8)
    //! \param device Device where the engine should be build on (GPU, DLA etc.)
    //! \param allow_gpu_fallback Allow GPU fallback if layer is not supported on DLA etc.
    //!
    //! \return True if initialization was successfull, False if not.
    //!
    bool Init(std::string onnx_model_path, PrecisionType precision, DeviceType device, bool allow_gpu_fallback);

    //!
    //! \brief Runs inference on a give input image and detects object in 2D.
    //!
    //! \param image Opencv Mat image where objects should be detected.
    //!
    //! \return True if inference was successfull, False if not.
    //!
    bool Detect(cv::Mat image);

    //!
    //! \brief Returns the current number of detections.
    //!
    //! \return Number of bounding boxes detected in frame.
    //!
    uint32_t GetNumDetections() const;

    //!
    //! \brief Returns the current detections (Array of bounding boxes)
    //!
    //! \return Pointer to the Bounding Box Array
    //!
    std::shared_ptr<ObjectBoundingBox[]> GetDetections() const;

private:
    //!
    //! \brief Preprocess the input image. Normalization and byte layout is changed from HWC to CHW
    //!
    //! \param image Opencv Mat image where objects should be detected.
    //!
    //! \return True if preprocessing was successfull, False if not.
    //!
    bool PreprocessInputs(cv::Mat& image);

    //!
    //! \brief Postprocesses the network results. Get 2D bounding boxes from network outputs and scales them to the
    //! original input image size.
    //!
    //! \param in_image_width Original input image width
    //! \param in_image_height Original input image height
    //!
    //! \return True if postprocessing was successfull, False if not.
    //!
    bool PostporcessOutputs(const uint32_t in_image_width, const uint32_t in_image_height);

    //!
    //! \brief Gets the input image width of the neural network in pixel.
    //!
    uint32_t GetNetworkInputWidth();

    //!
    //! \brief Gets the input image height of the neural network in pixel.
    //!
    uint32_t GetNetworkInputHeight();

private:
    std::string yolo_onnx_model_filepath_{}; //!< Filepath to ONNX model
    uint32_t num_detections_{0};             //!< current number of detections
    uint32_t max_detections_{0};             //!< maximum number of bounding boxes that can be detected per frame
    std::shared_ptr<ObjectBoundingBox[]> bounding_boxes_{}; //!< list of 2d bounding box detections
};

#endif