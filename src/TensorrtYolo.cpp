#include "tensorrt_yolo/TensorrtYolo.h"

#include <cuda.h>
#include <cuda_runtime.h>

TensorrtYolo::TensorrtYolo() {}

TensorrtYolo::~TensorrtYolo() {}

bool TensorrtYolo::Init(
    std::string onnx_model_path, PrecisionType precision, DeviceType device, bool allow_gpu_fallback)
{
    yolo_onnx_model_filepath_ = onnx_model_path;

    // General model loading
    if (!LoadNetwork(onnx_model_path, precision, device, allow_gpu_fallback))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to load network...");
        return false;
    }

    // Read max number of detections from detected classes input which has shape batchSize x MaxBoundingBoxes
    // Therefore .d[1]. Could be done with other outputs as well
    nvinfer1::Dims det_classes_dims = GetOutputDims("det_classes");

    if (det_classes_dims.nbDims >= 2)
    {
        max_detections_ = det_classes_dims.d[1];
        gLogger.log(nvinfer1::ILogger::Severity::kVERBOSE,
            ("Max number of detections: " + std::to_string(max_detections_)).c_str());
    }
    else
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to fetch max number of detections from ONNX model...");
        return false;
    }

    // Allocate detections with the size of max detections
    bounding_boxes_ = std::shared_ptr<ObjectBoundingBox[]>(new ObjectBoundingBox[max_detections_]);

    return true;
}

bool TensorrtYolo::Detect(cv::Mat image)
{
    num_detections_ = 0;

    const int input_image_width = image.cols;
    const int input_image_height = image.rows;

    if (image.empty())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kWARNING, "Input image empty...");
        return false;
    }

    if (!PreprocessInputs(image))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to preprocess network inputs...");
        return false;
    }

    if (!ProcessNetwork())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to run inference...");
        return false;
    }

    if (!PostporcessOutputs(input_image_width, input_image_height))
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to postprocess network outputs...");
        return false;
    }

    return true;
}

bool TensorrtYolo::PreprocessInputs(cv::Mat& image)
{
    // Resize image to have the same size as needed by the neural network
    cv::resize(image, image, cv::Size(GetNetworkInputWidth(), GetNetworkInputHeight()), 0, 0, cv::INTER_LINEAR);
    // Opencv Mat is BGR by default. Image input to network is RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // Total size in bytes to copy to GPU
    const size_t size = image.channels() * image.total() * sizeof(float);
    // Get pointer to mapped memory for input images
    float* blob = static_cast<float*>(inputs_["images"].CPU);
    // Normalize input pixels and transpose image from opencv NHWC to NCHW
    for (size_t c = 0; c < image.channels(); c++)
    {
        for (size_t h = 0; h < image.rows; h++)
        {
            for (size_t w = 0; w < image.cols; w++)
            {
                blob[c * image.cols * image.rows + h * image.cols + w] = (float) image.at<cv::Vec3b>(h, w)[c] / 255.0;
            }
        }
    }

    return true;
}

bool TensorrtYolo::PostporcessOutputs(const uint32_t in_image_width, const uint32_t in_image_height)
{
    // Reset object bounding boxes
    std::memset(bounding_boxes_.get(), 0, max_detections_ * sizeof(ObjectBoundingBox));

    int32_t* num_dets = static_cast<int32_t*>(outputs_["num_dets"].CPU);
    int32_t* detected_classes = static_cast<int32_t*>(outputs_["det_classes"].CPU);
    float* detected_scores = static_cast<float*>(outputs_["det_scores"].CPU);
    float* detected_boxes = static_cast<float*>(outputs_["det_boxes"].CPU);

    num_detections_ = *num_dets;

    const float image_scale_factor_x = static_cast<float>(in_image_width) / static_cast<float>(GetNetworkInputWidth());
    const float image_scale_factor_y
        = static_cast<float>(in_image_height) / static_cast<float>(GetNetworkInputHeight());

    for (int i = 0; i < num_detections_; i++)
    {
        bounding_boxes_[i].Instance = i;
        bounding_boxes_[i].ClassID = detected_classes[i];
        bounding_boxes_[i].Confidence = detected_scores[i];

        // Out of bounds check for bounding boxes
        float left = detected_boxes[i * 4] * image_scale_factor_x;
        float top = detected_boxes[i * 4 + 1] * image_scale_factor_y;
        float right = detected_boxes[i * 4 + 2] * image_scale_factor_x;
        float bottom = detected_boxes[i * 4 + 3] * image_scale_factor_y;
        left = std::max(std::min(left, static_cast<float>(in_image_width - 1)), 0.f);
        top = std::max(std::min(top, static_cast<float>(in_image_height - 1)), 0.f);
        right = std::max(std::min(right, static_cast<float>(in_image_width - 1)), 0.f);
        bottom = std::max(std::min(bottom, static_cast<float>(in_image_width - 1)), 0.f);

        // Actual bounding box coordinates
        bounding_boxes_[i].Left = left;
        bounding_boxes_[i].Top = top;
        bounding_boxes_[i].Right = right;
        bounding_boxes_[i].Bottom = bottom;
    }

    return true;
}

uint32_t TensorrtYolo::GetNetworkInputWidth()
{
    nvinfer1::Dims dims = GetInputDims("images");
    return dims.d[3];
}

uint32_t TensorrtYolo::GetNetworkInputHeight()
{
    nvinfer1::Dims dims = GetInputDims("images");
    return dims.d[2];
}

uint32_t TensorrtYolo::GetNumDetections() const
{
    return num_detections_;
}

std::shared_ptr<ObjectBoundingBox[]> TensorrtYolo::GetDetections() const
{
    return bounding_boxes_;
}