#ifndef TENSORRTYOLO_H
#define TENSORRTYOLO_H

#include <tensorrt_base/TensorrtBase.h>

#include <math.h>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

class TensorrtYolo : public TensorrtBase
{
public:
    // TODO: Move that into separate class for readability
    struct ObjectBoundingBox
    {
        // Object Info
        uint32_t Instance; //!<  Index of this unique object instance
        uint32_t ClassID;  //!<  Class index of the detected object.
        float Confidence;  //!<  Confidence value of the detected object.

        // Bounding Box Coordinates
        float Left;   //!<  Left bounding box coordinate (in pixels)
        float Right;  //!<  Right bounding box coordinate (in pixels)
        float Top;    //!<  Top bounding box cooridnate (in pixels)
        float Bottom; //!<  Bottom bounding box coordinate (in pixels)

        //!
        //! \brief Calculate the width of the object
        //!
        inline float Width() const
        {
            return Right - Left;
        }

        //!
        //! \brief Calculate the height of the object
        //!
        inline float Height() const
        {
            return Bottom - Top;
        }

        //!
        //! \brief Calculate the area of the object
        //!
        inline float Area() const
        {
            return Width() * Height();
        }

        //!
        //! \brief Calculate the width of the bounding box
        //!
        static inline float Width(float x1, float x2)
        {
            return x2 - x1;
        }

        //!
        //! \brief Calculate the height of the bounding box
        //!
        static inline float Height(float y1, float y2)
        {
            return y2 - y1;
        }

        //!
        //! \brief Calculate the area of the bounding box
        //!
        static inline float Area(float x1, float y1, float x2, float y2)
        {
            return Width(x1, x2) * Height(y1, y2);
        }

        //!
        //! \brief Return the center of the object
        //!
        inline void Center(float* x, float* y) const
        {
            if (x)
                *x = Left + Width() * 0.5f;
            if (y)
                *y = Top + Height() * 0.5f;
        }

        //!
        //! \brief Return true if the coordinate is inside the bounding box
        //!
        inline bool Contains(float x, float y) const
        {
            return x >= Left && x <= Right && y >= Top && y <= Bottom;
        }

        //!
        //! \brief  Return true if the bounding boxes intersect and exceeds area % threshold
        //!
        inline bool Intersects(const ObjectBoundingBox& det, float areaThreshold = 0.0f) const
        {
            return (IntersectionArea(det) / fmaxf(Area(), det.Area()) > areaThreshold);
        }

        //!
        //! \brief  Return true if the bounding boxes intersect and exceeds area % threshold
        //!
        inline bool Intersects(float x1, float y1, float x2, float y2, float areaThreshold = 0.0f) const
        {
            return (IntersectionArea(x1, y1, x2, y2) / fmaxf(Area(), Area(x1, y1, x2, y2)) > areaThreshold);
        }

        //!
        //! \brief  Return the area of the bounding box intersection
        //!
        inline float IntersectionArea(const ObjectBoundingBox& det) const
        {
            return IntersectionArea(det.Left, det.Top, det.Right, det.Bottom);
        }

        //!
        //! \brief  Return the area of the bounding box intersection
        //!
        inline float IntersectionArea(float x1, float y1, float x2, float y2) const
        {
            if (!Overlaps(x1, y1, x2, y2))
                return 0.0f;
            return (fminf(Right, x2) - fmaxf(Left, x1)) * (fminf(Bottom, y2) - fmaxf(Top, y1));
        }

        //!
        //! \brief  Return true if the bounding boxes overlap
        //!
        inline bool Overlaps(const ObjectBoundingBox& det) const
        {
            return !(det.Left > Right || det.Right < Left || det.Top > Bottom || det.Bottom < Top);
        }

        //!
        //! \brief  Return true if the bounding boxes overlap
        //!
        inline bool Overlaps(float x1, float y1, float x2, float y2) const
        {
            return !(x1 > Right || x2 < Left || y1 > Bottom || y2 < Top);
        }

        //!
        //! \brief  Expand the bounding box if they overlap (return true if so)
        //!
        inline bool Expand(float x1, float y1, float x2, float y2)
        {
            if (!Overlaps(x1, y1, x2, y2))
                return false;
            Left = fminf(x1, Left);
            Top = fminf(y1, Top);
            Right = fmaxf(x2, Right);
            Bottom = fmaxf(y2, Bottom);
            return true;
        }

        //!
        //! \brief  Expand the bounding box if they overlap (return true if so)
        //!
        inline bool Expand(const ObjectBoundingBox& det)
        {
            if (!Overlaps(det))
                return false;
            Left = fminf(det.Left, Left);
            Top = fminf(det.Top, Top);
            Right = fmaxf(det.Right, Right);
            Bottom = fmaxf(det.Bottom, Bottom);
            return true;
        }

        //!
        //! \brief  Reset all member variables to zero
        //!
        inline void Reset()
        {
            Instance = 0;
            ClassID = 0;
            Confidence = 0;
            Left = 0;
            Right = 0;
            Top = 0;
            Bottom = 0;
        }

        //!
        //! \brief  Default constructor
        //!
        inline ObjectBoundingBox()
        {
            Reset();
        }
    };

public:
    TensorrtYolo();
    ~TensorrtYolo();

    bool Init(std::string onnx_model_path, std::string class_labels_path, PrecisionType precision, DeviceType device,
        bool allow_gpu_fallback);

    bool Detect(cv::Mat image);

    uint32_t GetNumDetections() const
    {
        return num_detections_;
    }

    std::shared_ptr<ObjectBoundingBox[]> GetDetections() const
    {
        return bounding_boxes_;
    }

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