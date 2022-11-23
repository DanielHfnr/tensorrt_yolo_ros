#ifndef OBJECTBOUNDINGBOX_H
#define OBJECTBOUNDINGBOX_H

#include <cmath>
#include <cstdint>

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

#endif // OBJECTBOUNDINGBOX_H