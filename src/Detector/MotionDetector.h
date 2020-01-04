#pragma once

#include "DetectorBase.h"
#include "BGS.h"

///
/// \brief The MotionDetector class
///
class MotionDetector : public DetectorBase
{
public:
    MotionDetector(BGS::BGS_Type bgs, int minSize);
    ~MotionDetector(void);

    bool Initialize(const config_t& config);

    void Detect(cv::UMat& gray);

    UMat GetFGMask() const{
        return fgMask;
    };

private:
    void DetectContour();

    std::unique_ptr<BGS> myBGS;
    cv::UMat fgMask;
    BGS::BGS_Type bgsType;
};



