#pragma once

#include "DetectorBase.h"
#include "BGS.h"

///
/// \brief The FaceDetector class
///
class FaceDetector : public DetectorBase
{
public:
    FaceDetector(int minSize);
    ~FaceDetector(void);

    bool Initialize(const config_t& config);

    void Detect(cv::UMat& gray);

private:
    cv::CascadeClassifier m_cascade;
};



