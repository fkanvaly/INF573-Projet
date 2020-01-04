#pragma once

#include "DetectorBase.h"
#include "BGS.h"

///
/// \brief The FaceDetector class
///
class PeopleDetector : public DetectorBase
{
public:
    PeopleDetector(int minSize);
    ~PeopleDetector(void);

    bool Initialize(const config_t& config);

    void Detect(cv::UMat& gray);

private:
    HOGDescriptor m_hog;
};



