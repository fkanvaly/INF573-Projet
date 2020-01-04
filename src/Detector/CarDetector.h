#pragma once

#include "DetectorBase.h"
#include "BGS.h"

/**
 * @brief 
 * Car detector using cascade classifier
 */
class CarDetector : public DetectorBase
{
public:
    CarDetector(int minSize);
    ~CarDetector(void);

    bool Initialize(const config_t& config);

    void Detect(cv::UMat& gray);

private:
    cv::CascadeClassifier m_cascade;
};



