#pragma once

#include "definition.h"

class DetectorBase
{
protected:
    int minObjSize;
    regions_t regions;

public:
    DetectorBase(int minSize){
        minObjSize = minSize;
    };
    virtual ~DetectorBase(void);

    virtual bool Initialize(const config_t& conf)=0;
    virtual void Detect(UMat& frame)=0;

    void setMinObjSize(int minSize){
        minObjSize = minSize;
    }
    const regions_t GetDetection() const{
        return regions;
    }

};

DetectorBase* CreateDetector(DetectorsName detectorType, const config_t& config, int minSize);
