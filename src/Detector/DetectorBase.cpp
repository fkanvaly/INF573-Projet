#include "DetectorBase.h"
#include "MotionDetector.h"
#include "FaceDetector.h"
#include "PeopleDetector.h"
#include "CarDetector.h"


DetectorBase::~DetectorBase()
{
}

DetectorBase* CreateDetector(DetectorsName detectorType, const config_t& config, int minSize)
{
    DetectorBase* detector = nullptr;

    switch (detectorType)
    {
    case DetectorsName::Motion_MOG2:
        detector = new MotionDetector(BGS::BGS_Type::MOG2, minSize);
        break;
    case DetectorsName::Motion_GSOC:
        detector = new MotionDetector(BGS::BGS_Type::GSOC, minSize);
        break;
    case DetectorsName::People:
        detector = new PeopleDetector(minSize);
        break;
    case DetectorsName::Face:
        detector = new FaceDetector(minSize);
        break;
    case DetectorsName::Car:
        detector = new CarDetector(minSize);
        break;
    default:
        break;
    }

    if (!detector->Initialize(config))
    {
        delete detector;
        detector = nullptr;
    }
    return detector;
}
