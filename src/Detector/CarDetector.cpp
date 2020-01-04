#include "CarDetector.h"

CarDetector::CarDetector(int minSize)
    :
      DetectorBase(minSize)
{
}

CarDetector::~CarDetector(void)
{
}

bool CarDetector::Initialize(const config_t& config)
{
	auto cascadeFileName = config.find("cascadeFileName");
    if (cascadeFileName != config.end() &&
            (!m_cascade.load(cascadeFileName->second) || m_cascade.empty()))
    {
        std::cerr << "Cascade " << cascadeFileName->second << " not opened!" << std::endl;
        return false;
    }
    return true;
}

void CarDetector::Detect(cv::UMat& gray)
{
    bool findLargestObject = false;
    bool filterRects = true;
    std::vector<cv::Rect> carRects;
    m_cascade.detectMultiScale(gray,
                             carRects,
                             1.1, 1 );
    regions.clear();
    for (auto rect : carRects)
    {
        regions.push_back(rect);
    }
}
