#include "PeopleDetector.h"

///
/// \brief PeopleDetector::PeopleDetector
/// \param algType
/// \param gray
///
PeopleDetector::PeopleDetector(int minSize)
    :
      DetectorBase(minSize)
{
}

///
/// \brief PeopleDetector::~PeopleDetector
///
PeopleDetector::~PeopleDetector(void)
{
}

///
/// \brief PeopleDetector::Init
/// \param config
/// \return
///
bool PeopleDetector::Initialize(const config_t& config)
{
	m_hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    return true;
}

///
/// \brief PeopleDetector::Detect
/// \param gray
///
void PeopleDetector::Detect(cv::UMat& gray)
{
    vector<Rect> found;
    vector<double> weights;
    m_hog.detectMultiScale(gray, found, weights, 0, Size(8,8), Size(32,32), 1.05);

    regions.clear();
    for (auto rect : found)
    {
        regions.push_back(rect);
    }
}
