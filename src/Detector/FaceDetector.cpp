#include "FaceDetector.h"

///
/// \brief FaceDetector::FaceDetector
/// \param algType
/// \param gray
///
FaceDetector::FaceDetector(int minSize)
    :
      DetectorBase(minSize)
{
}

///
/// \brief FaceDetector::~FaceDetector
///
FaceDetector::~FaceDetector(void)
{
}

///
/// \brief FaceDetector::Init
/// \param config
/// \return
///
bool FaceDetector::Initialize(const config_t& config)
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

///
/// \brief FaceDetector::Detect
/// \param gray
///
void FaceDetector::Detect(cv::UMat& gray)
{
    bool findLargestObject = false;
    bool filterRects = true;
    std::vector<cv::Rect> faceRects;
    m_cascade.detectMultiScale(gray,
                             faceRects,
                             1.1,
                             (filterRects || findLargestObject) ? 3 : 0,
                             findLargestObject ? cv::CASCADE_FIND_BIGGEST_OBJECT : 0,
                             Size(minObjSize, minObjSize),
                             cv::Size(gray.cols / 2, gray.rows / 2));
    regions.clear();
    for (auto rect : faceRects)
    {
        regions.push_back(rect);
    }
}
