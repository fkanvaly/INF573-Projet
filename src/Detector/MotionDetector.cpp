#include "MotionDetector.h"

///
/// \brief MotionDetector::MotionDetector
/// \param algType
/// \param gray
///
MotionDetector::MotionDetector(BGS::BGS_Type bgs, int minSize)
    :
      DetectorBase(minSize),
      bgsType(bgs)
{
	myBGS = std::make_unique<BGS>(bgs);
}

///
/// \brief MotionDetector::~MotionDetector
///
MotionDetector::~MotionDetector(void)
{
}

///
/// \brief MotionDetector::Init
/// \param config
/// \return
///
bool MotionDetector::Initialize(const config_t& config)
{
    return myBGS->Initialize(config);
}

///
/// \brief MotionDetector::DetectContour
///
void MotionDetector::DetectContour()
{
	regions.clear();
	Mat canny_output;
	Canny(fgMask, canny_output, 100, 200 );
	vector<vector<Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(fgMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	vector<vector<Point> > contours_poly;
	
	for( size_t i = 0; i < contours.size(); i++ )
	{
		cv::Rect br = cv::boundingRect(contours[i]);

		if (br.width >=  minObjSize &&
			br.height >= minObjSize)
		{
			regions.push_back(CRegion(br));
		}
	}
}

///
/// \brief MotionDetector::Detect
/// \param gray
///
void MotionDetector::Detect(cv::UMat& gray)
{
    myBGS->Substract(gray, fgMask);
	DetectContour();
}
