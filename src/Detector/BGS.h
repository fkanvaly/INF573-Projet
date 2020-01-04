#pragma once

#include "definition.h"
#include <opencv2/bgsegm.hpp>

/**
 * @brief 
 * Background substraction class proposing 
 * OpenCV incorporeted methds
 */
class BGS
{
public:
	enum BGS_Type
	{
        MOG2,
        GSOC
	};

    BGS(BGS_Type bgs);

	~BGS();

    bool Initialize(const config_t& config);

    void Substract(const UMat& image, UMat& foreground);
	
	int m_channels;
	BGS_Type bgsType;

private:
	Ptr<cv::BackgroundSubtractor> myBGS;
};

