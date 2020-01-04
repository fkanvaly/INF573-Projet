#include "BGS.h"
#include <tuple>

BGS::BGS(BGS_Type bgs)
    :bgsType(bgs)
{}

BGS::~BGS()
{}

bool BGS::Initialize(const config_t& conf)
{
    bool failed=false;
    switch (bgsType)
    {
    case MOG2:
    {
        std::vector<std::string> paramsConf = { "history", "varThreshold", "detectShadows" };
        auto params = std::make_tuple(500, 16, 1);

        for (size_t i = 0; i < paramsConf.size(); ++i)
        {
            auto config = conf.find(paramsConf[i]);
            if (config != conf.end())
            {
                std::stringstream ss(config->second);

                switch (i)
                {
                case 0:
                    ss >> std::get<0>(params);
                    break;
                case 1:
                    ss >> std::get<1>(params);
                    break;
                case 2:
                    ss >> std::get<2>(params);
                    break;
                }
            }
        }

        myBGS = cv::createBackgroundSubtractorMOG2(std::get<0>(params), 
                                                   std::get<1>(params), 
                                                   std::get<2>(params) != 0)
                                                   .dynamicCast<cv::BackgroundSubtractor>();
        break;
    }
    case GSOC:
    {
        myBGS = cv::bgsegm::createBackgroundSubtractorGSOC();
        break;
    }
    default:
        failed=true;
        break;
    }
    return !failed;
}

void BGS::Substract(const UMat& image, UMat& foreground)
{
    if (foreground.channels() == 3){
        cvtColor(foreground, foreground, COLOR_BGR2GRAY);
    }
    if (image.channels() == 3){
        cvtColor(image, image, COLOR_BGR2GRAY);
    }

    myBGS->apply(image, foreground);
}