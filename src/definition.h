
#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <assert.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>

using namespace cv;
using namespace std;

typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define El_t CV_32F
#define Mat_t CV_32FC

typedef std::multimap<std::string, std::string> config_t;
typedef std::chrono::time_point<std::chrono::system_clock> Time_t;

/**
 * @brief 
 * Use to wether use bbox or rotated bbox and also
 * for conversion between them
 */
class CRegion
{
public:
    CRegion()
        : m_type(""), m_confidence(-1)
    {
    }

    CRegion(const cv::Rect& rect)
        : m_brect(rect)
    {
        B2RRect();
    }

    CRegion(const cv::RotatedRect& rrect)
        : m_rrect(rrect)
    {
        R2BRect();
    }

    CRegion(const cv::Rect& rect, const std::string& type, float confidence)
        : m_brect(rect), m_type(type), m_confidence(confidence)
    {
        B2RRect();
    }

    cv::RotatedRect m_rrect;
    cv::Rect m_brect;

    std::string m_type;
    float m_confidence = -1;

	mutable cv::Mat m_hist;

private:
    cv::Rect R2BRect()
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    cv::RotatedRect B2RRect()
    {
        m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
        return m_rrect;
    }
};

typedef std::vector<CRegion> regions_t;

enum DetectorsName
{
    Motion_MOG2,
    Motion_GSOC,
    People,
    Face, 
    Car
};

enum DistType
{
    DistJaccard,   // Intersection over Union, IoU, [0, 1]
	DistsCount
};

enum trackerType
{BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT};


struct alert_obj
{
    size_t m_ID;
    Mat m_roi;
};


//********************************************************************************
// COLOR INTERPOLATION
//*******************************************************************
//  https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}