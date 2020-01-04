#pragma once
#include "definition.h"
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/kalman_filters.hpp>

///
/// \brief The TKalmanFilter class
///
class TKalmanFilter
{
public:
    TKalmanFilter(track_t deltaTime = 0.2, track_t accelNoiseMag = 0.5);
    ~TKalmanFilter() = default;

    cv::Rect GetRectPrediction();
    cv::Rect Update(cv::Rect rect, bool dataCorrect);

	cv::Vec<track_t, 2> GetVelocity() const;

private:
    cv::KalmanFilter m_linearKalman;
    std::deque<Point_t> m_initialPoints;
    std::deque<cv::Rect> m_initialRects;
    static const size_t MIN_INIT_VALS = 4;

    Point_t m_lastPointResult;
    cv::Rect_<track_t> m_lastRectResult;
    cv::Rect_<track_t> m_lastRect;

    bool m_initialized = false;
    track_t m_deltaTime = 0.2f;
    track_t m_deltaTimeMin = 0.2f;
    track_t m_deltaTimeMax = 2 * 0.2f;
    track_t m_lastDist = 0;
    track_t m_deltaStep = 0;
    static const int m_deltaStepsCount = 20;
    track_t m_accelNoiseMag = 0.5f;

    void CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0);
};

//---------------------------------------------------------------------------
///
/// \brief sqr
/// \param val
/// \return
///
template<class T> inline
T sqr(T val)
{
    return val * val;
}

///
/// \brief get_lin_regress_params
/// \param in_data
/// \param start_pos
/// \param in_data_size
/// \param kx
/// \param bx
/// \param ky
/// \param by
///
template<typename T, typename CONT>
void get_lin_regress_params(
        const CONT& in_data,
        size_t start_pos,
        size_t in_data_size,
        T& kx, T& bx, T& ky, T& by)
{
    T m1(0.), m2(0.);
    T m3_x(0.), m4_x(0.);
    T m3_y(0.), m4_y(0.);

    const T el_count = static_cast<T>(in_data_size - start_pos);
    for (size_t i = start_pos; i < in_data_size; ++i)
    {
        m1 += i;
        m2 += sqr(i);

        m3_x += in_data[i].x;
        m4_x += i * in_data[i].x;

        m3_y += in_data[i].y;
        m4_y += i * in_data[i].y;
    }
    T det_1 = 1 / (el_count * m2 - sqr(m1));

    m1 *= -1;

    kx = det_1 * (m1 * m3_x + el_count * m4_x);
    bx = det_1 * (m2 * m3_x + m1 * m4_x);

    ky = det_1 * (m1 * m3_y + el_count * m4_y);
    by = det_1 * (m2 * m3_y + m1 * m4_y);
}
