#include "Kalman.h"
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------
TKalmanFilter::TKalmanFilter(
        track_t deltaTime, // time increment (lower values makes target more "massive")
        track_t accelNoiseMag
        )
    :
      m_initialized(false),
      m_deltaTime(deltaTime),
      m_deltaTimeMin(deltaTime),
      m_deltaTimeMax(2 * deltaTime),
      m_lastDist(0),
      m_accelNoiseMag(accelNoiseMag)
{
    m_deltaStep = (m_deltaTimeMax - m_deltaTimeMin) / m_deltaStepsCount;
}


void TKalmanFilter::CreateLinear(cv::Rect_<track_t> rect0, Point_t rectv0)
{
    // We don't know acceleration, so, assume it to process noise.
    // But we can guess, the range of acceleration values thich can be achieved by tracked object.
    // Process noise. (standard deviation of acceleration: m/s^2)
    // shows, woh much target can accelerate.

    // 6 state variables (x, y, dx, dy, width, height), 4 measurements (x, y, width, height)
    m_linearKalman.init(8, 4, 0, El_t);
    // Transition cv::Matrix
    m_linearKalman.transitionMatrix = (cv::Mat_<track_t>(8, 8) <<
                                        1, 0, 0, 0, m_deltaTime, 0,           0,           0,
                                        0, 1, 0, 0, 0,           m_deltaTime, 0,           0,
                                        0, 0, 1, 0, 0,           0,           m_deltaTime, 0,
                                        0, 0, 0, 1, 0,           0,           0,           m_deltaTime,
                                        0, 0, 0, 0, 1,           0,           0,           0,
                                        0, 0, 0, 0, 0,           1,           0,           0,
                                        0, 0, 0, 0, 0,           0,           1,           0,
                                        0, 0, 0, 0, 0,           0,           0,           1);

    // init...
    m_linearKalman.statePre.at<track_t>(0) = rect0.x;      // x
    m_linearKalman.statePre.at<track_t>(1) = rect0.y;      // y
    m_linearKalman.statePre.at<track_t>(2) = rect0.width;  // width
    m_linearKalman.statePre.at<track_t>(3) = rect0.height; // height
    m_linearKalman.statePre.at<track_t>(4) = rectv0.x;     // dx
    m_linearKalman.statePre.at<track_t>(5) = rectv0.y;     // dy
    m_linearKalman.statePre.at<track_t>(6) = 0;            // dw
    m_linearKalman.statePre.at<track_t>(7) = 0;            // dh

    m_linearKalman.statePost.at<track_t>(0) = rect0.x;
    m_linearKalman.statePost.at<track_t>(1) = rect0.y;
    m_linearKalman.statePost.at<track_t>(2) = rect0.width;
    m_linearKalman.statePost.at<track_t>(3) = rect0.height;
    m_linearKalman.statePost.at<track_t>(4) = rectv0.x;
    m_linearKalman.statePost.at<track_t>(5) = rectv0.y;
    m_linearKalman.statePost.at<track_t>(6) = 0;
    m_linearKalman.statePost.at<track_t>(7) = 0;

    cv::setIdentity(m_linearKalman.measurementMatrix);

    track_t n1 = pow(m_deltaTime, 4.f) / 4.f;
    track_t n2 = pow(m_deltaTime, 3.f) / 2.f;
    track_t n3 = pow(m_deltaTime, 2.f);
    m_linearKalman.processNoiseCov = (cv::Mat_<track_t>(8, 8) <<
                                       n1, 0,  0,  0,  n2, 0,  0,  0,
                                       0,  n1, 0,  0,  0,  n2, 0,  0,
                                       0,  0,  n1, 0,  0,  0,  n2, 0,
                                       0,  0,  0,  n1, 0,  0,  0,  n2,
                                       n2, 0,  0,  0,  n3, 0,  0,  0,
                                       0,  n2, 0,  0,  0,  n3, 0,  0,
                                       0,  0,  n2, 0,  0,  0,  n3, 0,
                                       0,  0,  0,  n2, 0,  0,  0,  n3);

    m_linearKalman.processNoiseCov *= m_accelNoiseMag;

    cv::setIdentity(m_linearKalman.measurementNoiseCov, cv::Scalar::all(0.1));

    cv::setIdentity(m_linearKalman.errorCovPost, cv::Scalar::all(.1));

    m_initialized = true;
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::Update(cv::Rect rect, bool dataCorrect)
{
    if (!m_initialized)
    {
        if (m_initialRects.size() < MIN_INIT_VALS)
        {
            if (dataCorrect)
            {
                m_initialRects.push_back(rect);
                m_lastRectResult.x = static_cast<track_t>(rect.x);
                m_lastRectResult.y = static_cast<track_t>(rect.y);
                m_lastRectResult.width = static_cast<track_t>(rect.width);
                m_lastRectResult.height = static_cast<track_t>(rect.height);
            }
        }
        if (m_initialRects.size() == MIN_INIT_VALS)
        {
            std::vector<Point_t> initialPoints;
            Point_t averageSize(0, 0);
            for (const auto& r : m_initialRects)
            {
                initialPoints.emplace_back(static_cast<track_t>(r.x), static_cast<track_t>(r.y));
                averageSize.x += r.width;
                averageSize.y += r.height;
            }
            averageSize.x /= MIN_INIT_VALS;
            averageSize.y /= MIN_INIT_VALS;

            track_t kx = 0;
            track_t bx = 0;
            track_t ky = 0;
            track_t by = 0;
            get_lin_regress_params(initialPoints, 0, MIN_INIT_VALS, kx, bx, ky, by);
            cv::Rect_<track_t> rect0(kx * (MIN_INIT_VALS - 1) + bx, ky * (MIN_INIT_VALS - 1) + by, averageSize.x, averageSize.y);
            Point_t rectv0(kx, ky);

            CreateLinear(rect0, rectv0);
        }
    }

    if (m_initialized)
    {
        cv::Mat measurement(4, 1, Mat_t(1));
        if (!dataCorrect)
        {
            measurement.at<track_t>(0) = m_lastRectResult.x;  // update using prediction
            measurement.at<track_t>(1) = m_lastRectResult.y;
            measurement.at<track_t>(2) = m_lastRectResult.width;
            measurement.at<track_t>(3) = m_lastRectResult.height;
        }
        else
        {
            measurement.at<track_t>(0) = static_cast<track_t>(rect.x);  // update using measurements
            measurement.at<track_t>(1) = static_cast<track_t>(rect.y);
            measurement.at<track_t>(2) = static_cast<track_t>(rect.width);
            measurement.at<track_t>(3) = static_cast<track_t>(rect.height);
        }
        // Correction
        cv::Mat estimated;
        estimated = m_linearKalman.correct(measurement);

        m_lastRectResult.x = estimated.at<track_t>(0);   //update using measurements
        m_lastRectResult.y = estimated.at<track_t>(1);
        m_lastRectResult.width = estimated.at<track_t>(2);
        m_lastRectResult.height = estimated.at<track_t>(3);

        // Inertia correction
        track_t currDist = sqrtf(sqr(estimated.at<track_t>(0) - rect.x) + sqr(estimated.at<track_t>(1) - rect.y) + sqr(estimated.at<track_t>(2) - rect.width) + sqr(estimated.at<track_t>(3) - rect.height));
        if (currDist > m_lastDist)
        {
            m_deltaTime = std::min(m_deltaTime + m_deltaStep, m_deltaTimeMax);
        }
        else
        {
            m_deltaTime = std::max(m_deltaTime - m_deltaStep, m_deltaTimeMin);
        }
        m_lastDist = currDist;

        m_linearKalman.transitionMatrix.at<track_t>(0, 4) = m_deltaTime;
        m_linearKalman.transitionMatrix.at<track_t>(1, 5) = m_deltaTime;
        m_linearKalman.transitionMatrix.at<track_t>(2, 6) = m_deltaTime;
        m_linearKalman.transitionMatrix.at<track_t>(3, 7) = m_deltaTime;
    }
    else
    {
        if (dataCorrect)
        {
            m_lastRectResult.x = static_cast<track_t>(rect.x);
            m_lastRectResult.y = static_cast<track_t>(rect.y);
            m_lastRectResult.width = static_cast<track_t>(rect.width);
            m_lastRectResult.height = static_cast<track_t>(rect.height);
        }
    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}

//---------------------------------------------------------------------------
cv::Rect TKalmanFilter::GetRectPrediction()
{
    if (m_initialized)
    {
        cv::Mat prediction;
        prediction = m_linearKalman.predict();
        m_lastRectResult = cv::Rect_<track_t>(prediction.at<track_t>(0), prediction.at<track_t>(1), prediction.at<track_t>(2), prediction.at<track_t>(3));
    }
    else
    {

    }
    return cv::Rect(static_cast<int>(m_lastRectResult.x), static_cast<int>(m_lastRectResult.y), static_cast<int>(m_lastRectResult.width), static_cast<int>(m_lastRectResult.height));
}


//---------------------------------------------------------------------------
cv::Vec<track_t, 2> TKalmanFilter::GetVelocity() const
{
	cv::Vec<track_t, 2> res(0, 0);
    if (m_linearKalman.statePre.rows > 3)
    {
        int indX = 2;
        int indY = 3;
        if (m_linearKalman.statePre.rows > 4)
        {
            indX = 4;
            indY = 5;
        }
        res[0] = m_linearKalman.statePre.at<track_t>(indX);
        res[1] = m_linearKalman.statePre.at<track_t>(indY);
    }

	return res;
}

