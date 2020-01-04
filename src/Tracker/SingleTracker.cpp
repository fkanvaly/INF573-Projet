#include "SingleTracker.h"

/// \param region
/// \param deltaTime
/// \param accelNoiseMag
/// \param trackID
/// \param filterObjectSize
/// \param externalTrackerForLost
///
CTrack::CTrack(
        const CRegion& region,
        track_t deltaTime,
        track_t accelNoiseMag,
        size_t trackID,
        trackerType tracker_type)
    :
      m_trackID(trackID),
      m_tracker_type(tracker_type),
      m_skippedFrames(0),
      m_time(std::chrono::system_clock::now()), 
      m_lastRegion(region),
      m_predictionPoint(region.m_rrect.center),
      m_predictionRect(region.m_rrect),
      m_kalman(deltaTime, accelNoiseMag),
      m_isRec_estimation(true),
      m_outOfTheFrame(false),
      m_color((rand()%255,rand()%255,rand()%255))
{
    m_kalman.Update(region.m_brect, true);
    m_trace.push_back(m_predictionPoint, m_predictionPoint);
}

///
/// \brief CTrack::CalcDistJaccard
/// \param reg
/// \return
///
track_t CTrack::CalcDistJaccard(const CRegion& reg) const
{
    track_t intArea = static_cast<track_t>((reg.m_brect & m_lastRegion.m_brect).area());
    track_t unionArea = static_cast<track_t>(reg.m_brect.area() + m_lastRegion.m_brect.area() - intArea);

    return 1 - intArea / unionArea;
}


///
/// \brief CTrack::Update
/// \*param region
/// \param dataCorrect
/// \param max_trace_length
/// \param prevFrame
/// \param currFrame
/// \param trajLen
///
void CTrack::Update(
        const CRegion& region,
        bool dataCorrect,
        size_t max_trace_length,
        cv::UMat prevFrame,
        cv::UMat currFrame,
        int trajLen
        )
{
    if (m_isRec_estimation) // Kalman filter for object coordinates and size
    {
        RectUpdate(region, dataCorrect, prevFrame, currFrame);
    }
    else // Kalman filter only for object center
    {
        assert(false);
    }
    if (dataCorrect)
    {
        //std::cout << m_lastRegion.m_brect << " - " << region.m_brect << std::endl;

        m_lastRegion = region;
        m_trace.push_back(m_predictionPoint, region.m_rrect.center);

    }
    else
    {
        m_trace.push_back(m_predictionPoint);
    }

    if (m_trace.size() > max_trace_length)
    {
        m_trace.pop_front(m_trace.size() - max_trace_length);
    }
}

///
/// \brief CTrack::IsOutOfTheFrame
/// \return
///
bool CTrack::IsOutOfTheFrame() const
{
	return m_outOfTheFrame;
}

///
/// \brief GetLastRect
/// \return
///
cv::RotatedRect CTrack::GetLastRect() const
{
    if (m_isRec_estimation)
    {
        return m_predictionRect;
    }
    else
    {
        return cv::RotatedRect(cv::Point2f(m_predictionPoint.x, m_predictionPoint.y), m_predictionRect.size, m_predictionRect.angle);
    }
}

///
/// \brief CTrack::LastRegion
/// \return
///
const CRegion& CTrack::LastRegion() const
{
    return m_lastRegion;
}

///
/// \brief CTrack::ConstructObject
/// \return
///
TrackingObject CTrack::ConstructObject() const
{
    return TrackingObject(GetLastRect(), m_trackID, m_trace, m_time, m_color,IsOutOfTheFrame(),
                          m_lastRegion.m_type, m_lastRegion.m_confidence, m_kalman.GetVelocity());
}

///
/// \brief CTrack::SkippedFrames
/// \return
///
size_t CTrack::SkippedFrames() const
{
    return m_skippedFrames;
}

///
/// \brief CTrack::SkippedFrames
/// \return
///
size_t& CTrack::SkippedFrames()
{
    return m_skippedFrames;
}

///
/// \brief RectUpdate
/// \param region
/// \param dataCorrect
/// \param prevFrame
/// \param currFrame
///
void CTrack::RectUpdate(
        const CRegion& region,
        bool dataCorrect,
        cv::UMat prevFrame,
        cv::UMat currFrame
        )
{
    m_kalman.GetRectPrediction();

    bool recalcPrediction = true;

    auto Clamp = [](int& v, int& size, int hi) -> int
    {
        int res = 0;

        if (size < 2)
        {
            size = 2;
        }
        if (v < 0)
        {
            res = v;
            v = 0;
            return res;
        }
        else if (v + size > hi - 1)
        {
            v = hi - 1 - size;
            if (v < 0)
            {
                size += v;
                v = 0;
            }
            res = v;
            return res;
        }
        return res;
    };

    auto UpdateRRect = [&](cv::Rect prevRect, cv::Rect newRect)
    {
        m_predictionRect.center.x += newRect.x - prevRect.x;
        m_predictionRect.center.y += newRect.y - prevRect.y;
        m_predictionRect.size.width *= newRect.width / static_cast<float>(prevRect.width);
        m_predictionRect.size.height *= newRect.height / static_cast<float>(prevRect.height);
    };

    if (!dataCorrect)
    {
        cv::Rect brect = m_predictionRect.boundingRect();

        cv::Size roiSize(std::max(2 * brect.width, currFrame.cols / 4), std::max(2 * brect.height, currFrame.rows / 4));
        if (roiSize.width > currFrame.cols)
        {
            roiSize.width = currFrame.cols;
        }
        if (roiSize.height > currFrame.rows)
        {
            roiSize.height = currFrame.rows;
        }
        cv::Point roiTL(brect.x + brect.width / 2 - roiSize.width / 2, brect.y + brect.height / 2 - roiSize.height / 2);
        cv::Rect roiRect(roiTL, roiSize);
        Clamp(roiRect.x, roiRect.width, currFrame.cols);
        Clamp(roiRect.y, roiRect.height, currFrame.rows);

        bool inited = false;
        if (!m_tracker || m_tracker.empty())
        {
            create_tracker();

            cv::Rect2d lastRect(brect.x - roiRect.x, brect.y - roiRect.y, brect.width, brect.height);
            if (m_staticFrame.empty())
            {
                int dx = 1;//m_predictionRect.width / 8;
                int dy = 1;//m_predictionRect.height / 8;
                lastRect = cv::Rect2d(brect.x - roiRect.x - dx, brect.y - roiRect.y - dy, brect.width + 2 * dx, brect.height + 2 * dy);
            }
            else
            {
                lastRect = cv::Rect2d(m_staticRect.x - roiRect.x, m_staticRect.y - roiRect.y, m_staticRect.width, m_staticRect.height);
            }

            if (lastRect.x >= 0 &&
                    lastRect.y >= 0 &&
                    lastRect.x + lastRect.width < roiRect.width &&
                    lastRect.y + lastRect.height < roiRect.height &&
                    lastRect.area() > 0)
            {
                if (m_staticFrame.empty())
                {
                    m_tracker->init(prevFrame(roiRect), lastRect);
                }
                else
                {
                    m_tracker->init(m_staticFrame(roiRect), lastRect);
                }

                inited = true;
                m_outOfTheFrame = false;
            }
            else
            {
                m_tracker.release();
                m_outOfTheFrame = true;
            }
        }

        cv::Rect2d newRect;
        if (!inited && !m_tracker.empty() && m_tracker->update(currFrame(roiRect), newRect))
        {
            cv::Rect prect(cvRound(newRect.x) + roiRect.x, cvRound(newRect.y) + roiRect.y, cvRound(newRect.width), cvRound(newRect.height));

            UpdateRRect(brect, m_kalman.Update(prect, true));

            recalcPrediction = false;
        }
    }
    else
    {
        if (m_tracker && !m_tracker.empty())
        {
            m_tracker.release();
        }
    }

    if (recalcPrediction)
    {
        UpdateRRect(m_predictionRect.boundingRect(), m_kalman.Update(region.m_brect, dataCorrect));
    }

    cv::Rect brect = m_predictionRect.boundingRect();
    int dx = Clamp(brect.x, brect.width, currFrame.cols);
    int dy = Clamp(brect.y, brect.height, currFrame.rows);
    m_predictionRect.center.x += dx;
    m_predictionRect.center.y += dy;

    m_outOfTheFrame = (dx != 0) || (dy != 0);

    m_predictionPoint = m_predictionRect.center;
}

void CTrack::create_tracker(){
    //Tracker
    switch (m_tracker_type)
    {
    case BOOSTING:
        m_tracker = TrackerBoosting::create();
        break;
    case MIL:
        m_tracker = TrackerMIL::create();
        break;
    case KCF:
        m_tracker = TrackerKCF::create();
        break;
    case TLD:
        m_tracker = TrackerTLD::create();
        break;
    case MEDIANFLOW:
        m_tracker = TrackerMedianFlow::create();
        break;
    case GOTURN:
        m_tracker = TrackerGOTURN::create();
        break;
    case MOSSE:
        m_tracker = TrackerMOSSE::create();
        break;
    case CSRT:
        m_tracker = TrackerCSRT::create();
        break;
    default:
        break;
    }
}