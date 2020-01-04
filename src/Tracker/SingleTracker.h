#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#include "definition.h"
#include "Kalman.h"
#include <opencv2/tracking.hpp>

struct TrajectoryPoint
{
    ///
    /// \brief TrajectoryPoint
    ///
    TrajectoryPoint()
        : m_hasRaw(false)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    ///
    TrajectoryPoint(const Point_t& prediction)
        :
          m_hasRaw(false),
          m_prediction(prediction)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    /// \param raw
    ///
    TrajectoryPoint(const Point_t& prediction, const Point_t& raw)
        :
          m_hasRaw(true),
          m_prediction(prediction),
          m_raw(raw)
    {
    }

    bool m_hasRaw = false;
    Point_t m_prediction;
    Point_t m_raw;
};

class Trace
{
public:
    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }
    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }
    const TrajectoryPoint& at(size_t i) const
    {
        return m_trace[i];
    }
    size_t size() const
    {
        return m_trace.size();
    }
    void push_back(const Point_t& prediction)
    {
        m_trace.emplace_back(prediction);
    }
    void push_back(const Point_t& prediction, const Point_t& raw)
    {
        m_trace.emplace_back(prediction, raw);
    }
    void pop_front(size_t count)
    {
        if (count < size())
        {
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        }
        else
        {
            m_trace.clear();
        }
    }
    size_t GetRawCount(size_t lastPeriod) const
    {
        size_t res = 0;

        size_t i = 0;
        if (lastPeriod < m_trace.size())
        {
            i = m_trace.size() - lastPeriod;
        }
        for (; i < m_trace.size(); ++i)
        {
            if (m_trace[i].m_hasRaw)
            {
                ++res;
            }
        }

        return res;
    }

private:
    std::deque<TrajectoryPoint> m_trace;
};

struct TrackingObject
{
    cv::RotatedRect m_rrect;           // Coordinates
	Trace m_trace;                     // Trajectory
	size_t m_ID = 0;                   // Objects ID
	bool m_outOfTheFrame = false;      // Is object out of freme
	std::string m_type;                // Objects type name or empty value
	float m_confidence = -1;           // From Detector with score (YOLO or SSD)
	cv::Vec<track_t, 2> m_velocity;    // pixels/sec
    Time_t m_start;
    Scalar m_color;


	///
    TrackingObject(const cv::RotatedRect& rrect, size_t ID, const Trace& trace, Time_t start, Scalar color,
		           bool outOfTheFrame, const std::string& type, float confidence, cv::Vec<track_t, 2> velocity)
		:
        m_rrect(rrect), m_ID(ID), m_outOfTheFrame(outOfTheFrame), m_type(type), 
        m_confidence(confidence), m_velocity(velocity), m_start(start), m_color(color)
	{
		for (size_t i = 0; i < trace.size(); ++i)
		{
            auto tp = trace.at(i);
            if (tp.m_hasRaw)
            {
                m_trace.push_back(tp.m_prediction, tp.m_raw);
            }
            else
            {
                m_trace.push_back(tp.m_prediction);
            }
		}
	}

	///
	bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const
	{
		bool res = m_trace.size() > static_cast<size_t>(minTraceSize);
		res &= m_trace.GetRawCount(m_trace.size() - 1) / static_cast<float>(m_trace.size()) > minRawRatio;
		if (sizeRatio.width + sizeRatio.height > 0)
		{
            float sr = m_rrect.size.width / m_rrect.size.height;
			if (sizeRatio.width > 0)
			{
				res &= (sr > sizeRatio.width);
			}
			if (sizeRatio.height > 0)
			{
				res &= (sr < sizeRatio.height);
			}
		}
		if (m_outOfTheFrame)
		{
			res = false;
		}
		return res;
	}

    int time_spent() const
    {
        Time_t t_now = std::chrono::system_clock::now();
        int time_elapsed = std::chrono::duration_cast<std::chrono::seconds>
                             (t_now-m_start).count();
        return time_elapsed;
    }

};

class CTrack
{
public:
    CTrack(const CRegion& region,
            track_t deltaTime,
            track_t accelNoiseMag,
            size_t trackID, 
            trackerType tracker_type);
    track_t CalcDistJaccard(const CRegion& reg) const;

    track_t IsInsideArea(const Point_t& pt, track_t minVal) const;

    void Update(const CRegion& region, bool dataCorrect, size_t max_trace_length, cv::UMat prevFrame, cv::UMat currFrame, int trajLen);

	bool IsOutOfTheFrame() const;

    cv::RotatedRect GetLastRect() const;

    const CRegion& LastRegion() const;
    size_t SkippedFrames() const;
    size_t& SkippedFrames();

    TrackingObject ConstructObject() const;

private:
    Trace m_trace;
    size_t m_trackID = 0;
    size_t m_skippedFrames = 0;
    CRegion m_lastRegion;
    Time_t m_time;
    Scalar m_color;

    Point_t m_predictionPoint;
    cv::RotatedRect m_predictionRect;
    TKalmanFilter m_kalman;
    cv::Ptr<cv::Tracker> m_tracker;
    trackerType m_tracker_type;
    bool m_isRec_estimation = true;
    bool m_outOfTheFrame = false;

    void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);
    void create_tracker();

    bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region);
    bool m_isStatic = false;
    int m_staticFrames = 0;
    cv::UMat m_staticFrame;
    cv::Rect m_staticRect;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
