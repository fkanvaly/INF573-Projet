#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <deque>
#include <numeric>
#include <map>
#include <set>

#include "definition.h"
#include "SingleTracker.h"
#include "HungarianAlg/HungarianAlg.h"

// ----------------------------------------------------------------------

///
/// \brief The TrackerSettings struct
///
struct TrackerSettings
{
	trackerType m_tracker_type = CSRT;
	std::array<track_t, DistsCount> m_distType;

    /// Time step for Kalman
    track_t m_dt = 1.0f;

    /// Noise magnitude for Kalman
    track_t m_accelNoiseMag = 0.1f;

    /// Distance threshold for Assignment problem: from 0 to 1
    track_t m_distThres = 0.8f;

    /// Minimal area radius in pixels for objects centers
    track_t m_minAreaRadius = 20.f;

    /// If the object don't assignment more than this frames then it will be removed
    size_t m_maximumAllowedSkippedFrames = 25;

    /// The maximum trajectory length
    size_t m_maxTraceLength = 50;

    /// After this time (in seconds) the object is considered abandoned
    int m_minStaticTime = 5;

    /// After this time (in seconds) the abandoned object will be removed
    int m_maxStaticTime = 25;

	/// Object types that can be matched while tracking
	std::map<std::string, std::set<std::string>> m_nearTypes;

	///
	TrackerSettings()
	{
		m_distType[DistJaccard] = 1.0f;
	}

};

///
/// \brief The CTracker class
///
class CTracker
{
public:
    CTracker(const TrackerSettings& settings);
	CTracker(const CTracker&) = delete;
	CTracker(CTracker&&) = delete;
	CTracker& operator=(const CTracker&) = delete;
	CTracker& operator=(CTracker&&) = delete;
	
	~CTracker(void);

    void Update(const regions_t& regions, cv::UMat currFrame);

    ///
    /// \brief GetTracksCount
    /// \return
    ///
	size_t GetTracksCount() const
	{
		return m_tracks.size();
	}
    ///
    /// \brief GetTracks
    /// \return
    ///
	std::vector<TrackingObject> GetTracks() const
	{
		std::vector<TrackingObject> tracks;
		if (!m_tracks.empty())
		{
			tracks.reserve(m_tracks.size());
			for (const auto& track : m_tracks)
			{	
                tracks.push_back(track->ConstructObject());
			}
		}
		return tracks;
	}

private:
    TrackerSettings m_settings;

	tracks_t m_tracks;

    size_t m_nextTrackID;

    cv::UMat m_prevFrame;

    void CreateDistaceMatrix(const regions_t& regions, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost);
    void SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment);
    void UpdateTrackingState(const regions_t& regions, cv::UMat currFrame);
};
