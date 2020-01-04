
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <map>
#include <vector>


using namespace cv;
using namespace std;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

const char* params
    = "{ help h         |           | Print usage }"
      "{ @1             | vtest.avi | Path to a video or a sequence of image }"
      "{ out            | out.avi   | Path to output video}";


bool OpenCapture(cv::VideoCapture& capture, string filepath, int& m_fps);
bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame, 
                string outFile, int fps);

enum trackerType
{BOOSTING, MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT};

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);

    int m_fps;
    VideoCapture capture;
    string path = parser.get<string>(0);
    if (!OpenCapture(capture, path, m_fps)){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>(0) << endl;
        return 0;
    }

    cv::VideoWriter writer;
    string outPath = parser.get<String>("out");

    trackerType trackerType = CSRT;
 
    Ptr<Tracker> tracker;

    //Tracker
    switch (trackerType)
    {
    case BOOSTING:
        tracker = TrackerBoosting::create();
        break;
    case MIL:
        tracker = TrackerMIL::create();
        break;
    case KCF:
        tracker = TrackerKCF::create();
        break;
    case TLD:
        tracker = TrackerTLD::create();
        break;
    case MEDIANFLOW:
        tracker = TrackerMedianFlow::create();
        break;
    case GOTURN:
        tracker = TrackerGOTURN::create();
        break;
    case MOSSE:
        tracker = TrackerMOSSE::create();
        break;
    case CSRT:
        tracker = TrackerCSRT::create();
        break;
    default:
        break;
    }
    
    // Read first frame 
    Mat frame; 
    capture >> frame;  
 
    // Define initial bounding box 
    Rect2d bbox(287, 23, 86, 320); 
 
    // Uncomment the line below to select a different bounding box 
    bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 
 
    imshow("Tracking", frame); 
    tracker->init(frame, bbox);
     
    while(true)
    { 
        capture >> frame;    
        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
         
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
 
        // Display frame.
        imshow("Tracking", frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }

    return 0;
}


bool OpenCapture(cv::VideoCapture& capture, string filepath, int& m_fps)
{
    if (filepath.size() == 1)
    {
        capture.open(atoi(filepath.c_str()));
    }
    else
    {
        capture.open(filepath);
    }
    if (capture.isOpened())
    {
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);

        m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));
        return true;
    }
    return true;
}

bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame, 
                string outFile, int fps)
{
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    if (!outFile.empty())
    {
        if (!writer.isOpened())
        {
            writer.open(outFile, fourcc, fps, frame.size(), true);
        }
        if (writer.isOpened())
        {
            writer << frame;
            return true;
        }
    }
    return false;
}
