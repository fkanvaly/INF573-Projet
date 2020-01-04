
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/bgsegm.hpp>


#include <map>
#include <vector>
#include <atomic>
#include <thread>
#include <queue> 


#include "definition.h"
#include "DetectorBase.h"
#include "MultiTracker.h"
#include "Classifier.h"

using namespace cv;
using namespace std;

const char* params
    = "{ help h         |           | Print usage }"
      "{ @1             | vtest.avi | Path to a video or a sequence of image }"
      "{ out            | out.avi   | Path to output video}"
      "{ clf            | false     | with classifyer}"
      "{ dtc            | motion    | choose your detector}"
      "{ tracker        | CSRT      | choose your tracker}"
      "{ all            | true      | choose your tracker}"
      ;

float max_time = 20;
map <size_t, int> alert_record;
float freq_check=10;
bool id_run = true;

int m_fps;


queue<alert_obj> to_identify;
map<size_t, string> identified;

std::unique_ptr<DetectorBase> motion_detector;
std::unique_ptr<DetectorBase> face_detector;
std::unique_ptr<DetectorBase> people_detector;
std::unique_ptr<DetectorBase> car_detector;
std::unique_ptr<YoloClassifier> yolo;

std::vector<cv::Scalar> m_colors = {Scalar(255, 0, 0), Scalar(0, 255, 0),
                                    Scalar(0, 0, 255), Scalar(255, 255, 0),
                                    Scalar(0, 255, 255), Scalar(255, 0, 255),
                                    Scalar(255, 127, 255), Scalar(127, 0, 255),
                                    Scalar(127, 0, 127)};

// Here we get four points from the user with left mouse clicks.
// On 5th click we output the overlayed image.
vector<Point2f> right_image;
vector<Point2f> left_image {Point2f(0.0f,0.0f), Point2f(0.0f, 480.0f),
                            Point2f(640.0f, 480.0f), Point2f(640.0f, 0.0f)};

Mat H; //homography matrix
bool isH_ready=false;
Mat board;

Rect f_zone;

/** Function Headers */
bool OpenCapture(cv::VideoCapture& capture, string filepath, int& m_fps);
bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame, string outFile, int fps);
void DrawTrack(cv::Mat& frame, int resizeCoeff, const TrackingObject& track, float bar_s, bool drawTrajectory);
void nms(const regions_t& srcRects, regions_t& resRects, float thresh, int neighbors =0);
void identification_thread();
void analyse_and_draw(std::vector<TrackingObject>& tracks, Mat roi);
void on_mouse( int e, int x, int y, int d, void *ptr );

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);

    VideoCapture capture;
    string path = parser.get<string>(0);
    if (!OpenCapture(capture, path, m_fps)){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>(0) << endl;
        return 0;
    }
    cout<<"fps: "<<m_fps<<endl;

    cv::VideoWriter writer;
    string outPath = parser.get<String>("out");

    config_t config;
    int minWidth = 35;
    int m_minStaticTime = 5;


    // Motion detection
    cout<<"initializing Motion Detector....";
    config.emplace("history", std::to_string(cvRound(10 * m_minStaticTime * m_fps)));
    config.emplace("varThreshold", "16");
    config.emplace("detectShadows", "0");
    config.emplace("useRotatedRect", "0");
    motion_detector = std::unique_ptr<DetectorBase>(CreateDetector(DetectorsName::Motion_GSOC, config, minWidth));
    cout<<"[OK]"<<endl;

    //Face Detection
    cout<<"initializing Face Detector....";
    std::string face_pathToModel = "../data/";
    config.emplace("cascadeFileName", face_pathToModel + "haarcascade_frontalface_alt2.xml");
    face_detector = std::unique_ptr<DetectorBase>(CreateDetector(DetectorsName::Face, config, minWidth));
    cout<<"[OK]"<<endl;

    //Car Detection
    cout<<"initializing Car Detector....";
    std::string car_pathToModel = "../data/";
    config.emplace("cascadeFileName", car_pathToModel + "cars.xml");
    car_detector = std::unique_ptr<DetectorBase>(CreateDetector(DetectorsName::Car, config, minWidth));
    cout<<"[OK]"<<endl;

    //People Detection
    cout<<"initializing People Detector....";
    people_detector = std::unique_ptr<DetectorBase>(CreateDetector(DetectorsName::People, config, minWidth));
    cout<<"[OK]"<<endl;

    //Yolo Classifier
    cout<<"initializing People Detector....";
    std::string classNames = "../data/yolo/class.names";
    std::string cfg = "../data/yolo/yolov3.cfg";
    std::string weights = "../data/yolo/yolov3.weights";
    config.emplace("classNames", classNames);
    config.emplace("cfg", cfg);
    config.emplace("weights", weights);
    yolo = std::unique_ptr<YoloClassifier>(CreateYolo(config));
    cout<<"[OK]"<<endl;


    // initialize tracker
    cout<<"initializing Tracker....";
    trackerType T = (parser.get<string>("tracker").compare("MOSSE")==0) ? MOSSE : CSRT; 
    TrackerSettings settings;
        settings.m_tracker_type = T;
        settings.m_dt = 0.4f;                             // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
        settings.m_distThres = 0.95f;                    // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        settings.m_maxTraceLength = cvRound(4 * m_fps);  // Maximum trace length
    
    std::unique_ptr<CTracker> m_tracker = std::make_unique<CTracker>(settings);
    cout<<"[OK]"<<endl;

    //thread of identification alarm object identification
    id_run = (parser.get<string>("clf").compare("true")==0) ? true : false;
    thread id_thred(identification_thread);
    

    //mouse callbakc
    namedWindow( "Frame", WINDOW_AUTOSIZE );// Create a window for display.
    setMouseCallback("Frame", on_mouse, NULL );

    board = Mat::zeros(480,640, CV_8UC3);
    rectangle(board, Rect(0,0,640,480), Scalar(199, 195, 189), -1);

    Mat frame;
    capture >> frame;
    bool all = (parser.get<string>("all").compare("true")==0) ? true : false; 
    if( ! all)
        f_zone = selectROI(frame, false); //use to select 
    else
    {
        f_zone = Rect(Point(0,0), Point(frame.size()));
    }
    

    cout<<"Start ! ";
    while (true) {
        //load frame
        Mat frame;
        capture >> frame;
        UMat uframe = frame.getUMat(cv::ACCESS_READ);
        if (frame.empty())
            break;
        
        Mat roi = frame(f_zone);
        UMat uroi = uframe(f_zone);

        rectangle(frame, f_zone.tl(), f_zone.br(),
                    Scalar(0,0,255), 2);

        UMat ugray;
        cvtColor(uroi, ugray, COLOR_BGR2GRAY);
        

        // Detection
        motion_detector->Detect(ugray);
        regions_t regions=  motion_detector->GetDetection();
        regions_t res_regions ;
        nms(regions, res_regions, 0.7);
        
        
        //Tracking
        m_tracker->Update(res_regions, uroi);

        // // Draw tracks
        auto tracks = m_tracker->GetTracks();
        std::cout << "tracks = " << tracks.size() << std::endl;
        analyse_and_draw(tracks, roi);

        //show the current frame and the fg masks
        frame(f_zone) = roi;

        //draw area where homography will be applied
        if (isH_ready)
        {
            imshow("Board", board);
            rectangle(board, Rect(0,0,640,480), Scalar(199, 195, 189), -1);
            Mat c_frame = frame.clone();
            Point poly [1][4];
            poly[0][0] = right_image[0];
            poly[0][1] = right_image[1];
            poly[0][2] = right_image[2];
            poly[0][3] = right_image[3];
            const Point* ppt[1] = { poly[0] };
            int npt[] = { 4 };
            fillPoly(c_frame, ppt, npt,1, Scalar(0,0,255), LINE_8);
            addWeighted(frame, 0.5, c_frame, 0.5, 0, frame);
        }
        
        imshow("Frame", frame);
        // WriteFrame(writer, frame, outPath, m_fps);

        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

    }
    id_run = false; 
        
    capture.release();
    writer.release();

    return 0;
}

/**
 * @brief 
 * Open video given in argument or webcam
 */
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

/**
 * @brief 
 * Write video in avi format
 */
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

/**
 * @brief 
 * Check if a bbox has spent to much time in forbidden area then 
 * then draw it's tracks
 * @param tracks 
 * @param frame 
 */
void analyse_and_draw(std::vector<TrackingObject>& tracks, Mat frame)
{
    auto paddedRect = [](const Rect rec, float p)
    {
        float x = std::max(0.0f, rec.x - p/2);
        float y = std::max(0.0f, rec.y - p/2);
        Rect padded (x, y, 
                        rec.width + p/2, 
                        rec.height + p/2);
        
        return padded;
    };

    for (const auto& track : tracks)
    {
        if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                            0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                            cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                )
        {
            int time_elapsed = track.time_spent();
            float bar_s = std::min(time_elapsed/max_time, 1.0f);

            //did he spent too much time there?
            if (time_elapsed < max_time)
                DrawTrack(frame, 1, track, bar_s, true);
            else{
                // is it recorded?
                if(alert_record.find(track.m_ID) == alert_record.end()){ //no
                    alert_record[track.m_ID]=1;
                    CRegion roi (track.m_rrect);
                    Mat mat_roi = frame(paddedRect(roi.m_brect, 50));

                    alert_obj to_add  {track.m_ID, mat_roi};
                    to_identify.push(to_add);
                }
                else{ //yes
                    if (time_elapsed/freq_check>alert_record[track.m_ID] && //rerun identification if identity not found
                        identified.find(track.m_ID) != identified.end() && //not yet identify
                        identified[track.m_ID].compare("nothing") ==0 //identifie as nothing
                        ) {
                        CRegion roi (track.m_rrect);
                        Mat mat_roi = frame(paddedRect(roi.m_brect, 50));

                        alert_obj to_add  {track.m_ID, mat_roi};
                        to_identify.push(to_add);
                        alert_record[track.m_ID]++;

                        //save for further analysis
                        // imwrite("../data/alert/"+to_string(track.m_ID)+
                        //         to_string(alert_record[track.m_ID])+".png", mat_roi);

                    }
                }
                DrawTrack(frame, 1, track, bar_s, true);
            }
        }
    }
}

/**
 * @brief 
 * Draw track trajectory, homography dot and trajectory
 */
void DrawTrack(cv::Mat& frame, int resizeCoeff, const TrackingObject& track, float bar_s, bool drawTrajectory)
{
    auto ResizePoint = [resizeCoeff](const cv::Point& pt) -> cv::Point
    {
        return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
    };

    cv::Point2f rectPoints[4];
    track.m_rrect.points(rectPoints);

    if (identified.find(track.m_ID) != identified.end())
    {
        putText(frame, identified[track.m_ID], rectPoints[1], FONT_HERSHEY_SIMPLEX, 0.5, LINE_AA, 1);
    }

    auto interpolate = [](const float a, const float b, const float s) -> float
    {
        return (1-s)*a + s*b;
    };

    float s = bar_s;
    Point start_p = rectPoints[0];
    Point end_p = rectPoints[3];
    Point i_p = Point(interpolate(start_p.x, end_p.x, s), start_p.y-5);

    //Charging color interpolation
    rgb green {0, 255, 0};
    rgb red {255, 0, 0};
    hsv hsv_green = rgb2hsv(green);
    hsv hsv_red = rgb2hsv(red);
    //interpolate color in hsv space
    hsv hsv_color {interpolate(hsv_green.h, hsv_red.h, s),
                   interpolate(hsv_green.s, hsv_red.s, s),
                   interpolate(hsv_green.v, hsv_red.v, s)};
    rgb rgb_c = hsv2rgb(hsv_color);

    //charging bar
    rectangle(frame, start_p, i_p, Scalar(rgb_c.b, rgb_c.g, rgb_c.r),-1);

    Scalar color = (s==1.0f) ? Scalar(0,0,255) : Scalar(0,255,0);
    for (int i = 0; i < 4; ++i)
    {
        cv::line(frame, ResizePoint(rectPoints[i]), ResizePoint(rectPoints[(i+1) % 4]), color, 1);
    }

    cv::Scalar cl = m_colors[track.m_ID % m_colors.size()];

    if (isH_ready)
    {
        CRegion bbox (track.m_rrect);
        vector<Point2f> v(1), dest(1);
        v[0] = track.m_rrect.center + Point2f(f_zone.tl().x,f_zone.tl().y); // add because the rect are in subplan
        // v[0] = (rectPoints[0]+rectPoints[3])/2 + Point2f(f_zone.tl().x,f_zone.tl().y); // add because the rect are in subplan
        perspectiveTransform( v, dest, H.inv());
        circle(board, dest[0], 7, cl, -1);
        if (identified.find(track.m_ID) != identified.end())
        {
            putText(board, identified[track.m_ID], dest[0], FONT_HERSHEY_SIMPLEX, 0.5, LINE_AA, 1);
        }
    }

    if (drawTrajectory)
    {

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            //on real map
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);

            //on board
            if (isH_ready && track.m_trace.size()-11>0)
            {
                Point p1 = ResizePoint(pt1.m_prediction); Point2f p1_f (p1.x, p1.y); //convert to Point2f
                Point p2 = ResizePoint(pt2.m_prediction); Point2f p2_f (p2.x, p2.y);
                vector<Point2f> v {p1_f + Point2f(f_zone.tl().x,f_zone.tl().y),
                                   p2_f + Point2f(f_zone.tl().x,f_zone.tl().y)};
                vector<Point2f> dest(2);
                perspectiveTransform( v, dest, H.inv()); //Apply homography

                cv::line(board, dest[0], dest[1], cl, 1, cv::LINE_AA);
            }
            
            if (!pt2.m_hasRaw)
            {
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, cv::LINE_AA);
            }
        }
    }
}

/**
 * @brief 
 * Received suspecious regions and identify the content
 */
void identification_thread(){
    while (id_run)
    {
        if(!to_identify.empty()){

            alert_obj unknown = to_identify.front();

            regions_t found;
            yolo->Classify(unknown.m_roi);
            identified[unknown.m_ID]= yolo->getClass();
            found.clear();
            to_identify.pop();

            std::cout << "To identify " << to_identify.size() << std::endl;


        }
    }
    
}

/**
 * @brief 
 * Non maximal suppression delete overlaping bbox
 */
void nms(const regions_t& srcRects, regions_t& resRects, float thresh, int neighbors)
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].m_brect.br().y, i));
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const CRegion& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const CRegion& rect2 = srcRects[pos->second];

			float intArea = static_cast<float>((rect1.m_brect & rect2.m_brect).area());
			float unionArea = static_cast<float>(rect1.m_brect.area() + rect2.m_brect.area() - intArea);
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
        {
            resRects.push_back(rect1);
        }
    }
}


// Here we get four points from the user with left mouse clicks.
// On 5th click we output the overlayed image.
void on_mouse( int e, int x, int y, int d, void *ptr )
{
    if (e == EVENT_LBUTTONDOWN )
    {
        if(right_image.size() < 4 )
        {

            right_image.push_back(Point2f(float(x),float(y)));
            cout << x << " "<< y <<endl;
        }
        else
        {
            cout << " Calculating Homography " <<endl;
            // Deactivate callback
            cv::setMouseCallback("Frame", NULL, NULL);
            // once we get 4 corresponding points in both images calculate homography matrix
            H = findHomography( left_image,right_image,0 );
            isH_ready=true;
        }

    }
}