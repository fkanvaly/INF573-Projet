
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/bgsegm.hpp>
#include <map>
#include <vector>


using namespace cv;
using namespace std;

const char* params
    = "{ help h         |           | Print usage }"
      "{ @1             | vtest.avi | Path to a video or a sequence of image }"
      "{ out            | out.avi   | Path to output video}";


bool OpenCapture(cv::VideoCapture& capture, string filepath, int& m_fps);
bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame, 
                string outFile, int fps);

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

    //BGS
    Ptr<cv::BackgroundSubtractor> BGS1 = cv::bgsegm::createBackgroundSubtractorGSOC();
    Ptr<cv::BackgroundSubtractor> BGS2 = createBackgroundSubtractorMOG2();
    Ptr<cv::BackgroundSubtractor> BGS3 = createBackgroundSubtractorKNN();
    Ptr<cv::BackgroundSubtractor> BGS4 = cv::bgsegm::createBackgroundSubtractorGMG();


    cout<<"Start ! ";
    while (true) {
        //load frame
        Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // Detection
        Mat fgMask;


        int operation = 0;
        int morph_elem = 0;
        int morph_size = 0;
        int morph_operator = 0;
        int const max_operator = 4;
        int const max_elem = 2;
        int const max_kernel_size = 21;
        Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

        Mat res1, res2, res3, res4;
        BGS1->apply(gray, res1);
        BGS2->apply(gray, res2);
        BGS3->apply(gray, res3);
        BGS4->apply(gray, res4);
        morphologyEx( res1, res1, operation, element );
        morphologyEx( res2, res2, operation, element );
        morphologyEx( res3, res3, operation, element );
        morphologyEx( res4, res4, operation, element );
        resize(res1, res1, cv::Size(), 0.75, 0.75);
        resize(res2, res2, cv::Size(), 0.75, 0.75);
        resize(res3, res3, cv::Size(), 0.75, 0.75);
        resize(res4, res4, cv::Size(), 0.75, 0.75);
        putText(res1,"GSOC", Point(0,25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),1);
        putText(res2,"MOG2", Point(0,25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),1);
        putText(res3,"KNN", Point(0,25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),1);
        putText(res4,"GMG", Point(0,25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255),1);

        Mat l1,l2, res;
        hconcat(res1, res2, l1);
        hconcat(res3, res4, l2);
        vconcat(l1, l2, res);
        
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("mask", res);
        WriteFrame(writer, frame, outPath, m_fps);

        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    capture.release();
    writer.release();

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
