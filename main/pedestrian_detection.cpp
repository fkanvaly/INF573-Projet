#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main( int argc, const char** argv )
{
    /// Use the cmdlineparser to process input arguments
    CommandLineParser parser(argc, argv,
        "{ help h       |      | show this message }"
        "{ @1           |      | (required) path to video }"
    );

    /// If help is entered
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }

    /// Parse arguments
    string video_location(parser.get<string>(0));
    if (video_location.empty()){
        parser.printMessage();
        return -1;
    }

    /// Create a videoreader interface
    Mat current_frame = imread(video_location);

    /// Set up the pedestrian detector --> let us take the default one
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    /// Check if the frame has content
    if(current_frame.empty()){
        cerr << "Video has ended or bad frame was read. Quitting." << endl;
        return 0;
    }

    /// run the detector with default parameters. to get a higher hit-rate
    /// (and more false alarms, respectively), decrease the hitThreshold and
    /// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).

    ///image, vector of rectangles, hit threshold, win stride, padding, scale, group th
    Mat img = current_frame.clone();
    resize(img,img,Size(img.cols*2, img.rows*2));

    vector<Rect> found;
    vector<double> weights;

    hog.detectMultiScale(img, found, weights, 0, Size(8,8), Size(32,32), 1.05);
    cout<<found.size()<<endl;

    /// draw detections and store location
    for( size_t i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];
        rectangle(img, found[i], cv::Scalar(0,0,255), 3);
    }

    /// Show
    imshow("detected person", img);
    waitKey(0);

    return 0;
}