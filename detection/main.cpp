
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

const char* params
    = "{ help h         |           | Print usage }"
      "{ input          | vtest.avi | Path to a video or a sequence of image }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";

Mat frame, fgMask;

int morph_elem = 0;
int morph_size = 2;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;


/** Function Headers */
void Morphology_Operations( int, void* );

int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program shows how to use background subtraction methods provided by "
                  " OpenCV. You can process both videos and images.\n" );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }

    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();

    // VideoCapture capture( samples::findFile( parser.get<String>("input") ) );
    VideoCapture capture( 0 ) ;
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }

     /// Create window
    namedWindow( "FG Mask", WINDOW_AUTOSIZE );

    /// Create Trackbar to select Morphology operation
    createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", "FG Mask",
                    &morph_operator, max_operator );

    /// Create Trackbar to select kernel type
    createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", "FG Mask",
                    &morph_elem, max_elem );

    /// Create Trackbar to choose kernel size
    createTrackbar( "Kernel size:\n 2n +1", "FG Mask",
                    &morph_size, max_kernel_size );

    createTrackbar( "Kernel size:\n 2n +1", "FG Mask",
                    &morph_size, max_kernel_size );


    // Filter by Circularity
    SimpleBlobDetector::Params params;
    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;
    
    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 10;
    int minArea = 10;

    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;

    Mat BG = imread("../bg.png");

    namedWindow( "FG", WINDOW_AUTOSIZE );
    createTrackbar( "min Aera thresol", "FG",
                    &minArea, 1000 );

    while (true) {

        capture >> frame;
        if (frame.empty())
            break;

        //update the background model
        // pBackSub->apply(frame, fgMask);
        Mat foreground;
        absdiff(BG, frame, foreground);
        imwrite("fore.png", foreground);
        if(foreground.channels()==3)
            cvtColor(foreground, foreground, COLOR_BGR2GRAY);
        threshold(foreground, fgMask, 15, 255, THRESH_BINARY);
        
        Morphology_Operations(0,0);
        Morphology_Operations(1,0);

        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);

        // Detect blobs.
        Mat output;
        // Set up the detector with default parameters.
        // Setup SimpleBlobDetector parameters.

        params.minArea = minArea;
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        std::vector<KeyPoint> keypoints;
        detector->detect( frame, keypoints, fgMask);
        for (int i=0;i<keypoints.size();i++)
            cout<<keypoints[i].pt<<"\n";
        drawKeypoints( frame, keypoints, output, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
 
        imshow("FG", output);
        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}

void Morphology_Operations( int, void*)
{
    // Since MORPH_X : 2,3,4,5 and 6
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
    morphologyEx( fgMask, fgMask, operation, element );
  }