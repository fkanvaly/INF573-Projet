#pragma once
#include "definition.h"
#include <opencv2/dnn.hpp>

using namespace dnn;

class YoloClassifier 
{
public:
    YoloClassifier();
    ~YoloClassifier(void);

    bool Initialize(const config_t& config);

    void Classify(cv::Mat& frame);

    // Remove the bounding boxes with low confidence using non-maxima suppression
    void postprocess(Mat& frame, const vector<Mat>& out);

    // Draw the predicted bounding box
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

    string getClass()
    {
        return m_classe;
    }

    vector<String> getOutputsNames(const Net& net);


private:
    // Initialize the parameters
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image
    vector<string> classes;
    string m_classe;
    Net net;
};

YoloClassifier* CreateYolo(const config_t& config);




