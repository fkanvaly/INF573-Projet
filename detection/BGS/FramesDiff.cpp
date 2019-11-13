#include "FramesDiff.hpp"

FramesDiff::FramesDiff(){}
FramesDiff::~FramesDiff(){}

void FramesDiff::process(const Mat &img_input, Mat &img_output, Mat &img_bgmodel){
    if(!img_input.empty())
        return;

    if(!img_previous.empty())
        img_input.copyTo(img_previous);
    
    Mat img_foreground;
    absdiff(img_previous, img_input, img_foreground);
    threshold(img_foreground, img_foreground, 15, 255, THRESH_BINARY);

    img_foreground.copyTo(img_output);
    img_previous.copyTo(img_bgmodel);

    img_previous.copyTo(img_previous);

}