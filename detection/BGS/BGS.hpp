
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class BGS  
{
    public:
        virtual void process(const Mat &img_input, Mat img_foreground, Mat &img_background){} 
        virtual ~BGS(){}
    private:
        virtual void saveConfig(){} ;
        virtual void loadConfig(){} ;
};
