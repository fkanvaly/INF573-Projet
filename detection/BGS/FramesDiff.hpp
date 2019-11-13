
#include "BGS.hpp"

class FramesDiff  : public BGS
{
    private:
        Mat img_previous;

    public:
        FramesDiff();
        ~FramesDiff();
        void process(const Mat &img_input, Mat &img_foreground, Mat &img_background)  ;
    
    private:
        void saveConfig();
        void loadConfig();
};