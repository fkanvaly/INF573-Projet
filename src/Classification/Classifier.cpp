#include "Classifier.h"

YoloClassifier::YoloClassifier()
{

}

YoloClassifier::~YoloClassifier()
{
}


YoloClassifier* CreateYolo(const config_t& config)
{
    YoloClassifier* yolo = nullptr;
    yolo = new YoloClassifier();

    if (!yolo->Initialize(config))
    {
        delete yolo;
        yolo = nullptr;
    }
    return yolo;
}

/**
 * @brief 
 * Load yolo pretrain model on dnn 
 */
bool YoloClassifier::Initialize(const config_t& config)
{
    // Load names of classes
    auto classesFile = config.find("classNames");
    assert( classesFile != config.end()) ;
    
    ifstream ifs(classesFile->second.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    auto cfg = config.find("cfg");
    assert( cfg != config.end()) ;
    String modelConfiguration = cfg->second ;

    auto weights = config.find("weights");
    assert( weights != config.end()) ;
    String modelWeights = weights->second;

    // Load the network
    net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);

    return true;
}

void YoloClassifier::Classify(Mat& frame)
{
    Mat blob;
    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
    
    //Sets the input to the network
    net.setInput(blob);
    
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    
    // Remove the bounding boxes with low confidence
    // and set max confidence box as the class
    postprocess(frame, outs);    
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void YoloClassifier::postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    float max_conf = 0;
    int max_idx=-1;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        if (confidences[idx]>max_conf){
            max_idx = idx;
        }
    }

    if(max_idx==-1) m_classe="nothing";
    else m_classe = classes[classIds[max_idx]];
}


// Get the names of the output layers
vector<String> YoloClassifier::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
