#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace Ort {
    class Session;
    class Env;
}

struct FaceInfo {
    cv::Rect bbox;                   // Bounding box (x, y, width, height)
    float score;                     // Detection confidence score
    std::vector<cv::Point2f> landmarks; // Facial landmarks (5 points)
};

class FaceDetector {
public:
    FaceDetector(const std::string& model_path);
    ~FaceDetector();
    
    // Detect faces in the given image
    std::vector<FaceInfo> detect(const cv::Mat& image, float conf_threshold = 0.7, float nms_threshold = 0.4);
    
    // Get the last processing time in milliseconds
    float getInferenceTime() const { return inference_time_; }
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    
    // Model parameters
    std::vector<int64_t> input_shape_;
    
    // Store the actual strings to ensure memory safety
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    
    // Pointers for ONNX Runtime
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // Pre-processing parameters
    float mean_vals_[3] = {104.0f, 117.0f, 123.0f}; // BGR order
    float scale_factor_ = 1.0f;
    
    // Input image size for the model (from model analysis)
    int input_width_ = 320;
    int input_height_ = 320;
    
    // Number of anchor boxes (from model analysis: 4200)
    int num_anchors_ = 4200;
    
    // Timing information
    float inference_time_ = 0.0f;
    
    // Helper methods
    void preprocess(const cv::Mat& image, float* input_data);
    std::vector<FaceInfo> postprocess(
        const std::vector<float>& loc_data, 
        const std::vector<float>& conf_data,
        const std::vector<float>& landms_data,
        float conf_threshold,
        float nms_threshold,
        const cv::Size& original_size);
    
    // Generate prior/anchor boxes
    std::vector<std::vector<float>> priorbox_;
    void generatePriorBoxes();
    
    // Non-maximum suppression
    std::vector<int> nms(const std::vector<FaceInfo>& faces, float threshold);
};
