#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace Ort {
    class Session;
    class Env;
}

// Structure to store a registered face
struct FaceFeature {
    std::string name;
    std::vector<float> feature;
    cv::Mat thumbnail;  // Small image for display purposes
};

class FaceRecognizer {
public:
    FaceRecognizer(const std::string& model_path);
    ~FaceRecognizer();
    
    // Extract features from aligned face image
    std::vector<float> extractFeature(const cv::Mat& aligned_face);
    
    // Compare two feature vectors and return similarity score (0-1)
    float compareFaces(const std::vector<float>& feature1, 
                      const std::vector<float>& feature2) const;
    
    // Add a face to the database
    bool addFace(const std::string& name, const cv::Mat& aligned_face);
    
    // Add a face with precomputed features
    bool addFace(const std::string& name, const std::vector<float>& feature, const cv::Mat& thumbnail);
    
    // Recognize a face from the database
    std::pair<std::string, float> recognize(const cv::Mat& aligned_face, float threshold = 0.6f) const;
    
    // Recognize a face using precomputed features
    std::pair<std::string, float> recognize(const std::vector<float>& feature, float threshold = 0.6f) const;
    
    // Save the database to a file
    bool saveFaceDatabase(const std::string& file_path) const;
    
    // Load the database from a file
    bool loadFaceDatabase(const std::string& file_path);
    
    // Get all registered faces
    const std::vector<FaceFeature>& getFaces() const { return face_database_; }
    
    // Get inference time in ms
    float getInferenceTime() const { return inference_time_; }
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    
    // Model parameters
    std::vector<int64_t> input_shape_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // Input size for the model (from model analysis)
    int input_width_ = 112;
    int input_height_ = 112;
    
    // Feature dimension from model (typically 512 for resnet100)
    int feature_dim_ = 512;
    
    // Face database
    std::vector<FaceFeature> face_database_;
    
    // Timing information
    float inference_time_ = 0.0f;
    
    // Pre-process the aligned face for the model
    void preprocess(const cv::Mat& face, float* input_data);
    
    // Normalize the feature vector to unit length
    void normalizeFeature(std::vector<float>& feature) const;
};
