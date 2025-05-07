#include "face_recognizer.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>

FaceRecognizer::FaceRecognizer(const std::string& model_path) {
    try {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceRecognizer");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get model info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Based on face_recognizer_debug, the input node is named "data"
        // and the output node is named "fc1"
        input_names_str_.push_back("data");
        output_names_str_.push_back("fc1");
        
        // Get input shape information
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();

        // Initialize input dimensions
        // input_shape_ = {1, 3, H, W}
        if (input_shape_.size() == 4) {
            input_height_ = static_cast<int>(input_shape_[2]);
            input_width_  = static_cast<int>(input_shape_[3]);
        } else {
            // fallback
            input_height_ = 112;
            input_width_ = 112;
        }
        
        // From the debug output, we know the feature dimension is 512
        feature_dim_ = 512;
        
        // Set up the input and output name pointers
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
        
        std::cout << "Face recognizer initialized with model: " << model_path << std::endl;
        std::cout << "Feature dimension: " << feature_dim_ << std::endl;
    }
    catch(const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw;
    }
}

FaceRecognizer::~FaceRecognizer() {
    // Smart pointers will clean up automatically
}

void FaceRecognizer::preprocess(const cv::Mat& face, float* input_data) {
    // Ensure the input face is the correct size
    cv::Mat resized_face;
    if (face.rows != input_height_ || face.cols != input_width_) {
        cv::resize(face, resized_face, cv::Size(input_width_, input_height_));
    } else {
        resized_face = face;
    }
    
    // Convert to float and normalize to [0,1]
    cv::Mat float_mat;
    resized_face.convertTo(float_mat, CV_32FC3, 1.0/255.0);

    // ArcFace standard normalization: (x - 0.5) / 0.5 = x * 2 - 1
    float_mat = float_mat * 2.0f - 1.0f;

    // Reorder from BGR to RGB if needed and copy to input buffer
    // NCHW layout: [batch, channels, height, width]
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_height_; h++) {
            for (int w = 0; w < input_width_; w++) {
                // For RGB order (most face recognition models)
                input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                    float_mat.at<cv::Vec3f>(h, w)[2 - c]; // BGR to RGB
            }
        }
    }
}

std::vector<float> FaceRecognizer::extractFeature(const cv::Mat& aligned_face) {
    if (aligned_face.empty()) {
        std::cerr << "Error: Empty face image provided for feature extraction" << std::endl;
        return {};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Prepare input data
    std::vector<float> input_data(input_width_ * input_height_ * 3);
    preprocess(aligned_face, input_data.data());
    
    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_dims.data(), input_dims.size());
    
    // Run inference
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), output_names_.size());
    
    auto end = std::chrono::high_resolution_clock::now();
    inference_time_ = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Extract output data
    std::vector<float> feature;
    if (!outputs.empty()) {
        float* output_data = outputs[0].GetTensorMutableData<float>();
        auto output_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // Debug: print output shape
        std::cout << "Model output shape: [";
        for (size_t i = 0; i < output_dims.size(); ++i) {
            std::cout << output_dims[i];
            if (i != output_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        size_t feature_size = 1;
        for (auto& dim : output_dims) {
            feature_size *= dim;
        }

        feature.assign(output_data, output_data + feature_size);

        // Debug: L2 norm before normalization
#if 0
        float norm = 0.0f;
        for (auto v : feature) norm += v * v;
        norm = std::sqrt(norm);
        std::cout << "Feature L2 norm before normalize: " << norm << std::endl;
#endif

        // Normalize the feature vector
        normalizeFeature(feature);

        // Debug: feature values and L2 norm after normalization
#if 0
        std::cout << "Feature (first 10, after normalize): ";
        for (size_t i = 0; i < std::min(feature.size(), size_t(10)); ++i) {
            std::cout << feature[i] << " ";
        }
        std::cout << std::endl;
        float norm2 = 0.0f;
        for (auto v : feature) norm2 += v * v;
        norm2 = std::sqrt(norm2);
        std::cout << "Feature L2 norm after normalize: " << norm2 << std::endl;
#endif
    }
    
    return feature;
}

void FaceRecognizer::normalizeFeature(std::vector<float>& feature) const {
    // Normalize to unit length (L2 norm)
    float norm = 0.0f;
    for (const auto& val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 1e-6f) {
        for (auto& val : feature) {
            val /= norm;
        }
    }
}

// Added: Calculate the mean of multiple features and normalize it
std::vector<float> FaceRecognizer::meanFeature(const std::vector<std::vector<float>>& features) const {
    if (features.empty()) return {};
    std::vector<float> mean(features[0].size(), 0.0f);
    for (const auto& feat : features) {
        for (size_t i = 0; i < feat.size(); ++i) {
            mean[i] += feat[i];
        }
    }
    for (auto& v : mean) v /= features.size();
    normalizeFeature(mean);
    return mean;
}

float FaceRecognizer::compareFaces(
    const std::vector<float>& feature1, 
    const std::vector<float>& feature2) const {
    
    if (feature1.empty() || feature2.empty() || feature1.size() != feature2.size()) {
        return 0.0f;
    }
    
    // Compute cosine similarity
    float dot_product = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        dot_product += feature1[i] * feature2[i];
    }
    
    // Directly return the cosine similarity without clipping
    return dot_product;
}

bool FaceRecognizer::addFace(const std::string& name, const cv::Mat& aligned_face) {
    if (aligned_face.empty() || name.empty()) {
        return false;
    }
    
    // Extract features
    std::vector<float> feature = extractFeature(aligned_face);
    if (feature.empty()) {
        return false;
    }
    
    // Create a small thumbnail for display
    cv::Mat thumbnail;
    cv::resize(aligned_face, thumbnail, cv::Size(64, 64));
    
    // Add to database
    FaceFeature face_data = {name, feature, thumbnail};
    face_database_.push_back(face_data);
    
    std::cout << "Added face: " << name << " to database" << std::endl;
    return true;
}

bool FaceRecognizer::addFace(const std::string& name, const std::vector<float>& feature, const cv::Mat& thumbnail) {
    if (feature.empty() || name.empty()) {
        return false;
    }
    
    // Add to database
    FaceFeature face_data = {name, feature, thumbnail.clone()};
    face_database_.push_back(face_data);
    
    std::cout << "Added face: " << name << " to database" << std::endl;
    return true;
}

std::pair<std::string, float> FaceRecognizer::recognize(const cv::Mat& aligned_face, float threshold) const {
    if (aligned_face.empty()) {
        return {"", 0.0f};
    }
    
    // Extract features
    std::vector<float> feature = const_cast<FaceRecognizer*>(this)->extractFeature(aligned_face);
    if (feature.empty()) {
        return {"", 0.0f};
    }
    
    return recognize(feature, threshold);
}

std::pair<std::string, float> FaceRecognizer::recognize(const std::vector<float>& feature, float threshold) const {
    if (feature.empty() || face_database_.empty()) {
        return {"", 0.0f};
    }
    
    std::string best_match = "";
    float best_score = 0.0f;
    
    // Compare with all faces in the database
    for (const auto& face : face_database_) {
        float similarity = compareFaces(feature, face.feature);
        if (similarity > best_score) {
            best_score = similarity;
            best_match = face.name;
        }
    }
    
    // Return the best match if score is above threshold
    if (best_score >= threshold) {
        return {best_match, best_score};
    }
    
    return {"Unknown", best_score};
}

bool FaceRecognizer::saveFaceDatabase(const std::string& file_path) const {
    // Create base directory if it doesn't exist
    std::filesystem::path base_dir = std::filesystem::path(file_path).parent_path();
    if (!std::filesystem::exists(base_dir)) {
        std::filesystem::create_directories(base_dir);
    }
    
    // Instead of saving a binary file, we'll save a simple text file with the list of names
    std::ofstream index_file(file_path);
    if (!index_file.is_open()) {
        std::cerr << "Error: Could not open index file for writing: " << file_path << std::endl;
        return false;
    }
    
    int face_count = 0;
    for (const auto& face : face_database_) {
        // Create directory for each person if it doesn't exist - this is the key fix
        // Ensure we're placing these directories under the same parent directory as the database file
        std::filesystem::path person_dir = base_dir / face.name;
        if (!std::filesystem::exists(person_dir)) {
            std::filesystem::create_directories(person_dir);
        }
        
        // Save feature vector to a simple binary file
        std::string feature_path = (person_dir / "feature.bin").string();
        std::ofstream feature_file(feature_path, std::ios::binary);
        if (feature_file.is_open()) {
            feature_file.write(reinterpret_cast<const char*>(face.feature.data()), 
                              face.feature.size() * sizeof(float));
            feature_file.close();
        }
        
        // Save face thumbnail
        std::string face_path = (person_dir / "face.jpg").string();
        cv::imwrite(face_path, face.thumbnail);
        
        // Add to index
        index_file << face.name << std::endl;
        face_count++;
    }
    
    index_file.close();
    std::cout << "Saved " << face_count << " faces to directory database" << std::endl;
    return true;
}

bool FaceRecognizer::loadFaceDatabase(const std::string& file_path) {
    // Clear existing database
    face_database_.clear();
    
    std::filesystem::path base_dir = std::filesystem::path(file_path).parent_path();
    std::filesystem::path index_path = file_path;
    
    // If the index file doesn't exist, return false
    if (!std::filesystem::exists(index_path)) {
        std::cerr << "Error: Database index file not found: " << file_path << std::endl;
        return false;
    }
    
    // Read the index file to get the list of names
    std::ifstream index_file(index_path);
    if (!index_file.is_open()) {
        std::cerr << "Error: Could not open index file: " << file_path << std::endl;
        return false;
    }
    
    int face_count = 0;
    std::string name;
    while (std::getline(index_file, name)) {
        if (name.empty()) continue;
        
        std::filesystem::path person_dir = base_dir / name;
        std::string feature_path = (person_dir / "feature.bin").string();
        std::string face_path = (person_dir / "face.jpg").string();
        
        // Check if required files exist
        if (!std::filesystem::exists(feature_path) || !std::filesystem::exists(face_path)) {
            std::cerr << "Warning: Missing files for person: " << name << std::endl;
            continue;
        }
        
        // Load feature
        std::vector<float> feature(feature_dim_);
        std::ifstream feature_file(feature_path, std::ios::binary);
        if (feature_file.is_open()) {
            feature_file.read(reinterpret_cast<char*>(feature.data()), 
                             feature_dim_ * sizeof(float));
            feature_file.close();
        } else {
            std::cerr << "Error: Could not open feature file: " << feature_path << std::endl;
            continue;
        }
        
        // Load thumbnail
        cv::Mat thumbnail = cv::imread(face_path);
        if (thumbnail.empty()) {
            std::cerr << "Error: Could not load face image: " << face_path << std::endl;
            continue;
        }
        
        // Resize thumbnail if needed
        if (thumbnail.cols != 64 || thumbnail.rows != 64) {
            cv::resize(thumbnail, thumbnail, cv::Size(64, 64));
        }
        
        // Add to database
        FaceFeature face_data = {name, feature, thumbnail};
        face_database_.push_back(face_data);
        face_count++;
    }
    
    index_file.close();
    std::cout << "Loaded " << face_count << " faces from directory database" << std::endl;
    return face_count > 0;
}
