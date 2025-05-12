#include "face_recognizer.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread> // Required for std::thread::hardware_concurrency
#include <opencv2/dnn.hpp> // Added for cv::dnn::blobFromImage
#include <cstring> // Added for memcpy

FaceRecognizer::FaceRecognizer(const std::string& model_path, int intra_op_num_threads, int inter_op_num_threads) {
    try {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceRecognizer");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Set thread counts
        if (intra_op_num_threads > 0) {
            session_options.SetIntraOpNumThreads(intra_op_num_threads);
            std::cout << "Setting intra_op_num_threads to: " << intra_op_num_threads << std::endl;
        } else {
            unsigned int core_count = std::thread::hardware_concurrency();
            if (core_count > 0) {
                session_options.SetIntraOpNumThreads(core_count);
                std::cout << "Defaulting intra_op_num_threads to hardware concurrency: " << core_count << std::endl;
            } else {
                 std::cout << "Could not determine hardware concurrency, letting ONNX Runtime decide intra_op_num_threads." << std::endl;
            }
        }

        if (inter_op_num_threads > 0) {
            session_options.SetInterOpNumThreads(inter_op_num_threads);
            std::cout << "Setting inter_op_num_threads to: " << inter_op_num_threads << std::endl;
        } else {
            session_options.SetInterOpNumThreads(1);
            std::cout << "Defaulting inter_op_num_threads to 1." << std::endl;
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get model info
        Ort::AllocatorWithDefaultOptions allocator;

        if (model_path.find("arcfaceresnet100-11-int8") != std::string::npos) {
            input_names_str_.push_back("data");
            output_names_str_.push_back("fc1");
            feature_dim_ = 512;
        } else if (model_path.find("w600k_mbf") != std::string::npos) {
            input_names_str_.push_back("input.1");
            output_names_str_.push_back("516");
            feature_dim_ = 512;
        } else {
            throw std::runtime_error("Unsupported model: " + model_path);
        }

        checkNormalizationRequirement();
        
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();

        if (input_shape_.size() == 4) {
            input_height_ = static_cast<int>(input_shape_[2]);
            input_width_  = static_cast<int>(input_shape_[3]);
        } else {
            input_height_ = 112;
            input_width_ = 112;
        }
        
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
    cv::Mat resized_face;
    if (face.rows != input_height_ || face.cols != input_width_) {
        cv::resize(face, resized_face, cv::Size(input_width_, input_height_));
    } else {
        resized_face = face;
    }
    
    double scalefactor = requires_normalization_ ? (1.0 / 255.0) : 1.0;

    cv::Mat blob = cv::dnn::blobFromImage(resized_face, scalefactor,
                                          cv::Size(input_width_, input_height_),
                                          cv::Scalar(0, 0, 0), false, false, CV_32F);

    size_t expected_elements = static_cast<size_t>(3) * input_height_ * input_width_;

    if (static_cast<size_t>(blob.total()) != expected_elements) {
        std::cerr << "Error: Blob size mismatch in preprocess. Expected: " << expected_elements 
                  << ", Got: " << blob.total() << std::endl;
        throw std::runtime_error("Blob size mismatch during preprocessing.");
    }
    
    memcpy(input_data, blob.ptr<float>(0), expected_elements * sizeof(float));
}

std::vector<float> FaceRecognizer::extractFeature(const cv::Mat& aligned_face) {
    if (aligned_face.empty()) {
        std::cerr << "Error: Empty face image provided for feature extraction" << std::endl;
        return {};
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> input_data_vec(input_width_ * input_height_ * 3);
    preprocess(aligned_face, input_data_vec.data());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data_vec.data(), input_data_vec.size(), 
        input_dims.data(), input_dims.size());
    
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), &input_tensor, 1, 
        output_names_.data(), output_names_.size());
    
    auto end = std::chrono::high_resolution_clock::now();
    inference_time_ = std::chrono::duration<float, std::milli>(end - start).count();
    
    std::vector<float> feature;
    if (!outputs.empty()) {
        float* raw_output_data = outputs[0].GetTensorMutableData<float>();
        auto output_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t feature_size = 1;
        for (auto& dim : output_dims) {
            if (dim <= 0) {
                std::cerr << "[ERROR] extractFeature: Output dimension is non-positive: " << dim << std::endl;
                return {};
            }
            feature_size *= dim;
        }
        
        if (feature_size == 0) {
             std::cerr << "[ERROR] extractFeature: Calculated feature_size is zero." << std::endl;
             return {};
        }
        if (feature_size != feature_dim_) {
            std::cout << "[WARNING] extractFeature: Model output feature size (" << feature_size 
                      << ") does not match expected feature_dim_ (" << feature_dim_ 
                      << "). Using model output size." << std::endl;
        }

        feature.assign(raw_output_data, raw_output_data + feature_size);

        normalizeFeature(feature);
    } else {
        std::cerr << "[ERROR] extractFeature: Model outputs vector is empty." << std::endl;
    }
    
    return feature;
}

void FaceRecognizer::normalizeFeature(std::vector<float>& feature) const {
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
    
    float dot_product = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        dot_product += feature1[i] * feature2[i];
    }
    
    return dot_product;
}

bool FaceRecognizer::addFace(const std::string& name, const cv::Mat& aligned_face) {
    if (aligned_face.empty() || name.empty()) {
        return false;
    }
    
    std::vector<float> feature = extractFeature(aligned_face);
    if (feature.empty()) {
        return false;
    }
    
    cv::Mat thumbnail;
    cv::resize(aligned_face, thumbnail, cv::Size(64, 64));
    
    FaceFeature face_data = {name, feature, thumbnail};
    face_database_.push_back(face_data);
    
    std::cout << "Added face: " << name << " to database" << std::endl;
    return true;
}

bool FaceRecognizer::addFace(const std::string& name, const std::vector<float>& feature, const cv::Mat& thumbnail) {
    if (feature.empty() || name.empty()) {
        return false;
    }
    
    FaceFeature face_data = {name, feature, thumbnail.clone()};
    face_database_.push_back(face_data);
    
    std::cout << "Added face: " << name << " to database" << std::endl;
    return true;
}

std::pair<std::string, float> FaceRecognizer::recognize(const cv::Mat& aligned_face, float threshold) const {
    if (aligned_face.empty()) {
        return {"", 0.0f};
    }
    
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
    
    for (const auto& face : face_database_) {
        float similarity = compareFaces(feature, face.feature);
        if (similarity > best_score) {
            best_score = similarity;
            best_match = face.name;
        }
    }
    
    if (best_score >= threshold) {
        return {best_match, best_score};
    }
    
    return {"Unknown", best_score};
}

bool FaceRecognizer::saveFaceDatabase(const std::string& file_path) const {
    std::filesystem::path base_dir = std::filesystem::path(file_path).parent_path();
    if (!std::filesystem::exists(base_dir)) {
        std::filesystem::create_directories(base_dir);
    }
    
    std::ofstream index_file(file_path);
    if (!index_file.is_open()) {
        std::cerr << "Error: Could not open index file for writing: " << file_path << std::endl;
        return false;
    }
    
    int face_count = 0;
    for (const auto& face : face_database_) {
        std::filesystem::path person_dir = base_dir / face.name;
        if (!std::filesystem::exists(person_dir)) {
            std::filesystem::create_directories(person_dir);
        }
        
        std::string feature_path = (person_dir / "feature.bin").string();
        std::ofstream feature_file(feature_path, std::ios::binary);
        if (feature_file.is_open()) {
            feature_file.write(reinterpret_cast<const char*>(face.feature.data()), 
                              face.feature.size() * sizeof(float));
            feature_file.close();
        }
        
        std::string face_path = (person_dir / "face.jpg").string();
        cv::imwrite(face_path, face.thumbnail);
        
        index_file << face.name << std::endl;
        face_count++;
    }
    
    index_file.close();
    std::cout << "Saved " << face_count << " faces to directory database" << std::endl;
    return true;
}

bool FaceRecognizer::loadFaceDatabase(const std::string& file_path) {
    face_database_.clear();
    
    std::filesystem::path base_dir = std::filesystem::path(file_path).parent_path();
    std::filesystem::path index_path = file_path;
    
    if (!std::filesystem::exists(index_path)) {
        std::cerr << "Error: Database index file not found: " << file_path << std::endl;
        return false;
    }
    
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
        
        if (!std::filesystem::exists(feature_path) || !std::filesystem::exists(face_path)) {
            std::cerr << "Warning: Missing files for person: " << name << std::endl;
            continue;
        }
        
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
        
        cv::Mat thumbnail = cv::imread(face_path);
        if (thumbnail.empty()) {
            std::cerr << "Error: Could not load face image: " << face_path << std::endl;
            continue;
        }
        
        if (thumbnail.cols != 64 || thumbnail.rows != 64) {
            cv::resize(thumbnail, thumbnail, cv::Size(64, 64));
        }
        
        FaceFeature face_data = {name, feature, thumbnail};
        face_database_.push_back(face_data);
        face_count++;
    }
    
    index_file.close();
    std::cout << "Loaded " << face_count << " faces from directory database" << std::endl;
    return face_count > 0;
}

void FaceRecognizer::checkNormalizationRequirement(){
    try {
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType input_type = tensor_info.GetElementType();
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cout << "Model input type is FLOAT. Normalization to [0, 1] may be required." << std::endl;
            requires_normalization_ = true;
        } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            std::cout << "Model input type is UINT8. Normalization may not be required." << std::endl;
            requires_normalization_ = false;
        } else {
            std::cout << "Model input type is " << input_type << ". Check model documentation for preprocessing requirements." << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "Error while checking normalization requirement: " << e.what() << std::endl;
    }
}
