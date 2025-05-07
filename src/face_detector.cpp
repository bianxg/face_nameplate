#include "face_detector.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include <chrono>

FaceDetector::FaceDetector(const std::string& model_path) {
    try {
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FaceDetector");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get model info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input info
        Ort::AllocatedStringPtr input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_names_str_.push_back(input_name_ptr.get());
        
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = tensor_info.GetShape();
        
        // Get output names
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            Ort::AllocatedStringPtr output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(output_name_ptr.get());
        }
        
        // Set up the input and output name pointers (points to the stored strings)
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }
        
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }
        
        // Generate anchor boxes
        generatePriorBoxes();
        
        std::cout << "Face detector initialized with model: " << model_path << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << input_shape_[i];
        }
        std::cout << "]" << std::endl;
    }
    catch(const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        throw;
    }
}

FaceDetector::~FaceDetector() {
    // Smart pointers will clean up automatically
}

std::vector<FaceInfo> FaceDetector::detect(const cv::Mat& image, float conf_threshold, float nms_threshold, int max_faces) {
    if (image.empty()) {
        std::cerr << "Error: Empty image provided for detection" << std::endl;
        return {};
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Prepare input data
    std::vector<float> input_data(input_width_ * input_height_ * 3);
    preprocess(image, input_data.data());
    
    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), 
        input_dims.data(), input_dims.size());
    
    // Run inference
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, 
        &input_names_[0], &input_tensor, 1, 
        output_names_.data(), output_names_.size());
    
    // Process outputs
    std::vector<float> loc_data, conf_data, landms_data;
    
    // Extract output data
    if (outputs.size() >= 3) {
        float* loc_ptr = outputs[0].GetTensorMutableData<float>();
        float* conf_ptr = outputs[1].GetTensorMutableData<float>();
        float* landms_ptr = outputs[2].GetTensorMutableData<float>();
        
        auto loc_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto conf_dims = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        auto landms_dims = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t loc_size = 1, conf_size = 1, landms_size = 1;
        for (auto& dim : loc_dims) loc_size *= dim;
        for (auto& dim : conf_dims) conf_size *= dim;
        for (auto& dim : landms_dims) landms_size *= dim;
        
        loc_data.assign(loc_ptr, loc_ptr + loc_size);
        conf_data.assign(conf_ptr, conf_ptr + conf_size);
        landms_data.assign(landms_ptr, landms_ptr + landms_size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    inference_time_ = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Apply post-processing
    return postprocess(loc_data, conf_data, landms_data, conf_threshold, nms_threshold, image.size(), max_faces);
}

void FaceDetector::preprocess(const cv::Mat& image, float* input_data) {
    // Resize and normalize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));
    
    // Convert to float and normalize
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32FC3);
    
    // RetinaFace uses BGR order
    for (int h = 0; h < input_height_; h++) {
        for (int w = 0; w < input_width_; w++) {
            for (int c = 0; c < 3; c++) {
                // NCHW layout
                input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                    float_mat.at<cv::Vec3f>(h, w)[c] - mean_vals_[c];
            }
        }
    }
}

void FaceDetector::generatePriorBoxes() {
    // Implementation for generating RetinaFace anchor boxes based on model output (4200 boxes)
    // For the RetinaFace model with 320x320 input
    
    std::vector<std::vector<int>> feature_maps = {
        {40, 40},  // 320/8
        {20, 20},  // 320/16
        {10, 10}   // 320/32
    };
    
    std::vector<std::vector<int>> min_sizes = {
        {16, 32}, 
        {64, 128}, 
        {256, 512}
    };
    
    std::vector<int> steps = {8, 16, 32};
    
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        int height = feature_maps[k][0];
        int width = feature_maps[k][1];
        int step = steps[k];
        
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int min_size : min_sizes[k]) {
                    // Center point of the anchor box
                    float cx = (j + 0.5f) * step / input_width_;
                    float cy = (i + 0.5f) * step / input_height_;
                    
                    // Create anchor box (center form: cx, cy, width, height)
                    float s_kx = min_size / (float)input_width_;
                    float s_ky = min_size / (float)input_height_;
                    
                    std::vector<float> box = {cx, cy, s_kx, s_ky};
                    priorbox_.push_back(box);
                }
            }
        }
    }
    
    std::cout << "Generated " << priorbox_.size() << " anchor boxes" << std::endl;
    if (priorbox_.size() != num_anchors_) {
        std::cerr << "Warning: Expected " << num_anchors_ << " anchor boxes but generated " 
                 << priorbox_.size() << std::endl;
    }
}

std::vector<FaceInfo> FaceDetector::postprocess(
    const std::vector<float>& loc_data, 
    const std::vector<float>& conf_data,
    const std::vector<float>& landms_data,
    float conf_threshold,
    float nms_threshold,
    const cv::Size& original_size,
    int max_faces) {
    
    std::vector<FaceInfo> face_list;
    
    // Scale factors for mapping back to original image
    float scale_w = original_size.width / (float)input_width_;
    float scale_h = original_size.height / (float)input_height_;
    
    // Process each anchor box
    for (size_t i = 0; i < priorbox_.size(); ++i) {
        // Get confidence score
        float score = conf_data[i * 2 + 1]; // Second column for the positive class
        
        // Filter out low confidence detections
        if (score < conf_threshold) continue;
        
        // Get anchor box (center form: cx, cy, width, height)
        const auto& anchor = priorbox_[i];
        float cx = anchor[0];
        float cy = anchor[1];
        float anchor_w = anchor[2];
        float anchor_h = anchor[3];
        
        // Decode bounding box (center form)
        float box_cx = loc_data[i * 4 + 0] * 0.1f * anchor_w + cx;
        float box_cy = loc_data[i * 4 + 1] * 0.1f * anchor_h + cy;
        float box_w = std::exp(loc_data[i * 4 + 2] * 0.2f) * anchor_w;
        float box_h = std::exp(loc_data[i * 4 + 3] * 0.2f) * anchor_h;
        
        // Convert to corner form (x1, y1, x2, y2)
        float x1 = (box_cx - box_w / 2.0f) * input_width_ * scale_w;
        float y1 = (box_cy - box_h / 2.0f) * input_height_ * scale_h;
        float x2 = (box_cx + box_w / 2.0f) * input_width_ * scale_w;
        float y2 = (box_cy + box_h / 2.0f) * input_height_ * scale_h;
        
        // Clip to image boundaries
        x1 = std::max(0.0f, std::min(x1, (float)original_size.width - 1));
        y1 = std::max(0.0f, std::min(y1, (float)original_size.height - 1));
        x2 = std::max(0.0f, std::min(x2, (float)original_size.width - 1));
        y2 = std::max(0.0f, std::min(y2, (float)original_size.height - 1));
        
        // Create face info
        FaceInfo face_info;
        face_info.score = score;
        face_info.bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
        
        // Decode landmarks (5 points)
        face_info.landmarks.resize(5);
        for (int j = 0; j < 5; ++j) {
            float lm_x = landms_data[i * 10 + j * 2] * 0.1f * anchor_w + cx;
            float lm_y = landms_data[i * 10 + j * 2 + 1] * 0.1f * anchor_h + cy;
            
            // Scale to original image size
            lm_x = lm_x * input_width_ * scale_w;
            lm_y = lm_y * input_height_ * scale_h;
            
            // Clip to image boundaries
            lm_x = std::max(0.0f, std::min(lm_x, (float)original_size.width - 1));
            lm_y = std::max(0.0f, std::min(lm_y, (float)original_size.height - 1));
            
            face_info.landmarks[j] = cv::Point2f(lm_x, lm_y);
        }
        
        face_list.push_back(face_info);
    }
    
    // Apply non-maximum suppression
    std::vector<int> keep_indices = nms(face_list, nms_threshold);
    
    // Keep only the best faces after NMS
    std::vector<FaceInfo> result;
    for (int idx : keep_indices) {
        result.push_back(face_list[idx]);
        // If we've reached the maximum number of faces, stop adding more
        if (max_faces > 0 && result.size() >= static_cast<size_t>(max_faces)) {
            break;
        }
    }
    
    return result;
}

std::vector<int> FaceDetector::nms(const std::vector<FaceInfo>& faces, float threshold) {
    if (faces.empty()) return {};
    
    // Sort by score in descending order
    std::vector<std::pair<int, float>> score_index_pairs;
    for (size_t i = 0; i < faces.size(); ++i) {
        score_index_pairs.push_back({i, faces[i].score});
    }
    
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
             [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                 return a.second > b.second;
             });
    
    std::vector<int> keep_indices;
    std::vector<bool> is_merged(faces.size(), false);
    
    for (size_t i = 0; i < score_index_pairs.size(); ++i) {
        int idx = score_index_pairs[i].first;
        if (is_merged[idx]) continue;
        
        keep_indices.push_back(idx);
        
        // Get current bbox
        cv::Rect bbox1 = faces[idx].bbox;
        float area1 = bbox1.width * bbox1.height;
        
        // Check overlap with all other bboxes
        for (size_t j = i + 1; j < score_index_pairs.size(); ++j) {
            int other_idx = score_index_pairs[j].first;
            if (is_merged[other_idx]) continue;
            
            // Get other bbox
            cv::Rect bbox2 = faces[other_idx].bbox;
            float area2 = bbox2.width * bbox2.height;
            
            // Compute intersection
            cv::Rect intersection = bbox1 & bbox2;
            if (intersection.empty()) continue;
            
            float intersect_area = intersection.width * intersection.height;
            float union_area = area1 + area2 - intersect_area;
            float iou = intersect_area / union_area;
            
            if (iou > threshold) {
                is_merged[other_idx] = true;
            }
        }
    }
    
    return keep_indices;
}
