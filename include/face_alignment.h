#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "face_detector.h"

class FaceAlignment {
public:
    // Default constructor
    FaceAlignment();
    
    // Constructor with target face size
    FaceAlignment(const cv::Size& target_size, bool keep_aspect_ratio = true);
    
    // Align a single face based on 5-point landmarks
    cv::Mat align(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    
    // Align a face using FaceInfo from detector
    cv::Mat align(const cv::Mat& image, const FaceInfo& face_info);
    
    // Align multiple faces
    std::vector<cv::Mat> alignMultiple(const cv::Mat& image, const std::vector<FaceInfo>& faces);
    
    // Get/Set the target size for aligned faces
    cv::Size getTargetSize() const { return target_size_; }
    void setTargetSize(const cv::Size& size) { target_size_ = size; }
    
    // Set reference landmarks (ideal positions in the target face)
    void setReferenceLandmarks(const std::vector<cv::Point2f>& reference);
    
private:
    // Target size for aligned faces
    cv::Size target_size_;
    
    // Whether to keep aspect ratio during alignment
    bool keep_aspect_ratio_;
    
    // Reference landmarks (ideal positions in the target face)
    std::vector<cv::Point2f> reference_landmarks_;
    
    // Initialize reference landmarks
    void initReferenceLandmarks();
    
    // Calculate similarity transform (scale, rotation, translation)
    cv::Mat getSimilarityTransform(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst);
};
