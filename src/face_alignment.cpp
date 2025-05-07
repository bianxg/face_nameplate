#include "face_alignment.h"
#include <iostream>

FaceAlignment::FaceAlignment() 
    : target_size_(112, 112), keep_aspect_ratio_(true) {
    initReferenceLandmarks();
}

FaceAlignment::FaceAlignment(const cv::Size& target_size, bool keep_aspect_ratio) 
    : target_size_(target_size), keep_aspect_ratio_(keep_aspect_ratio) {
    initReferenceLandmarks();
}

void FaceAlignment::initReferenceLandmarks() {
    // Standard reference points for 112x112 face image
    // These points represent ideal positions for:
    // left eye, right eye, nose tip, left mouth corner, right mouth corner
    reference_landmarks_ = {
        cv::Point2f(38.2946f, 51.6963f),  // Left eye
        cv::Point2f(73.5318f, 51.5014f),  // Right eye
        cv::Point2f(56.0252f, 71.7366f),  // Nose tip
        cv::Point2f(41.5493f, 92.3655f),  // Left mouth corner
        cv::Point2f(70.7299f, 92.2041f)   // Right mouth corner
    };
    
    // If target size is not 112x112, scale the reference landmarks accordingly
    if (target_size_.width != 112 || target_size_.height != 112) {
        float scale_x = target_size_.width / 112.0f;
        float scale_y = target_size_.height / 112.0f;
        
        for (auto& point : reference_landmarks_) {
            point.x *= scale_x;
            point.y *= scale_y;
        }
    }
}

void FaceAlignment::setReferenceLandmarks(const std::vector<cv::Point2f>& reference) {
    if (reference.size() != 5) {
        std::cerr << "Reference landmarks must contain exactly 5 points" << std::endl;
        return;
    }
    reference_landmarks_ = reference;
}

cv::Mat FaceAlignment::getSimilarityTransform(
    const std::vector<cv::Point2f>& src, 
    const std::vector<cv::Point2f>& dst) {
    
    // We need at least 2 points to compute similarity transform
    if (src.size() != dst.size() || src.size() < 2) {
        std::cerr << "Invalid point sets for similarity transform" << std::endl;
        return cv::Mat();
    }
    
    // Use OpenCV's estimateAffinePartial2D which computes a similarity transform
    // (translation, rotation, and uniform scale)
    cv::Mat transform = cv::estimateAffinePartial2D(src, dst);
    
    // If the estimation failed, fall back to our manual implementation
    if (transform.empty()) {
        std::cerr << "OpenCV's estimateAffinePartial2D failed, using manual implementation" << std::endl;
        
        // Calculate centroids
        cv::Point2f src_centroid(0, 0), dst_centroid(0, 0);
        for (size_t i = 0; i < src.size(); i++) {
            src_centroid += src[i];
            dst_centroid += dst[i];
        }
        src_centroid = src_centroid * (1.0f / src.size());
        dst_centroid = dst_centroid * (1.0f / dst.size());
        
        // Subtract centroids
        std::vector<cv::Point2f> src_centered, dst_centered;
        for (size_t i = 0; i < src.size(); i++) {
            src_centered.push_back(src[i] - src_centroid);
            dst_centered.push_back(dst[i] - dst_centroid);
        }

        // Calculate the parameters of similarity transform:
        // scale, rotation, and translation
        float a = 0, b = 0;
        for (size_t i = 0; i < src_centered.size(); i++)
        {
            a += src_centered[i].x * dst_centered[i].x + src_centered[i].y * dst_centered[i].y;
            b += src_centered[i].y * dst_centered[i].x - src_centered[i].x * dst_centered[i].y;
        }
        float norm_squared = 0;
        for (const auto &p : src_centered)
        {
            norm_squared += p.x * p.x + p.y * p.y;
        }

        float scale = std::sqrt(a * a + b * b) / norm_squared;
        float theta = std::atan2(b, a);

        // Create 2x3 transformation matrix
        transform = cv::Mat::zeros(2, 3, CV_32F);
        transform.at<float>(0, 0) = std::cos(theta) * scale;
        transform.at<float>(0, 1) = -std::sin(theta) * scale;
        transform.at<float>(1, 0) = std::sin(theta) * scale;
        transform.at<float>(1, 1) = std::cos(theta) * scale;
        transform.at<float>(0, 2) = dst_centroid.x - (transform.at<float>(0, 0) * src_centroid.x + 
                                               transform.at<float>(0, 1) * src_centroid.y);
        transform.at<float>(1, 2) = dst_centroid.y - (transform.at<float>(1, 0) * src_centroid.x + 
                                               transform.at<float>(1, 1) * src_centroid.y);
    }
    
    return transform;
}

cv::Mat FaceAlignment::align(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5 || image.empty()) {
        std::cerr << "Invalid input for face alignment" << std::endl;
        return cv::Mat();
    }
    
    // Get the similarity transformation
    cv::Mat transform = getSimilarityTransform(landmarks, reference_landmarks_);
    if (transform.empty()) {
        std::cerr << "Failed to compute alignment transform" << std::endl;
        return cv::Mat();
    }
    
    // Apply the transformation to align the face
    cv::Mat aligned;
    cv::warpAffine(image, aligned, transform, target_size_, 
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    
    return aligned;
}

cv::Mat FaceAlignment::align(const cv::Mat& image, const FaceInfo& face_info) {
    return align(image, face_info.landmarks);
}

std::vector<cv::Mat> FaceAlignment::alignMultiple(
    const cv::Mat& image, 
    const std::vector<FaceInfo>& faces) {
    
    std::vector<cv::Mat> aligned_faces;
    for (const auto& face : faces) {
        cv::Mat aligned = align(image, face);
        if (!aligned.empty()) {
            aligned_faces.push_back(aligned);
        }
    }
    
    return aligned_faces;
}
