#include "face_detector.h"
#include "face_alignment.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

int main() {
    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    
    std::cout << "Camera opened successfully." << std::endl;
    
    // Initialize face detector
    std::string model_path = "../models/RetinaFace_mobile320.onnx";
    FaceDetector detector(model_path);
    
    // Initialize face alignment
    FaceAlignment aligner(cv::Size(112, 112));
    
    std::cout << "Face detector and aligner initialized." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save aligned faces." << std::endl;
    
    // Create a window for display
    cv::namedWindow("Face Detection", cv::WINDOW_NORMAL);
    cv::namedWindow("Aligned Faces", cv::WINDOW_NORMAL);
    
    // Create data directory if it doesn't exist
    std::filesystem::path data_dir = "../data/aligned";
    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directories(data_dir);
        std::cout << "Created aligned faces directory: " << std::filesystem::absolute(data_dir) << std::endl;
    }
    
    // Main loop
    cv::Mat frame;
    int frame_counter = 0;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        
        // Detect faces
        std::vector<FaceInfo> faces = detector.detect(frame);
        
        // Draw detection results
        cv::Mat display_frame = frame.clone();
        for (const auto& face : faces) {
            // Draw bounding box
            cv::rectangle(display_frame, face.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Draw facial landmarks
            for (const auto& landmark : face.landmarks) {
                cv::circle(display_frame, landmark, 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // Display performance info
        std::string info = "Faces detected: " + std::to_string(faces.size()) + 
                          " | Inference time: " + std::to_string(static_cast<int>(detector.getInferenceTime())) + "ms";
        cv::putText(display_frame, info, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        
        cv::imshow("Face Detection", display_frame);
        
        // Prepare aligned faces display - create a default empty display
        cv::Mat aligned_display = cv::Mat::zeros(224, 224, CV_8UC3);
        
        // Process and display aligned faces
        if (faces.empty()) {
            cv::putText(aligned_display, "No faces detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        } else {
            try {
                std::vector<cv::Mat> aligned_faces;
                
                for (const auto& face : faces) {
                    // Skip if landmarks are not valid
                    if (face.landmarks.size() != 5) {
                        continue;
                    }
                    
                    // Create a simple crop as fallback
                    cv::Rect safe_bbox = face.bbox;
                    
                    // Make sure the bbox is within image bounds
                    safe_bbox.x = std::max(0, safe_bbox.x);
                    safe_bbox.y = std::max(0, safe_bbox.y);
                    safe_bbox.width = std::min(safe_bbox.width, frame.cols - safe_bbox.x);
                    safe_bbox.height = std::min(safe_bbox.height, frame.rows - safe_bbox.y);
                    
                    // Simple crop and resize as fallback
                    cv::Mat fallback_face;
                    if (safe_bbox.width > 0 && safe_bbox.height > 0) {
                        cv::Mat cropped = frame(safe_bbox);
                        cv::resize(cropped, fallback_face, cv::Size(112, 112));
                    }
                    
                    // Try to align this face
                    cv::Mat aligned = aligner.align(frame, face);
                    
                    if (!aligned.empty()) {
                        // Convert to BGR if it's grayscale
                        if (aligned.channels() == 1) {
                            cv::cvtColor(aligned, aligned, cv::COLOR_GRAY2BGR);
                        }
                        aligned_faces.push_back(aligned);
                    } else if (!fallback_face.empty()) {
                        // Use fallback if alignment failed
                        aligned_faces.push_back(fallback_face);
                    }
                }
                
                // If we have any aligned faces, display them
                if (!aligned_faces.empty()) {
                    int max_faces_per_row = 3;
                    int rows = (aligned_faces.size() + max_faces_per_row - 1) / max_faces_per_row;
                    int face_size = aligner.getTargetSize().width;
                    
                    // Create a properly sized canvas
                    aligned_display = cv::Mat::zeros(rows * face_size, 
                                                    std::min((int)aligned_faces.size(), max_faces_per_row) * face_size, 
                                                    CV_8UC3);
                    
                    for (size_t i = 0; i < aligned_faces.size(); i++) {
                        int row = i / max_faces_per_row;
                        int col = i % max_faces_per_row;
                        
                        cv::Rect roi(col * face_size, row * face_size, face_size, face_size);
                        
                        // Copy aligned face to display if valid
                        if (roi.x >= 0 && roi.y >= 0 && 
                            roi.x + roi.width <= aligned_display.cols && 
                            roi.y + roi.height <= aligned_display.rows &&
                            aligned_faces[i].cols == face_size && 
                            aligned_faces[i].rows == face_size &&
                            !aligned_faces[i].empty()) {
                            
                            aligned_faces[i].copyTo(aligned_display(roi));
                            
                            // Add a label
                            cv::putText(aligned_display, "Face " + std::to_string(i+1), 
                                      cv::Point(roi.x + 5, roi.y + 15),
                                      cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                                      cv::Scalar(0, 255, 0), 1);
                        }
                    }
                } else {
                    cv::putText(aligned_display, "Alignment failed for all faces", cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
            } catch (const std::exception& e) {
                cv::putText(aligned_display, "Error in face alignment process", cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // Show aligned faces
        cv::imshow("Aligned Faces", aligned_display);
        
        // Handle key presses
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            std::cout << "Exiting..." << std::endl;
            break;
        } else if (key == 's') { // 's' to save aligned faces and original frame
            frame_counter++;
            // Save original frame
            std::string orig_filename = "../data/aligned/original_" + std::to_string(frame_counter) + ".jpg";
            cv::imwrite(orig_filename, frame);
            
            // Save the alignment display
            std::string aligned_filename = "../data/aligned/aligned_display_" + std::to_string(frame_counter) + ".jpg";
            cv::imwrite(aligned_filename, aligned_display);
            
            std::cout << "Saved images to: " << orig_filename << " and " << aligned_filename << std::endl;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
