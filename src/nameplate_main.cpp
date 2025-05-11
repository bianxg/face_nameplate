#include "face_detector.h"
#include "face_alignment.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

// Function to draw a simple nameplate below the face
void drawNameplate(cv::Mat& image, const cv::Rect& face_bbox, 
                  const std::string& name, float confidence) {
    try {
        // Nameplate dimensions
        int plate_width = std::max(150, face_bbox.width);
        int plate_height = 40;
        
        // Position below the face
        int plate_x = face_bbox.x + face_bbox.width/2 - plate_width/2;
        int plate_y = face_bbox.y + face_bbox.height + 10;
        
        // Ensure within image boundaries
        plate_x = std::max(0, std::min(plate_x, image.cols - plate_width - 1));
        plate_y = std::max(0, std::min(plate_y, image.rows - plate_height - 1));
        
        // Safety check
        if (plate_x < 0 || plate_y < 0 || 
            plate_x + plate_width >= image.cols || 
            plate_y + plate_height >= image.rows) {
            return;
        }
        
        // Create the plate rectangle
        cv::Rect plate_rect(plate_x, plate_y, plate_width, plate_height);
        
        // Background color based on recognition
        cv::Scalar bg_color = (name == "Unknown") ? 
                           cv::Scalar(0, 0, 200) :  // Red for unknown
                           cv::Scalar(0, 200, 0);   // Green for known
        
        // Draw plate background
        cv::rectangle(image, plate_rect, bg_color, -1);
        
        // Add border
        cv::rectangle(image, plate_rect, cv::Scalar(255, 255, 255), 1);
        
        // Display text with confidence
        std::string display_text = name + " (" + std::to_string(int(confidence * 100)) + "%)";
        
        // Calculate text position to center it
        int text_width = cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, 
                                        0.6, 2, nullptr).width;
        int text_x = plate_x + (plate_width - text_width) / 2;
        int text_y = plate_y + 28;  // Vertically centered
        
        // Draw text
        cv::putText(image, display_text, cv::Point(text_x, text_y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
    } catch (const std::exception& e) {
        std::cerr << "Error drawing nameplate: " << e.what() << std::endl;
    }
}

// Structure to represent a tracked face
struct TrackedFace {
    cv::Rect bbox;                  // Current bounding box
    std::string name;               // Recognized name
    float confidence;               // Recognition confidence
    std::vector<cv::Point2f> landmarks; // Facial landmarks
    int frames_since_detection;     // How many frames since last detection
};

int main() {
    try {
        // Initialize camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }
        
        // Get original camera resolution
        int original_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int original_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "Original camera resolution: " << original_width << "x" << original_height << std::endl;
        
        // Set camera resolution to 720p
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        
        // Verify the new resolution
        int new_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int new_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "Camera resolution set to: " << new_width << "x" << new_height << std::endl;
        
        // Initialize face detector
        std::string detector_model_path = "../models/RetinaFace_mobile320.onnx";
        std::cout << "Initializing face detector..." << std::endl;
        FaceDetector detector(detector_model_path);
        
        // Initialize face alignment and recognition
        std::cout << "Initializing face alignment and recognition..." << std::endl;
        FaceAlignment aligner(cv::Size(112, 112));
        
        std::string recognizer_model_path = "../models/w600k_mbf.onnx";
        FaceRecognizer recognizer(recognizer_model_path);
        
        // Load face database
        std::string db_file = "../data/faces/face_database.txt";
        if (std::filesystem::exists(db_file)) {
            bool success = recognizer.loadFaceDatabase(db_file);
            if (success) {
                std::cout << "Loaded face database with " << recognizer.getFaces().size() << " faces." << std::endl;
            } else {
                std::cout << "Failed to load face database." << std::endl;
            }
        } else {
            std::cout << "No face database found at: " << db_file << std::endl;
        }
        
        // Create window
        cv::namedWindow("Face Recognition Test", cv::WINDOW_NORMAL);
        cv::resizeWindow("Face Recognition Test", 1280, 720);
        
        // Configure detection/recognition intervals
        const int DETECT_INTERVAL = 20;  // Only run detection every 3 frames
        int frame_count = 0;
        
        // Create a vector to store tracked faces
        std::vector<TrackedFace> tracked_faces;
        
        // Performance tracking
        auto last_time = std::chrono::high_resolution_clock::now();
        float fps = 0.0f;
        
        std::cout << "System ready. Press 'q' to quit." << std::endl;
        
        // Main loop - with face detection and recognition
        cv::Mat frame;
        while (true) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) break;
            
            // Measure FPS
            auto current_time = std::chrono::high_resolution_clock::now();
            float time_diff = std::chrono::duration<float, std::milli>(current_time - last_time).count();
            last_time = current_time;
            fps = 0.9f * fps + 0.1f * (1000.0f / time_diff);  // Smoothed FPS
            
            // Create display copy
            cv::Mat display = frame.clone();
            
            // Increment frame counter
            frame_count++;
            bool do_detection = (frame_count % DETECT_INTERVAL == 0);
            
            if (do_detection) {
                // Run face detection on this frame
                std::vector<FaceInfo> detected_faces = detector.detect(frame,0.7, 0.4, 3);
                
                // Clear the tracked faces list and update with new detections
                tracked_faces.clear();
                
                // Process each detected face
                for (const auto& face : detected_faces) {
                    // Skip invalid faces
                    if (face.landmarks.size() != 5) continue;
                    
                    // Perform alignment and recognition
                    cv::Mat aligned_face = aligner.align(frame, face);
                    if (!aligned_face.empty()) {
                        std::vector<float> feature = recognizer.extractFeature(aligned_face);
                        std::pair<std::string, float> result = recognizer.recognize(feature);
                        
                        // Create tracked face
                        TrackedFace tracked_face;
                        tracked_face.bbox = face.bbox;
                        tracked_face.landmarks = face.landmarks;
                        tracked_face.name = result.first;
                        tracked_face.confidence = result.second;
                        tracked_face.frames_since_detection = 0;
                        
                        tracked_faces.push_back(tracked_face);
                    }
                }
            }
            else {
                // For frames where we don't do detection, we'll just track existing faces
                // This is a very simple approach - just increment the counter
                for (auto& face : tracked_faces) {
                    face.frames_since_detection++;
                }
            }
            
            // Draw all tracked faces
            for (const auto& face : tracked_faces) {
                // Draw bounding box
                cv::Scalar box_color = (face.name == "Unknown") ? 
                                     cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(display, face.bbox, box_color, 2);
                
                // Draw landmarks
                for (const auto& point : face.landmarks) {
                    cv::circle(display, point, 2, cv::Scalar(0, 0, 255), -1);
                }
                
                // Draw nameplate
                drawNameplate(display, face.bbox, face.name, face.confidence);
            }
            
            // Show FPS and detection interval
            cv::putText(display, "FPS: " + std::to_string(int(fps)), 
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 
                        0.7, cv::Scalar(0, 255, 0), 2);
            
            cv::putText(display, "Detection every " + std::to_string(DETECT_INTERVAL) + " frames", 
                        cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 
                        0.6, cv::Scalar(200, 200, 200), 1);
            
            // Display the frame
            cv::imshow("Face Recognition Test", display);
            
            // Check for key press
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                std::cout << "Exiting..." << std::endl;
                break;
            }
        }
        
        // Cleanup
        cap.release();
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
