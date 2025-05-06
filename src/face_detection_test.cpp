#include "face_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

int main() {
    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    
    std::cout << "Camera opened successfully." << std::endl;
    
    // Initialize face detector
    std::string model_path = "../models/RetinaFace_resnet50_320.onnx";
    FaceDetector detector(model_path);
    
    std::cout << "Face detector initialized. Press 'q' to quit, 's' to save frame." << std::endl;
    
    // Variables for FPS calculation
    const int FPS_WINDOW_SIZE = 30;
    std::vector<float> processing_times(FPS_WINDOW_SIZE, 0.0f);
    int frame_counter = 0;
    
    // Main loop
    cv::Mat frame;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        
        // Record start time
        auto start = std::chrono::high_resolution_clock::now();
        
        // Detect faces
        std::vector<FaceInfo> faces = detector.detect(frame);
        
        // Record end time and calculate FPS
        auto end = std::chrono::high_resolution_clock::now();
        float processing_time = std::chrono::duration<float, std::milli>(end - start).count();
        processing_times[frame_counter % FPS_WINDOW_SIZE] = processing_time;
        frame_counter++;
        
        // Calculate average FPS over window
        float avg_processing_time = 0;
        for (const auto& time : processing_times) {
            avg_processing_time += time;
        }
        avg_processing_time /= FPS_WINDOW_SIZE;
        float fps = 1000.0f / avg_processing_time;
        
        // Draw detection results
        for (const auto& face : faces) {
            // Draw bounding box
            cv::rectangle(frame, face.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Draw confidence score
            std::string score_text = "Score: " + std::to_string(face.score).substr(0, 4);
            cv::putText(frame, score_text, 
                       cv::Point(face.bbox.x, face.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            
            // Draw facial landmarks
            for (const auto& landmark : face.landmarks) {
                cv::circle(frame, landmark, 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // Display performance info
        std::string performance_info = 
            "FPS: " + std::to_string(static_cast<int>(fps)) +
            " | Inference time: " + std::to_string(static_cast<int>(detector.getInferenceTime())) + "ms" +
            " | Faces: " + std::to_string(faces.size());
        
        cv::putText(frame, performance_info, cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        
        // Show the frame
        cv::imshow("Face Detection", frame);
        
        // Handle key presses
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            std::cout << "Exiting..." << std::endl;
            break;
        } else if (key == 's') { // 's' to save the current frame
            std::string filename = "../data/detection_" + std::to_string(frame_counter) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved frame to: " << filename << std::endl;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
