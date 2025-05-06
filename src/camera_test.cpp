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
    
    // Get camera properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Camera initialized successfully:" << std::endl;
    std::cout << "Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    
    // Create data directory if it doesn't exist
    std::filesystem::path data_dir = "../data";
    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directories(data_dir);
        std::cout << "Created data directory: " << std::filesystem::absolute(data_dir) << std::endl;
    }
    
    std::cout << "Press 'q' to quit, 's' to save the current frame" << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        
        // Display frame information
        frame_count++;
        std::string info = "Frame: " + std::to_string(frame_count);
        cv::putText(frame, info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Show the frame
        cv::imshow("Camera Test", frame);
        
        // Handle key presses
        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) { // 'q' or ESC to quit
            std::cout << "Exiting..." << std::endl;
            break;
        } else if (key == 's') { // 's' to save the current frame
            std::string filename = "../data/frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved frame to: " << filename << std::endl;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Camera test completed. Processed " << frame_count << " frames." << std::endl;
    return 0;
}
