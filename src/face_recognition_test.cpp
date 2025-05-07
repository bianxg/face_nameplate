#include "face_detector.h"
#include "face_alignment.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

// Function to draw a nice-looking recognition result box
void drawRecognitionBox(cv::Mat& image, const std::string& name, float score, 
                       const cv::Rect& face_rect, const cv::Scalar& color) {
    // Draw name and score
    char buf[64];
    snprintf(buf, sizeof(buf), "%s (%.2f)", name.c_str(), score);
    std::string display_text = buf;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, 
                                        0.5, 1, &baseline);
    
    // Draw the box around the face
    cv::rectangle(image, face_rect, color, 2);
    
    // Draw the background rectangle for the name
    cv::Rect text_rect(face_rect.x, face_rect.y - text_size.height - 10, 
                      text_size.width + 10, text_size.height + 10);
    cv::rectangle(image, text_rect, color, -1);
    
    // Draw the name text with black color for better visibility on colored backgrounds
    cv::putText(image, display_text, 
               cv::Point(face_rect.x + 5, face_rect.y - 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

int main() {
    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    
    // Get and display camera resolution
    int cam_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int cam_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Camera opened successfully with resolution: " << cam_width << "x" << cam_height << std::endl;
    
    // Optional: Set a specific camera resolution
    // Uncomment these lines to force a specific resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    // std::cout << "Camera resolution set to: " 
    //           << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
    //           << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    
    // Initialize face detector
    std::string detector_model_path = "../models/RetinaFace_mobile320.onnx";
    FaceDetector detector(detector_model_path);
    
    // Initialize face alignment
    FaceAlignment aligner(cv::Size(112, 112));
    
    // Initialize face recognizer with the correct model path
    std::string recognizer_model_path = "../models/arcfaceresnet100-8.onnx";
    FaceRecognizer recognizer(recognizer_model_path);
    
    // Create data directories if they don't exist
    std::filesystem::path data_dir = "../data/faces";
    std::filesystem::path db_file = "../data/faces/face_database.txt";
    
    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directories(data_dir);
        std::cout << "Created faces directory: " << std::filesystem::absolute(data_dir) << std::endl;
    }
    
    // Try to load an existing face database
    bool db_loaded = false;
    if (std::filesystem::exists(db_file)) {
        db_loaded = recognizer.loadFaceDatabase(db_file);
    }
    
    if (!db_loaded) {
        std::cout << "No existing face database found or failed to load." << std::endl;
    }
    
    // Create windows for display
    cv::namedWindow("Face Recognition", cv::WINDOW_NORMAL);
    cv::namedWindow("Face Database", cv::WINDOW_NORMAL);
    
    // Set initial window sizes
    cv::resizeWindow("Face Recognition", 1280, 720);
    cv::resizeWindow("Face Database", 640, 480);
    
    // Flag to control database enrollment mode
    bool enrollment_mode = false;
    std::string enrollment_name = "";
    bool name_entry_active = false;
    
    // Set ArcFace recommended recognition threshold
    float recognition_threshold = 0.5f;

    std::cout << "Face recognition system initialized." << std::endl;
    std::cout << "Press 'q' or 'ESC' to quit, 'e' to toggle enrollment mode, 's' to save the database" << std::endl;
    
    // Main loop
    cv::Mat frame;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        
        // Create a display copy of the frame
        cv::Mat display_frame = frame.clone();
        
        // Detect faces
        std::vector<FaceInfo> faces = detector.detect(frame);
        
        // Process each face for recognition
        for (const auto& face : faces) {
            // Make sure we have landmarks
            if (face.landmarks.size() != 5) {
                continue;
            }
            
            // Align the face
            cv::Mat aligned_face = aligner.align(frame, face);
            if (aligned_face.empty()) {
                continue;
            }
            
            if (enrollment_mode) {
                // In enrollment mode, draw a special marker
                cv::rectangle(display_frame, face.bbox, cv::Scalar(0, 165, 255), 2);
                if (!name_entry_active) {
                    cv::putText(display_frame, "Press 'n' to name this face", 
                               cv::Point(face.bbox.x, face.bbox.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 1);
                }
            } else {
                // Normal recognition mode
                // Extract features and recognize
                std::vector<float> feature = recognizer.extractFeature(aligned_face);

                // Debug: Print similarity scores for all database faces
                const auto& face_db = recognizer.getFaces();
                std::cout << "Compare scores: ";
                for (const auto& db_face : face_db) {
                    float sim = recognizer.compareFaces(feature, db_face.feature);
                    std::cout << db_face.name << ":" << sim << " ";
                }
                std::cout << std::endl;

                std::pair<std::string, float> result = recognizer.recognize(feature, recognition_threshold);
                
                // Draw recognition result
                cv::Scalar color;
                if (result.first == "Unknown") {
                    color = cv::Scalar(0, 0, 255); // Red for unknown
                } else {
                    color = cv::Scalar(0, 255, 0); // Green for known
                }
                
                drawRecognitionBox(display_frame, result.first, result.second, face.bbox, color);
            }
            
            // Draw facial landmarks
            for (const auto& landmark : face.landmarks) {
                cv::circle(display_frame, landmark, 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // Display enrollment mode status and UI
        if (enrollment_mode) {
            // Draw a semi-transparent overlay at the top of the screen
            cv::Mat overlay = display_frame(cv::Rect(0, 0, display_frame.cols, 80)).clone();
            cv::addWeighted(overlay, 0.5, cv::Scalar(0, 0, 0), 0.5, 0, display_frame(cv::Rect(0, 0, display_frame.cols, 80)));
            
            if (name_entry_active) {
                // Display name entry UI
                std::string name_prompt = "Enter name: " + enrollment_name;
                // Add blinking cursor for feedback
                if ((cv::getTickCount() / 15) % 2 == 0) {
                    name_prompt += "|";
                }
                
                cv::putText(display_frame, name_prompt, cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                cv::putText(display_frame, "Press Enter to confirm, ESC to cancel", cv::Point(10, 60),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1); // Brighter text
            } else {
                cv::putText(display_frame, "ENROLLMENT MODE", cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 255), 2); // Brighter orange
                
                cv::putText(display_frame, "Press 'n' to name a face, 'e' to exit enrollment mode", 
                           cv::Point(10, 60),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1); // Brighter text
            }
        } else {
            std::string status = "RECOGNITION MODE";
            cv::putText(display_frame, status, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
            // Show key commands
            cv::putText(display_frame, "Press 'e' for enrollment mode, 's' to save database, 'q' to quit", 
                       cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1); // Brighter text
        }
        
        // Add inference info
        std::string info = "Detection time: " + std::to_string(static_cast<int>(detector.getInferenceTime())) + "ms | "
                         + "Recognition time: " + std::to_string(static_cast<int>(recognizer.getInferenceTime())) + "ms";
        cv::putText(display_frame, info, cv::Point(10, display_frame.rows - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Show the main display frame
        cv::imshow("Face Recognition", display_frame);
        
        // Create and show a display of the face database
        const auto& face_db = recognizer.getFaces();
        if (!face_db.empty()) {
            int max_faces_per_row = 5;
            int thumbnail_size = 64;
            int rows = (face_db.size() + max_faces_per_row - 1) / max_faces_per_row;
            
            cv::Mat db_display = cv::Mat::zeros(rows * (thumbnail_size + 30), 
                                              max_faces_per_row * thumbnail_size, CV_8UC3);
            
            for (size_t i = 0; i < face_db.size(); i++) {
                int row = i / max_faces_per_row;
                int col = i % max_faces_per_row;
                
                cv::Rect roi(col * thumbnail_size, row * (thumbnail_size + 30), 
                            thumbnail_size, thumbnail_size);
                
                if (!face_db[i].thumbnail.empty()) {
                    // Copy thumbnail to display
                    face_db[i].thumbnail.copyTo(db_display(roi));
                    
                    // Add name label
                    cv::putText(db_display, face_db[i].name, 
                               cv::Point(roi.x, roi.y + roi.height + 20),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
            }
            
            cv::imshow("Face Database", db_display);
        } else {
            // Show an empty database message
            cv::Mat empty_display = cv::Mat::zeros(200, 400, CV_8UC3);
            cv::putText(empty_display, "Face Database Empty", 
                       cv::Point(100, 100),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            cv::imshow("Face Database", empty_display);
        }
        
        // Handle key presses
        int key = cv::waitKey(1);
        
        // Common key handling
        if (key == 'q' || key == 27) { // 'q' or ESC to quit/cancel
            if (name_entry_active) {
                // Just cancel name entry
                name_entry_active = false;
                enrollment_name = "";
            } else {
                // Exit program
                std::cout << "Exiting..." << std::endl;
                break;
            }
        } else if (key == 's' && !name_entry_active) { // 's' to save database
            if (recognizer.saveFaceDatabase(db_file)) {
                std::cout << "Face database saved successfully." << std::endl;
            }
        } else if (key == 'e' && !name_entry_active) { // 'e' to toggle enrollment mode
            enrollment_mode = !enrollment_mode;
            if (enrollment_mode) {
                std::cout << "Enrollment mode activated." << std::endl;
            } else {
                std::cout << "Recognition mode activated." << std::endl;
                enrollment_name = "";
            }
        } else if (key == '+' || key == '=') { // '+' to increase threshold
            recognition_threshold += 0.02f;
            std::cout << "Threshold: " << recognition_threshold << std::endl;
        } else if (key == '-' || key == '_') { // '-' to decrease threshold
            recognition_threshold -= 0.02f;
            std::cout << "Threshold: " << recognition_threshold << std::endl;
        }
        
        // Enrollment-specific key handling
        if (enrollment_mode) {
            if (name_entry_active) {
                if (key == 13) { // Enter to confirm name
                    if (!enrollment_name.empty()) {
                        // Check if we have a valid face to enroll
                        if (faces.size() == 1) {
                            cv::Mat aligned_face = aligner.align(frame, faces[0]);
                            if (!aligned_face.empty()) {
                                // Check if this name already exists in the database
                                bool overwriting = false;
                                for (const auto& face : recognizer.getFaces()) {
                                    if (face.name == enrollment_name) {
                                        overwriting = true;
                                        break;
                                    }
                                }
                                
                                if (recognizer.addFace(enrollment_name, aligned_face)) {
                                    // Show success message on screen with appropriate text
                                    cv::Mat success_overlay = display_frame.clone();
                                    std::string message = overwriting ? 
                                        "Updated existing face: " + enrollment_name : 
                                        "Successfully enrolled: " + enrollment_name;
                                    
                                    cv::putText(success_overlay, message, 
                                               cv::Point(display_frame.cols/2 - 150, display_frame.rows/2),
                                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                                    cv::imshow("Face Recognition", success_overlay);
                                    cv::waitKey(1000); // Show message for 1 second
                                    
                                    // Save image for reference
                                    std::string img_path = "../data/faces/" + enrollment_name + "/original.jpg";
                                    std::filesystem::create_directories(std::filesystem::path(img_path).parent_path());
                                    cv::imwrite(img_path, aligned_face);
                                    
                                    // Save database immediately if we're updating
                                    if (overwriting) {
                                        recognizer.saveFaceDatabase(db_file);
                                    }
                                }
                            }
                        } else {
                            // Show error message
                            cv::Mat error_overlay = display_frame.clone();
                            cv::putText(error_overlay, "Error: Please ensure only one face is visible", 
                                       cv::Point(display_frame.cols/2 - 200, display_frame.rows/2),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                            cv::imshow("Face Recognition", error_overlay);
                            cv::waitKey(1000); // Show message for 1 second
                        }
                        
                        // Exit name entry mode
                        name_entry_active = false;
                        enrollment_name = "";
                    }
                } else if (key >= 32 && key <= 126) { // Printable ASCII characters
                    enrollment_name += (char)key;
                } else if ((key == 8 || key == 127) && !enrollment_name.empty()) { // Backspace
                    enrollment_name.pop_back();
                }
            } else if (key == 'n') { // 'n' to start name entry
                name_entry_active = true;
                enrollment_name = "";
            }
        }
    }
    
    // Save the database before exiting
    if (!recognizer.getFaces().empty()) {
        if (recognizer.saveFaceDatabase(db_file)) {
            std::cout << "Face database saved before exit." << std::endl;
        }
    }
    
    // Release resources
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
