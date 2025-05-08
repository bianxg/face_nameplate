#include "face_detector.h"
#include "face_alignment.h"
#include "face_recognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <regex>
#include <fstream>
#include <unordered_map>
#include <cmath>

// Extract name from filename (remove extension)
std::string getNameFromFilename(const std::string& filepath) {
    std::filesystem::path path(filepath);
    return path.stem().string();
}

// Function to check if file is an image
bool isImageFile(const std::string& filename) {
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff";
}

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path_or_pattern>" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " /path/to/images/" << std::endl;
        std::cout << "  " << argv[0] << " /path/to/images/*.jpg" << std::endl;
        return 1;
    }
    
    std::string path_pattern = argv[1];
    std::filesystem::path db_file = "../data/faces/face_database.txt";
    
    // Initialize models
    std::string detector_model = "../models/RetinaFace_resnet50_320.onnx";
    std::string recognizer_model = "../models/arcfaceresnet100-8.onnx";
    
    std::cout << "Initializing face detection and recognition models..." << std::endl;
    
    // Create data directory
    std::filesystem::path data_dir = "../data/faces";
    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directories(data_dir);
    }
    
    // Initialize models
    FaceDetector detector(detector_model);
    FaceAlignment aligner(cv::Size(112, 112));
    FaceRecognizer recognizer(recognizer_model);
    
    // Load existing database if available
    if (std::filesystem::exists(db_file)) {
        recognizer.loadFaceDatabase(db_file);
        std::cout << "Loaded existing face database with " 
                  << recognizer.getFaces().size() << " faces." << std::endl;
    }
    
    // Find all matching image files
    std::vector<std::filesystem::path> image_files;
    
    // Check if pattern has a wildcard
    if (path_pattern.find('*') != std::string::npos) {
        // Extract directory from pattern
        std::filesystem::path dir_path = std::filesystem::path(path_pattern).parent_path();
        if (dir_path.empty()) dir_path = ".";
        
        // Extract filename pattern
        std::string filename_pattern = std::filesystem::path(path_pattern).filename().string();
        
        // Convert glob pattern to regex
        std::string regex_pattern = std::regex_replace(filename_pattern, std::regex("\\*"), ".*");
        std::regex file_regex(regex_pattern);
        
        // Iterate through directory
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && 
                isImageFile(entry.path().string()) &&
                std::regex_match(entry.path().filename().string(), file_regex)) {
                image_files.push_back(entry.path());
            }
        }
    } else {
        // Check if it's a directory or file
        if (std::filesystem::is_directory(path_pattern)) {
            // Get all image files in directory
            for (const auto& entry : std::filesystem::directory_iterator(path_pattern)) {
                if (entry.is_regular_file() && isImageFile(entry.path().string())) {
                    image_files.push_back(entry.path());
                }
            }
        } else if (std::filesystem::exists(path_pattern) && 
                  isImageFile(path_pattern)) {
            // Single file
            image_files.push_back(path_pattern);
        }
    }
    
    if (image_files.empty()) {
        std::cout << "No image files found matching pattern: " << path_pattern << std::endl;
        return 1;
    }
    
    std::cout << "Found " << image_files.size() << " images to process." << std::endl;
    
    // Process images
    int processed = 0;
    int successful = 0;
    int errors = 0;
    
    // Get names of existing faces for checking duplicates
    std::unordered_map<std::string, bool> existing_faces;
    for (const auto& face : recognizer.getFaces()) {
        existing_faces[face.name] = true;
    }

    // 存储所有已录入的名字和特征
    std::vector<std::string> enrolled_names;
    std::vector<std::vector<float>> enrolled_features;
    
    // Process each image
    for (const auto& image_path : image_files) {
        processed++;
        std::string name = getNameFromFilename(image_path.string());
        
        std::cout << "[" << processed << "/" << image_files.size() << "] "
                  << "Processing " << image_path.filename().string() << " (name: " << name << ")";
        
        if (existing_faces.find(name) != existing_faces.end()) {
            std::cout << " [OVERWRITE]";
        }
        std::cout << std::endl;
        
        try {
            // Load image
            cv::Mat image = cv::imread(image_path.string());
            if (image.empty()) {
                throw std::runtime_error("Failed to load image");
            }

            // Detect faces
            std::vector<FaceInfo> faces = detector.detect(image);
            
            if (faces.empty()) {
                throw std::runtime_error("No faces detected");
            }
            
            std::cout << "  Found " << faces.size() << " faces in image" << std::endl;
            
            // Select the best (largest) face if multiple detected
            FaceInfo best_face;
            float max_area = 0;
            
            for (const auto& face : faces) {
                float area = face.bbox.width * face.bbox.height;
                if (area > max_area) {
                    max_area = area;
                    best_face = face;
                }
            }
            
            // Align the face
            cv::Mat aligned_face = aligner.align(image, best_face);
            if (aligned_face.empty()) {
                throw std::runtime_error("Failed to align face");
            }

            // Extract feature vector
            std::vector<float> feature = recognizer.extractFeature(aligned_face);

            // Create directory for this person
            std::filesystem::path person_dir = data_dir / name;
            if (!std::filesystem::exists(person_dir)) {
                std::filesystem::create_directories(person_dir);
            }
            
            // Save aligned face image
            std::string face_path = (person_dir / "original.jpg").string();
            cv::imwrite(face_path, aligned_face);
            
            // Extract and save feature data
            std::string feature_path = (person_dir / "feature.bin").string();
            std::ofstream feature_file(feature_path, std::ios::binary);
            if (feature_file.is_open()) {
                feature_file.write(reinterpret_cast<const char*>(feature.data()), 
                                  feature.size() * sizeof(float));
                feature_file.close();
            } else {
                throw std::runtime_error("Failed to save feature data");
            }

            // 保存特征用于后续距离计算
            enrolled_names.push_back(name);
            enrolled_features.push_back(feature);
            
            // Create and save thumbnail
            cv::Mat thumbnail;
            cv::resize(aligned_face, thumbnail, cv::Size(64, 64));
            std::string thumbnail_path = (person_dir / "face.jpg").string();
            cv::imwrite(thumbnail_path, thumbnail);
            
            // Update our tracking of existing faces
            existing_faces[name] = true;
            successful++;
            std::cout << "  Successfully processed face: " << name << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << std::endl;
            errors++;
        }
    }

    // 打印每个人与其他人的余弦相似度
    std::cout << "\nPairwise feature similarity among enrolled faces:\n";
    for (size_t i = 0; i < enrolled_names.size(); ++i) {
        for (size_t j = i + 1; j < enrolled_names.size(); ++j) {
            // 余弦相似度
            double dot = 0.0;
            for (size_t k = 0; k < enrolled_features[i].size(); ++k) {
                dot += enrolled_features[i][k] * enrolled_features[j][k];
            }

            std::cout << "  " << enrolled_names[i] << " <-> " << enrolled_names[j]
                      << " | Cosine: " << dot << std::endl;
        }
    }
    
    // Rebuild the database from saved face data
    std::cout << "Rebuilding face database..." << std::endl;
    
    // Create a fresh recognizer
    FaceRecognizer new_recognizer(recognizer_model);
    
    // Iterate through all person directories
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            std::string feature_path = (entry.path() / "feature.bin").string();
            std::string face_path = (entry.path() / "face.jpg").string();
            
            if (std::filesystem::exists(feature_path) && std::filesystem::exists(face_path)) {
                // Load feature
                std::vector<float> feature(512); // feature dimension
                std::ifstream feature_file(feature_path, std::ios::binary);
                if (feature_file.is_open()) {
                    feature_file.read(reinterpret_cast<char*>(feature.data()), 
                                     512 * sizeof(float));
                    feature_file.close();
                    
                    // Load face thumbnail
                    cv::Mat thumbnail = cv::imread(face_path);
                    if (!thumbnail.empty()) {
                        // Add to the new database
                        new_recognizer.addFace(name, feature, thumbnail);
                    }
                }
            }
        }
    }
    
    // Save the rebuilt database
    new_recognizer.saveFaceDatabase(db_file);
    
    // Print summary
    std::cout << "\nBatch enrollment complete." << std::endl;
    std::cout << "Total processed: " << processed << std::endl;
    std::cout << "Successfully enrolled: " << successful << std::endl;
    std::cout << "Errors: " << errors << std::endl;
    std::cout << "Database saved to: " << db_file.string() << std::endl;
    
    return 0;
}
