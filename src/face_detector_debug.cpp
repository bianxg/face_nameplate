#include <iostream>
#include <filesystem>
#include <onnxruntime/onnxruntime_cxx_api.h>

void printModelInfo(const std::string& model_path) {
    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file does not exist: " << model_path << std::endl;
            return;
        }
        
        std::cout << "Analyzing model: " << model_path << std::endl;
        
        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelAnalyzer");
        Ort::SessionOptions session_options;
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get input information
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            // Use updated API method for getting input name
            Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
            std::string input_name = input_name_ptr.get(); // Convert to std::string
            
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            std::vector<int64_t> input_dims = tensor_info.GetShape();
            
            std::cout << "Input #" << i << ": Name=" << input_name;
            std::cout << " Dimensions=[";
            for (size_t j = 0; j < input_dims.size(); j++) {
                if (j > 0) std::cout << ",";
                std::cout << input_dims[j];
            }
            std::cout << "]" << std::endl;
            
            // No need to free - managed by AllocatedStringPtr
        }
        
        // Get output information
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "Number of output nodes: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            // Use updated API method for getting output name
            Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            std::string output_name = output_name_ptr.get(); // Convert to std::string
            
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            std::vector<int64_t> output_dims = tensor_info.GetShape();
            
            std::cout << "Output #" << i << ": Name=" << output_name;
            std::cout << " Dimensions=[";
            for (size_t j = 0; j < output_dims.size(); j++) {
                if (j > 0) std::cout << ",";
                std::cout << output_dims[j];
            }
            std::cout << "]" << std::endl;
            
            // No need to free - managed by AllocatedStringPtr
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default RetinaFace model path
    std::string model_path = "../models/RetinaFace_resnet50_320.onnx";
    
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Model Analysis Tool\nCurrent working directory: " << std::filesystem::current_path() << std::endl;
    printModelInfo(model_path);
    
    return 0;
}
