#include <iostream>
#include <filesystem>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

void printModelInfo(const std::string& model_path) {
    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file does not exist: " << model_path << std::endl;
            return;
        }
        
        std::cout << "Analyzing face recognition model: " << model_path << std::endl;
        
        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RecognizerAnalyzer");
        Ort::SessionOptions session_options;
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get input information
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
            std::string input_name = input_name_ptr.get();
            
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
        }
        
        // Get output information
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "Number of output nodes: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            std::string output_name = output_name_ptr.get();
            
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
        }

        // Print additional details about data types
        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
            
            std::string type_str;
            switch (elem_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: type_str = "float"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: type_str = "uint8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: type_str = "int8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: type_str = "uint16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: type_str = "int16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: type_str = "int32"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: type_str = "int64"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: type_str = "double"; break;
                default: type_str = "unknown";
            }
            
            Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
            std::cout << "Input #" << i << " (" << input_name_ptr.get() << ") data type: " << type_str << std::endl;
        }
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
            
            std::string type_str;
            switch (elem_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: type_str = "float"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: type_str = "uint8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: type_str = "int8"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: type_str = "uint16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: type_str = "int16"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: type_str = "int32"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: type_str = "int64"; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: type_str = "double"; break;
                default: type_str = "unknown";
            }
            
            Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            std::cout << "Output #" << i << " (" << output_name_ptr.get() << ") data type: " << type_str << std::endl;
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default resnet model path
    std::string model_path = "../models/face_recognition.onnx";
    
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Face Recognition Model Analysis Tool\n";
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    printModelInfo(model_path);
    
    return 0;
}
