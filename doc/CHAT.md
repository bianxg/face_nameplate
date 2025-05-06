# 电子铭牌系统开发指南

## 0. 项目初始化与框架设置

首先，创建基本的项目目录结构：

```bash
mkdir -p project/face/src
mkdir -p project/face/include
mkdir -p project/face/models
mkdir -p project/face/build
mkdir -p project/face/data
mkdir -p project/face/doc
```

创建基本的CMake配置文件，支持OpenCV和ONNX Runtime：

## 1. 模型分析与诊断

在开始开发之前，创建一个模型诊断工具来分析ONNX模型的结构，这对于理解模型输入输出非常重要。

创建 `face_detector_debug.cpp` 用于分析模型结构：

```cpp
// 在src/face_detector_debug.cpp中
#include <iostream>
#include <filesystem>
#include <onnxruntime_cxx_api.h>

void printModelInfo(const std::string& model_path) {
    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "模型文件不存在: " << model_path << std::endl;
            return;
        }
        
        std::cout << "正在分析模型: " << model_path << std::endl;
        
        // 初始化ONNX Runtime环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelAnalyzer");
        Ort::SessionOptions session_options;
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // 获取输入信息
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "输入节点数量: " << num_input_nodes << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            char* input_name = session.GetInputName(i, allocator);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            std::vector<int64_t> input_dims = tensor_info.GetShape();
            
            std::cout << "输入 #" << i << ": 名称=" << input_name;
            std::cout << " 维度=[";
            for (size_t j = 0; j < input_dims.size(); j++) {
                if (j > 0) std::cout << ",";
                std::cout << input_dims[j];
            }
            std::cout << "]" << std::endl;
            
            allocator.Free(input_name);
        }
        
        // 获取输出信息
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "输出节点数量: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            char* output_name = session.GetOutputName(i, allocator);
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            std::vector<int64_t> output_dims = tensor_info.GetShape();
            
            std::cout << "输出 #" << i << ": 名称=" << output_name;
            std::cout << " 维度=[";
            for (size_t j = 0; j < output_dims.size(); j++) {
                if (j > 0) std::cout << ",";
                std::cout << output_dims[j];
            }
            std::cout << "]" << std::endl;
            
            allocator.Free(output_name);
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // 默认分析RetinaFace模型
    std::string model_path = "../models/RetinaFace_resnet50_320.onnx";
    
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "模型分析工具\n当前工作目录: " << std::filesystem::current_path() << std::endl;
    printModelInfo(model_path);
    
    // 如果有第二个参数，分析resnet100模型
    if (argc > 2) {
        std::string face_recognition_model = argv[2];
        std::cout << "\n分析人脸识别模型: " << face_recognition_model << std::endl;
        printModelInfo(face_recognition_model);
    } else {
        // 默认也分析人脸识别模型
        std::string face_recognition_model = "../models/resnet100.onnx";
        if (std::filesystem::exists(face_recognition_model)) {
            std::cout << "\n分析人脸识别模型: " << face_recognition_model << std::endl;
            printModelInfo(face_recognition_model);
        }
    }
    
    return 0;
}
```

在CMakeLists.txt中添加此可执行文件：

```cmake
add_executable(face_detector_debug src/face_detector_debug.cpp)
target_link_libraries(face_detector_debug
    ${OpenCV_LIBS}
    onnxruntime
)
```

在开始实现功能前，先下载并准备好必要的模型：

1. RetinaFace_resnet50_320.onnx - 用于人脸检测和关键点定位
2. resnet100.onnx - 用于人脸特征提取和识别

将这些模型放在项目的models目录下。

## 2. 基础摄像头测试

请创建一个基础的摄像头测试程序 `camera_test.cpp`，验证以下功能：
- 摄像头初始化
- 图像采集
- 基本GUI显示
- 键盘交互（按q退出，按s保存图像）

确认摄像头正常工作后，我们将进行下一步。

## 3. 人脸检测测试

基于摄像头测试程序，创建 `face_detection_test.cpp`，集成以下功能：
- 使用RetinaFace_resnet50_320.onnx模型进行人脸检测
- 提取人脸关键点（facial landmarks）
- 在UI上显示检测框和关键点
- 显示检测时间和置信度

**验证要点**：确认人脸检测正常工作，能够准确标记人脸边界框和关键点。

## 4. 人脸对齐实现

创建人脸对齐模块 `face_alignment.h/cpp`，实现以下功能：
- 基于RetinaFace检测的关键点进行人脸对齐
- 提供一致大小和位置的标准化人脸图像
- 支持旋转、缩放和裁剪操作

创建测试程序 `face_alignment_test.cpp` 验证对齐功能是否有效。

**验证要点**：对比对齐前后的人脸图像，确认对齐后的图像面部特征位置一致。

## 5. 人脸特征提取与识别

创建人脸特征提取和识别模块 `face_recognizer.h/cpp`，使用resnet100.onnx模型：
- 提取对齐后人脸的特征向量
- 计算特征向量之间的相似度
- 定义匹配阈值

创建测试程序 `face_recognition_test.cpp`，集成前面的人脸检测和对齐功能，并添加：
- 人脸特征提取
- 与数据库中的人脸进行匹配
- 显示识别结果和相似度

**验证要点**：系统能够正确识别已知人脸，并显示准确的相似度。

## 6. 人脸数据库管理

创建人脸数据库管理模块 `face_database.h/cpp`：
- 支持添加、删除、查询人脸信息
- 存储人脸特征向量和对应的名称
- 提供持久化存储功能

创建两个工具程序：
1. `face_enrollment.cpp` - 通过摄像头实时录入人脸
2. `batch_enrollment.cpp` - 从照片文件批量录入人脸

**验证要点**：确认数据库能正确保存和加载人脸信息，两种录入方式都能有效工作。

## 7. 电子铭牌主程序

最后，创建电子铭牌主程序 `nameplate_main.cpp`，集成所有功能：
- 实时人脸检测和对齐
- 人脸识别与匹配
- 显示识别结果和个人信息
- 支持录入新人脸
- 美观的UI界面

**验证要点**：
- 识别准确度达到95%以上
- 处理速度满足实时要求（至少15FPS）
- UI界面友好，显示完整信息

## 实现注意事项

1. 人脸检测使用 RetinaFace_resnet50_320.onnx 模型
2. 人脸对齐基于检测的5点关键点进行
3. 人脸识别使用 resnet100.onnx 模型
4. UI提示使用英文，避免显示乱码
5. 注重代码模块化，便于后续维护和扩展
6. 各阶段都添加详细日志，便于调试和问题定位

## 系统优化建议

1. 考虑添加人脸活体检测，防止照片欺骗
2. 实现多线程处理，提高系统响应速度
3. 添加用户友好的人脸数据库管理界面
4. 支持不同光照条件下的人脸识别
5. 添加简单的人脸表情识别功能

通过逐步完成上述每个阶段，并在每个阶段后进行充分验证，我们可以构建一个稳定可靠的电子铭牌系统。
