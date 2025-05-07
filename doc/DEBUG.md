./face_detector_debug
Model Analysis Tool
Analyzing model: ../models/RetinaFace_resnet50_320.onnx
Number of input nodes: 1
Input #0: Name=input0 Dimensions=[1,3,320,320]
Number of output nodes: 3
Output #0: Name=output0 Dimensions=[1,4200,4]
Output #1: Name=819 Dimensions=[1,4200,2]
Output #2: Name=818 Dimensions=[1,4200,10]

## 模型信息参数解释

### 模型概述
- **RetinaFace_resnet50_320.onnx**: 这是一个基于ResNet50骨干网络的RetinaFace人脸检测模型，输入分辨率为320×320像素。RetinaFace是一种高效的单阶段人脸检测器。

### 输入节点解释
- **input0** [1,3,320,320]:
  - 第一维"1"表示批处理大小（batch size），即一次处理一张图片
  - 第二维"3"表示输入图像的通道数，即RGB三通道
  - 第三维和第四维"320,320"表示输入图像的高度和宽度（像素）

### 输出节点解释
- **output0** [1,4200,4]: 
  - 包含4200个候选边界框的坐标信息
  - 每个边界框有4个值，表示边界框的位置（x, y, width, height或x1, y1, x2, y2等格式）

- **819** [1,4200,2]:
  - 包含4200个候选框的置信度分数
  - 每个框有2个值，通常表示是人脸的概率和不是人脸的概率

- **818** [1,4200,10]:
  - 包含4200个候选框的面部关键点信息
  - 每个框有10个值，通常表示5个关键点（眼睛、鼻子、嘴角）的x和y坐标
