存在问题：不同人脸之间的相似度(点积)非常接近，很难区分不同人脸。

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

./face_recognizer_debug
Face Recognition Model Analysis Tool
Analyzing face recognition model: ../models/arcfaceresnet100-8.onnx
Number of input nodes: 1
Input #0: Name=data Dimensions=[1,3,112,112]
Number of output nodes: 1
Output #0: Name=fc1 Dimensions=[1,512]
Input #0 (data) data type: float
Output #0 (fc1) data type: float

## 模型信息参数解释 - arcfaceresnet100-8.onnx

### 模型概述
- **arcfaceresnet100-8.onnx**: 这是一个基于ResNet100架构的ArcFace人脸识别模型。ArcFace通过添加角度间隔（angular margin）来增强特征判别能力，是目前高性能人脸识别的主流算法之一。

### 输入节点解释
- **data** [1,3,112,112]:
  - 第一维"1"表示批处理大小（batch size），即一次处理一张人脸图片
  - 第二维"3"表示输入图像的通道数，即RGB三通道
  - 第三维和第四维"112,112"表示输入人脸图像的高度和宽度（像素）
  - 数据类型：float（浮点型）

### 输出节点解释
- **fc1** [1,512]:
  - 包含一个512维的特征向量（face embedding）
  - 这个特征向量是人脸的高维表示，用于人脸比对和识别
  - 通常使用余弦相似度（cosine similarity）来比较两个特征向量的相似度
  - 数据类型：float（浮点型）
  
### 使用流程
1. 人脸检测后，将检测到的人脸区域裁剪并调整为112×112像素
2. 将图像输入到模型中，获取512维特征向量
3. 通过计算特征向量之间的相似度来进行人脸验证或识别



疑问1：进行人脸检测的时候，预处理把要识别的图像缩放到模型输入的大小，两者宽纵比不同，对人脸检测和人脸对齐有影响吗？