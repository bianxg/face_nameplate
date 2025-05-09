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

## 1. 先下载并准备好必要的模型：

模型下载地址
https://gitee.com/wirelesser/rknn_model_zoo/blob/main/examples/RetinaFace/README.md
https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface

1. RetinaFace_resnet50_320.onnx - 用于人脸检测和关键点定位
2. arcfaceresnet100-8.onnx - 用于人脸特征提取和识别

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
- 显示识别结果（电子铭牌显示在人脸的下方）
- 美观的UI界面
- 跟踪功能：检测是每几帧检测一次（如20帧），降低CPU使用率但保持电子铭牌的显示

### 电子铭牌实现进展

在实现电子铭牌主程序时，经历了以下步骤和改进：

1. **逐步实现与调试**
   - 首先，实现了最基础的摄像头采集和显示，确保硬件工作正常
   - 添加人脸检测功能，验证模型正常工作
   - 集成人脸识别和铭牌显示，完成基础功能
   - 通过摄像头分辨率设置为1280x720提高图像质量

2. **稳定性问题排查**
   - 初始版本尝试使用OpenCV跟踪器进行高级跟踪，但出现段错误
   - 通过逐步简化系统，定位问题根源
   - 最终使用简单的跟踪策略，通过跳帧检测实现稳定性

3. **性能优化**
   - 设置`DETECT_INTERVAL = 20`，实现每20帧进行一次人脸检测和识别
   - 在非检测帧中保持显示先前的检测结果
   - 显著提高了系统帧率，同时降低CPU使用率

4. **用户界面设计**
   - 实现简洁有效的电子铭牌设计，显示在人脸下方
   - 不同颜色区分已知人脸和未知人脸
   - 显示FPS和检测间隔信息，便于性能监控

### 当前工作模式

当前的系统使用"跳帧检测"策略，其工作原理是：

1. 每20帧进行一次完整的人脸检测和识别
2. 在非检测帧中，继续显示之前检测到的人脸和识别结果
3. 这种策略显著降低了CPU使用率，同时保持了良好的用户体验

检测和识别同步进行，每次检测帧都会执行人脸识别。系统没有实现复杂的跟踪算法，而是简单地保持显示最近一次的检测结果，这种做法虽然简单，但非常稳定可靠。

### 后续改进方向

1. **智能跟踪算法**
   - 实现基于运动预测的位置更新
   - 在非检测帧中使用速度估计来更新人脸位置

2. **分离检测和识别频率**
   - 可以更频繁地进行人脸检测（如每3帧）
   - 降低识别频率（如每5次检测进行一次识别）
   - 这样可以保持位置更新的同时降低计算负担

3. **多人同时识别优化**
   - 改进多人场景下的识别优先级
   - 为不同人员设置不同的识别频率

4. **UI美化与增强**
   - 添加渐变背景和圆角效果
   - 显示人脸缩略图
   - 优化字体和布局

通过这些进一步的优化，可以在保持系统稳定性的基础上，提高用户体验和性能表现.

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

通过逐步完成上述每个阶段，并在每个阶段后进行充分验证，我们可以构建一个稳定可靠的电子铭牌系统.


# 视频会议MCU上电子铭牌应用时机分析

## 在混屏前应用电子铭牌

### 优势
1. **个性化处理**：可对每个参会者单独进行人脸识别和铭牌生成，实现个性化效果
2. **资源分配灵活**：可根据参会者重要程度分配不同的处理资源和频率
3. **准确定位**：铭牌位置更精确，不受混屏后缩放和布局变化影响
4. **分布式处理**：可在参会者终端就完成铭牌生成，减轻MCU负担
5. **选择性处理**：只对需要显示铭牌的参会者进行处理，节约资源

### 劣势
1. **资源消耗**：需要对多路视频流分别进行处理，总体计算量较大
2. **实现复杂**：需要与MCU混屏系统深度集成
3. **一致性挑战**：不同参会者的铭牌风格可能不一致

## 在混屏后应用电子铭牌

### 优势
1. **集中处理**：只需处理一次最终合成画面，计算资源集中
2. **实现简单**：作为视频处理流水线的最后一步，架构简单
3. **统一风格**：所有铭牌样式可统一管理
4. **适应布局**：能适应动态变化的会议布局

### 劣势
1. **识别难度**：混屏后的合成画面中人脸大小不一，小尺寸人脸识别难度增加
2. **图像质量**：混屏处理可能导致图像质量下降，影响人脸检测准确率
3. **处理延迟**：在视频处理管线末端增加处理，可能增加端到端延迟
4. **负载集中**：系统负载集中在MCU的最后处理阶段

## 建议方案

根据视频会议系统特点，建议采用**混屏前应用电子铭牌**的方式：

1. 在各参会者视频流进入MCU时就进行人脸识别和铭牌生成
2. 针对不同角色参会者设置不同的处理优先级（主讲人优先）
3. 将铭牌信息作为元数据与视频流一起传递给混屏模块
4. 混屏模块在合成画面时根据元数据正确放置铭牌

这种方案既能保证铭牌的准确性，又能灵活适应会议系统的资源分配需求。


准备将电子铭牌库应用到MCU中，充分考虑设计一些新的接口，方便在MCU使用，尽量减少数据的转换和拷贝，保证视频会议的实时性。备注：MCU内部解码出来的图像使用yuv数据表示。