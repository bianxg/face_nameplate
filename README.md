# Smart Name Tags

An AI-powered face recognition system that displays personalized namep tags below identified faces. The system uses deep learning models to detect faces, perform facial recognition, and display information in real-time.

## Sponsors

This project is sponsored by [AgileVideo](https://www.agilevideovc.com).

## Features

- Real-time face detection using RetinaFace
- Face alignment for robust recognition
- Face recognition with customizable database
- Personalized name tags displayed below faces
- Interactive enrollment mode for adding new faces
- Optimized performance with configurable detection intervals

## Prerequisites

- Ubuntu 20.04 or newer
- CMake 3.10 or newer
- OpenCV 4.x
- ONNX Runtime
- C++17 compatible compiler

## Installation

### Install Dependencies

```bash
# Install required packages
sudo apt update
sudo apt install build-essential cmake git pkg-config
sudo apt install libopencv-dev

# Install ONNX Runtime
# Option 1: Using apt (if available)
sudo apt install libonnxruntime-dev

# Option 2: Download and install from source
# Visit: https://github.com/microsoft/onnxruntime/releases
# Download and follow installation instructions

wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
tar -xzf onnxruntime-linux-x64-1.14.1.tgz

sudo mkdir -p /usr/local/include/onnxruntime
sudo cp -r onnxruntime-linux-x64-1.14.1/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-1.14.1/lib/* /usr/local/lib/
```

### Clone the Repository

```bash
git clone https://github.com/bianxg/face_nametag.git
cd face_nametag
```

## Building the Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j4
```

## Usage

The project includes several executables for different purposes:

### 1. Basic Camera Test

Verifies your camera is working correctly:

```bash
./camera_test
```

### 2. Face Detection Test

Tests the face detection functionality:

```bash
./face_detection_test
```

### 3. Face Recognition Test

Interactive program for testing face recognition:

```bash
./face_recognition_test
```

Controls:
- Press 'e' to toggle enrollment mode
- In enrollment mode, press 'n' to name a detected face
- Press 's' to save the face database
- Press 'q' or ESC to quit

### 4. Batch Enrollment

Enroll multiple faces from image files:

```bash
./batch_enrollment /path/to/images
```

### 5. Main Name Tag Application

The main application with all features:

```bash
./nameplate_main
```

Controls:
- Press 'q' or ESC to quit
- Press 'f' to toggle FPS display
- Press 'l' to toggle landmarks display
- Press 'd' to toggle detection boxes

## Directory Structure

- `src/`: Source code files
- `include/`: Header files
- `models/`: ONNX models for face detection and recognition
- `data/`: Runtime data, including the face database
- `build/`: Build output directory

## Models

The system requires two ONNX models:
1. `RetinaFace_resnet50_320.onnx` - For face detection and landmark location
2. `arcfaceresnet100-8.onnx` - For face recognition (feature extraction)

Place these models in the `models/` directory before running the applications.

https://gitee.com/wirelesser/rknn_model_zoo/blob/main/examples/RetinaFace/README.md
https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface

## Configuration

Detection interval and other parameters can be adjusted in the source code:
- `DETECT_INTERVAL`: Number of frames between face detections (default: 20)
- Camera resolution is set to 1280x720 by default

## License

MIT License
