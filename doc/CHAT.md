# Electronic Nameplate System Development Guide

## 0. Project Initialization and Framework Setup

First, create the basic project directory structure:

```bash
mkdir -p project/face/src
mkdir -p project/face/include
mkdir -p project/face/models
mkdir -p project/face/build
mkdir -p project/face/data
mkdir -p project/face/doc
```

Create a basic CMake configuration file to support OpenCV and ONNX Runtime:

## 1. Download and prepare the necessary models:

1. RetinaFace_resnet50_320.onnx - For face detection and landmark localization
2. arcfaceresnet100-8.onnx - For face feature extraction and recognition

Place these models in the project's models directory.

## 2. Basic Camera Test

Create a basic camera test program `camera_test.cpp` to verify the following functionalities:
- Camera initialization
- Image acquisition
- Basic GUI display
- Keyboard interaction (press q to exit, press s to save images)

After confirming the camera works properly, we will proceed to the next step.

## 3. Face Detection Test

Based on the camera test program, create `face_detection_test.cpp` to integrate the following functionalities:
- Use RetinaFace_resnet50_320.onnx model for face detection
- Extract facial landmarks
- Display detection boxes and landmarks on the UI
- Show detection time and confidence

**Validation Points**: Ensure face detection works properly and accurately marks face bounding boxes and landmarks.

## 4. Face Alignment Implementation

Create a face alignment module `face_alignment.h/cpp` to implement the following functionalities:
- Perform face alignment based on facial landmarks detected by RetinaFace
- Provide standardized face images with consistent size and position
- Support rotation, scaling, and cropping operations

Create a test program `face_alignment_test.cpp` to verify the alignment functionality.

**Validation Points**: Compare pre- and post-alignment face images to ensure consistent facial feature positions in aligned images.

## 5. Face Feature Extraction and Recognition

Create a face feature extraction and recognition module `face_recognizer.h/cpp` using the resnet100.onnx model:
- Extract feature vectors from aligned face images
- Calculate similarity between feature vectors
- Define matching thresholds

Create a test program `face_recognition_test.cpp` to integrate previous face detection and alignment functionalities, and add:
- Face feature extraction
- Matching with faces in the database
- Display recognition results and similarity

**Validation Points**: Ensure the system correctly recognizes known faces and displays accurate similarity.

## 6. Face Database Management

Create a face database management module `face_database.h/cpp`:
- Support adding, deleting, and querying face information
- Store face feature vectors and corresponding names
- Provide persistent storage functionality

Create two utility programs:
1. `face_enrollment.cpp` - Real-time face enrollment via camera
2. `batch_enrollment.cpp` - Batch face enrollment from photo files

**Validation Points**: Ensure the database correctly saves and loads face information, and both enrollment methods work effectively.

## 7. Electronic Nameplate Main Program

Finally, create the electronic nameplate main program `nameplate_main.cpp` to integrate all functionalities:
- Real-time face detection and alignment
- Face recognition and matching
- Display recognition results (electronic nameplate displayed below the face)
- Aesthetic UI interface
- Tracking functionality: detect every few frames (e.g., 20 frames) to reduce CPU usage while maintaining nameplate display

### Electronic Nameplate Implementation Progress

During the implementation of the electronic nameplate main program, the following steps and improvements were made:

1. **Gradual Implementation and Debugging**
   - Initially implemented basic camera acquisition and display to ensure hardware works properly
   - Added face detection functionality to verify the model works correctly
   - Integrated face recognition and nameplate display to complete basic functionality
   - Improved image quality by setting camera resolution to 1280x720

2. **Stability Issue Troubleshooting**
   - The initial version attempted to use OpenCV trackers for advanced tracking but encountered segmentation faults
   - Simplified the system step by step to locate the root cause
   - Ultimately used a simple tracking strategy with frame skipping detection for stability

3. **Performance Optimization**
   - Set `DETECT_INTERVAL = 20` to perform face detection and recognition every 20 frames
   - Maintain display of previous detection results during non-detection frames
   - Significantly improved system frame rate while reducing CPU usage

4. **User Interface Design**
   - Implemented a simple and effective electronic nameplate design displayed below the face
   - Used different colors to distinguish between known and unknown faces
   - Displayed FPS and detection interval information for performance monitoring

### Current Working Mode

The current system uses a "frame skipping detection" strategy, which works as follows:

1. Perform complete face detection and recognition every 20 frames
2. During non-detection frames, continue displaying previously detected faces and recognition results
3. This strategy significantly reduces CPU usage while maintaining a good user experience

Detection and recognition are synchronized, with face recognition performed on every detection frame. The system does not implement complex tracking algorithms but simply maintains the display of the most recent detection results. This approach, while simple, is very stable and reliable.

### Future Improvement Directions

1. **Intelligent Tracking Algorithm**
   - Implement position updates based on motion prediction
   - Use velocity estimation to update face positions in non-detection frames

2. **Separate Detection and Recognition Frequencies**
   - Perform face detection more frequently (e.g., every 3 frames)
   - Reduce recognition frequency (e.g., perform recognition every 5 detections)
   - This maintains position updates while reducing computational load

3. **Multi-Person Recognition Optimization**
   - Improve recognition priorities in multi-person scenarios
   - Set different recognition frequencies for different individuals

4. **UI Enhancement and Beautification**
   - Add gradient backgrounds and rounded corners
   - Display face thumbnails
   - Optimize fonts and layouts

Through these further optimizations, we can improve user experience and performance while maintaining system stability.

## Implementation Considerations

1. Use RetinaFace_resnet50_320.onnx model for face detection
2. Perform face alignment based on 5-point facial landmarks from detection
3. Use resnet100.onnx model for face recognition
4. Use English for UI prompts to avoid encoding issues
5. Focus on code modularity for easier maintenance and expansion
6. Add detailed logging at each stage for debugging and issue localization

## System Optimization Suggestions

1. Consider adding face liveness detection to prevent photo spoofing
2. Implement multi-threading to improve system response time
3. Add user-friendly face database management interface
4. Support face recognition under different lighting conditions
5. Add simple facial expression recognition functionality

By gradually completing each stage above and thoroughly validating after each phase, we can build a stable and reliable electronic nameplate system.
