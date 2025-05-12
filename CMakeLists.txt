cmake_minimum_required(VERSION 3.10)
project(face_nameplate)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set RPATH handling for portability of build artifacts
# This ensures that executables can find their .so dependencies when copied elsewhere.
set(CMAKE_SKIP_BUILD_RPATH FALSE)          # Generate RPATH for targets in the build tree.
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)   # Use CMAKE_INSTALL_RPATH for build tree targets as well.
                                           # This means RPATH settings for installation apply to build tree.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # Add directories from link_directories() and find_library()
                                           # (e.g., ONNX_RUNTIME_DIR/lib) to the RPATH of targets.

# Define RPATH entries for executables.
# $ORIGIN allows finding libraries relative to the executable's location.
#   - "$ORIGIN": For libraries in the same directory as the executable.
#   - "$ORIGIN/../lib": For libraries in a 'lib' subdirectory relative to the executable's directory.
# This helps locate libface_core.so if it's copied alongside executables.
# For external libraries like onnxruntime, CMAKE_INSTALL_RPATH_USE_LINK_PATH should handle them.
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")

# Find required packages
find_package(OpenCV REQUIRED)
# Update ONNXRuntime path to match your system
set(ONNX_RUNTIME_DIR "/usr/local")

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNX_RUNTIME_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

# Link directories
link_directories(
    ${ONNX_RUNTIME_DIR}/lib
)

# Add executable targets
add_executable(camera_test src/camera_test.cpp)
target_link_libraries(camera_test ${OpenCV_LIBS})

add_executable(face_detector_debug src/face_detector_debug.cpp)
target_link_libraries(face_detector_debug
    ${OpenCV_LIBS}
    onnxruntime
)

# Add face recognizer debug executable
add_executable(face_recognizer_debug src/face_recognizer_debug.cpp)
target_link_libraries(face_recognizer_debug
    ${OpenCV_LIBS}
    onnxruntime
)

# Face detection library and test
add_library(face_core SHARED
    src/face_detector.cpp
    src/face_alignment.cpp
    src/face_recognizer.cpp
)
target_link_libraries(face_core
    ${OpenCV_LIBS}
    onnxruntime
)

add_executable(face_detection_test src/face_detection_test.cpp)
target_link_libraries(face_detection_test face_core)

# Add alignment test executable
add_executable(face_alignment_test src/face_alignment_test.cpp)
target_link_libraries(face_alignment_test
    face_core
    ${OpenCV_LIBS}
    onnxruntime
)

# Face feature extraction and recognition executable
add_executable(face_recognition_test src/face_recognition_test.cpp)
target_link_libraries(face_recognition_test
    face_core
    ${OpenCV_LIBS}
    onnxruntime
)

# Add batch enrollment executable
add_executable(batch_enrollment src/batch_enrollment.cpp)
target_link_libraries(batch_enrollment
    face_core
    ${OpenCV_LIBS}
    onnxruntime
)

# Electronic Nameplate main application
add_executable(nameplate_main src/nameplate_main.cpp)
target_link_libraries(nameplate_main
    face_core
    ${OpenCV_LIBS}
    onnxruntime
)
