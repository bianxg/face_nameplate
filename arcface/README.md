https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface

# Python Face Recognition Process (Input: Aligned Face, Output: Feature Vector)

## 1. Read the Aligned Face Image
```python
img = cv2.imread(image_path)
```

## 2. Ensure Image Size is 112x112 (ArcFace Standard Input Size)
```python
if img.shape[0] != 112 or img.shape[1] != 112:
    img = cv2.resize(img, (112, 112))
```

## 3. Convert Color Channels from BGR to RGB
```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

## 4. Transpose Dimensions to Format Required by Model (Channel, Height, Width)
```python
aligned = np.transpose(img, (2, 0, 1))  # From (112,112,3) to (3,112,112)
```

## 5. Prepare Model Input (Add Batch Dimension and Convert to float32 Type)
```python
input_blob = np.expand_dims(aligned, axis=0).astype(np.float32)  # Change to (1,3,112,112)
```

## 6. Get Input Tensor Name
```python
input_name = model.get_inputs()[0].name  # Usually 'data'
```

## 7. Run Model Inference
```python
outputs = model.run(None, {input_name: input_blob})
embedding = outputs[0][0]  # Get the first batch of the first output
```

## 8. Apply L2 Normalization to the Feature Vector
```python
embedding = sklearn.preprocessing.normalize(embedding.reshape(1, -1)).flatten()
```

## 9. Calculate Similarity Between Two Feature Vectors (Optional)
```python
# Euclidean distance
dist = np.sum(np.square(feature1 - feature2))
# Cosine similarity
sim = np.dot(feature1, feature2.T)
```

## Notes
- Python version does not normalize pixel values, uses original pixel values in [0-255] range
- Only performs channel order conversion (BGR→RGB) and dimension transposition
- Relies on sklearn's normalize function for L2 normalization of feature vectors

The above is the complete Python face recognition process, from reading an aligned face image to generating a standardized facial feature vector.

# C++ Face Recognition Process (Input: Aligned Face, Output: Feature Vector)

## 1. Read and Resize Image
```cpp
cv::Mat resized_face;
if (face.rows != input_height_ || face.cols != input_width_) {
    resized_face = cv::resize(face, cv::Size(input_width_, input_height_));
} else {
    resized_face = face;
}
```

## 2. Convert to Floating Point Image
```cpp
cv::Mat float_mat;
resized_face.convertTo(float_mat, CV_32FC3);
// Pixel values range [0, 255]
```

## 3. Perform Pixel Value Normalization (Important Difference)
```cpp
// ArcFace standard normalization: (pixel - 127.5) / 128.0
float_mat.convertTo(float_mat, CV_32FC3, 1.0f/128.0f, -127.5f/128.0f);
// Normalized pixel values range [-1, 1]
```

## 4. Rearrange Dimensions to NCHW Format (Preserve BGR Channel Order)
```cpp
for (int c = 0; c < 3; c++) {
    for (int h = 0; h < input_height_; h++) {
        for (int w = 0; w < input_width_; w++) {
            input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                float_mat.at<cv::Vec3f>(h, w)[c]; // BGR order
        }
    }
}
```

## 5. Create ORT Inference Input
```cpp
std::vector<float> input_tensor_values(batch_size * input_channels_ * input_height_ * input_width_);
// Copy preprocessed data to input_tensor_values

// Create input tensor
Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_tensor_values.data(), input_tensor_values.size(),
    input_shape_.data(), input_shape_.size());
```

## 6. Run Model Inference
```cpp
std::vector<const char*> input_names = {input_name_.c_str()};
std::vector<const char*> output_names = {output_name_.c_str()};

// Execute inference
auto output_tensors = session_->Run(
    Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
    output_names.data(), 1);
```

## 7. Retrieve Output Feature Vector
```cpp
float* output_data = output_tensors[0].GetTensorMutableData<float>();
std::vector<float> feature(output_data, output_data + feature_dim_);
```

## 8. Apply L2 Normalization to the Feature Vector
```cpp
void FaceRecognizer::normalizeFeature(std::vector<float>& feature) const {
    float norm = 0.0f;
    for (const auto& val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-6f) {
        for (auto& val : feature) {
            val /= norm;
        }
    }
}
```

## 9. Calculate Similarity Between Two Feature Vectors (Optional)
```cpp
float FaceRecognizer::compareFaces(const std::vector<float>& feature1, const std::vector<float>& feature2) const {
    float similarity = 0.0f;
    // Calculate cosine similarity (dot product)
    for (size_t i = 0; i < feature1.size(); i++) {
        similarity += feature1[i] * feature2[i];
    }
    return similarity;
}
```

## Key Differences
1. **Pixel Value Normalization**: C++ version uses the `(pixel - 127.5) / 128.0` formula to normalize pixel values to [-1,1] range, while Python version uses original pixel values in [0,255] range
2. **Channel Order**: C++ version maintains BGR channel order, Python version converts to RGB
3. **Feature Normalization**: C++ version implements L2 normalization manually, Python version uses sklearn library functions

These differences, especially pixel normalization, are likely the main causes of result discrepancies.

Conclusion:
The C++ version should remove the pixel value normalization step.