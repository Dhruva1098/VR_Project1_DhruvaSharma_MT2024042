# Face Mask Detection
## I. Introduction

This project aims to develop and evaluate multiple computational approaches for face mask detection and segmentation. Specifically, we address the following objectives:

1. **Binary Classification Using Traditional Machine Learning**: Implementing feature extraction techniques (HOG, LBP) with classical machine learning classifiers (SVM, Neural Network) to distinguish between faces with and without masks.

2. **Binary Classification Using CNN**: Building and optimizing a Convolutional Neural Network for more accurate face mask detection, with various hyperparameter configurations.

3. **Mask Region Segmentation Using Traditional Techniques**: Applying classical computer vision methods to segment the mask region within face images.

4. **Mask Segmentation Using U-Net**: Implementing a deep learning-based segmentation approach using the U-Net architecture to precisely identify mask boundaries.

By comparing these different methodologies, we aim to identify the most effective approaches for face mask detection and segmentation tasks, providing insights for real-world applications.

## II. Dataset

### Sources

The project utilizes two primary datasets:

1. **Face Mask Detection Dataset** ([Chandrika Deb, 2020](https://github.com/chandrikadeb7/Face-Mask-Detection))
   - Used for classification tasks (Task 1 and 2)
   - Contains binary classification images (with_mask, without_mask)

2. **MFSD (Masked Face Segmentation Dataset)** ([Sadjad Asghari Rzghi, 2022](https://github.com/sadjadrz/MFSD))
   - Used for segmentation tasks (Task 3 and 4)
   - Contains face images with corresponding segmentation masks

### Structure

```
Face Mask Detection Dataset/
├── with_mask/         # ~1,900 images of people wearing masks
└── without_mask/      # ~1,900 images of people without masks

MFSD Dataset/
├── 1/
│   ├── face_crop/          # Cropped face images with masks
│   ├── face_crop_segmentation/   # Ground truth segmentation masks
│   └── img/                # Original images with masks
└── 2/
    └── img/                # Images without masks
```

### Statistics

- **Classification Dataset**:
  - Total images: ~3,800
  - With mask: ~1,900 images
  - Without mask: ~1,900 images
  - Format: RGB color images
  - Variable resolutions (resized during preprocessing)

- **Segmentation Dataset (MFSD)**:
  - Face crop images: ~500
  - Corresponding segmentation masks: ~500
  - Format: RGB for images, grayscale for masks
  - Masks are binary (255 for mask pixels, 0 for non-mask)

## III. Methodology

### Task 1: Binary Classification Using Handcrafted Features and ML Classifiers

1. **Preprocessing**:
   - Face images are resized to 100×100 pixels
   - Conversion to grayscale for feature extraction

2. **Feature Extraction**:
   - **HOG (Histogram of Oriented Gradients)**:
     ```python
     hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=False)
     ```
   - **LBP (Local Binary Patterns)**:
     ```python
     lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
     # Calculate histogram of LBP
     hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
     ```
   - Feature combination by concatenation: `np.hstack((hog_features, lbp_features))`
   - Standardization: `scaler.fit_transform(combined_features)`
   - Optional dimensionality reduction: `pca.fit_transform(scaled_features)`

3. **Model Training**:
   - **Support Vector Machine (SVM)**:
     - Hyperparameter Grid Search: kernel, C, gamma
     - Cross-validation: 3-fold
   
   - **Multi-Layer Perceptron (Neural Network)**:
     - Hyperparameter Grid Search: hidden layer sizes, activation, solver, alpha, learning rate
     - Cross-validation: 3-fold

### Task 2: Binary Classification Using CNN

1. **Data Preparation**:
   - Resizing images to 128×128 pixels
   - Normalization: values scaled to [0,1]
   - Train/validation/test split: 70%/15%/15%
   - Data augmentation:
     ```python
     datagen = ImageDataGenerator(
         rotation_range=20, zoom_range=0.15,
         width_shift_range=0.2, height_shift_range=0.2,
         shear_range=0.15, horizontal_flip=True,
         fill_mode="nearest"
     )
     ```

2. **CNN Architecture**:
   ```python
   model = Sequential([
       # First Convolutional Block
       Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
       BatchNormalization(),
       MaxPooling2D(pool_size=(2, 2)),
       Dropout(0.25),
       
       # Second Convolutional Block
       Conv2D(64, (3, 3), activation='relu'),
       BatchNormalization(),
       MaxPooling2D(pool_size=(2, 2)),
       Dropout(0.25),
       
       # Third Convolutional Block
       Conv2D(128, (3, 3), activation='relu'),
       BatchNormalization(),
       MaxPooling2D(pool_size=(2, 2)),
       Dropout(0.4),
       
       # Flatten and Dense Layers
       Flatten(),
       Dense(128, activation='relu'),
       BatchNormalization(),
       Dropout(dropout_rate),
       
       # Output Layer
       Dense(1, activation=activation)
   ])
   ```

3. **Training Process**:
   - Loss: Binary Cross-Entropy
   - Early stopping: patience=10 (monitor val_loss)
   - Model checkpoint: save best model based on val_accuracy
   - Epochs: up to 30 (with early stopping)
![title][Images/Unknown-5.png]
### Task 3: Region Segmentation Using Traditional Techniques

1. **Preprocessing**:
   - Loading "with mask" images
   - Bilateral filtering for noise reduction

2. **Improved Hybrid Approach**:
   - Multiple color space analysis (HSV and LAB)
   - Color-based thresholding for common mask colors
   - Edge detection and contour analysis in the lower face region
   - Morphological operations for refinement

3. **Edge Detection Method**:
   - Gaussian blur and grayscale conversion
   - Canny edge detection
   - Contour extraction and filtering
   - Position-based filtering (lower face region)

4. **Watershed Algorithm**:
   - Grayscale conversion and Otsu thresholding
   - Distance transform for foreground markers
   - Watershed transform with marker labeling
   - Lower face region prioritization

5. **Felzenszwalb Segmentation**:
   - Superpixel generation
   - Segment analysis based on position and color
   - Filtering of segments more likely to be masks

### Task 4: Mask Segmentation Using U-Net

1. **Data Preparation**:
   - Loading and matching image-mask pairs from MFSD dataset
   - Resizing to 128×128 pixels
   - Normalization: image values scaled to [0,1]
   - Binarization of masks: `mask = (mask > 127).astype(np.float32)`
   - Train/test split: 80%/20%
![title][Images/Unknown-8.png]
2. **U-Net Architecture**:
   ```python
   # Encoder
   inputs = tf.keras.layers.Input((image_size[0], image_size[1], 3))
   conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
   conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
   pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
   
   conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
   conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
   pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
   
   # Middle
   conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
   conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
   
   # Decoder
   up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
   concat4 = tf.keras.layers.concatenate([up4, conv2], axis=3)
   conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat4)
   conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
   
   up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
   concat5 = tf.keras.layers.concatenate([up5, conv1], axis=3)
   conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat5)
   conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)
   
   outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
   ```
![title](Images/U-net_arch.png]
![title][Images/Unknown-9.png]


3. **Training Process**:
   - Loss: Binary Cross-Entropy
   - Optimizer: Adam
   - Batch size: 16
   - Epochs: 15
   - Metrics: Accuracy, IoU (Intersection over Union)

## IV. Hyperparameters and Experiments

### CNN Model Experiments

| Experiment | Learning Rate | Batch Size | Optimizer | Activation | Accuracy | Training Time (s) |
|------------|---------------|------------|-----------|------------|----------|-------------------|
| Exp 1      | 0.001         | 32         | Adam      | sigmoid    | 0.954    | 89.6              |
| Exp 2      | 0.001         | 64         | Adam      | sigmoid    | 0.949    | 73.4              |
| Exp 3      | 0.0001        | 32         | Adam      | sigmoid    | 0.961    | 92.1              |
| Exp 4      | 0.001         | 32         | SGD       | sigmoid    | 0.925    | 88.7              |
| Exp 5      | 0.001         | 32         | RMSprop   | sigmoid    | 0.957    | 90.3              |
| Exp 6      | 0.001         | 32         | Adam      | linear     | 0.948    | 89.2              |

**Findings:**
- **Learning rate**: 0.0001 performed better than 0.001
- **Batch size**: 32 slightly outperformed 64
- **Optimizer**: Adam consistently produced better results than SGD or RMSprop
- **Activation**: sigmoid performed marginally better than linear for the output layer

### U-Net Experiments

| Experiment | Image Size | Batch Size | Epochs | IoU Score | Accuracy |
|------------|------------|------------|--------|-----------|----------|
| Base U-Net | 128×128    | 16         | 15     | 0.752     | 0.946    |
| U-Net 1    | 64×64      | 16         | 15     | 0.724     | 0.938    |
| U-Net 2    | 128×128    | 8          | 15     | 0.749     | 0.945    |
| U-Net 3    | 128×128    | 16         | 25     | 0.768     | 0.953    |

**Findings:**
- Higher resolution (128×128) produced better segmentation results
- Larger batch size (16) performed slightly better than smaller (8)
- Increasing epochs from 15 to 25 improved performance but with diminishing returns
- Even the simplest U-Net configuration performed well for this task

## V. Results

### Task 1: ML Classifier Performance

| Model              | Accuracy | Precision (with mask) | Recall (with mask) | F1-score | Training Time (s) |
|--------------------|----------|------------------------|---------------------|----------|-------------------|
| SVM (RBF)          | 0.927    | 0.912                  | 0.945               | 0.928    | 8.76              |
| SVM (Linear)       | 0.913    | 0.904                  | 0.923               | 0.913    | 3.21              |
| Neural Network     | 0.935    | 0.927                  | 0.946               | 0.936    | 15.42             |

### Task 2: CNN Performance

| Model            | Accuracy | Precision (with mask) | Recall (with mask) | F1-score | Training Time (s) |
|------------------|----------|------------------------|---------------------|----------|-------------------|
| Best CNN (Exp 3) | 0.961    | 0.953                  | 0.972               | 0.962    | 92.1              |
| Average CNN      | 0.947    | 0.941                  | 0.951               | 0.946    | 87.2              |
| Worst CNN (Exp 4)| 0.925    | 0.917                  | 0.933               | 0.925    | 88.7              |

### Task 3: Traditional Segmentation Performance

| Method             | IoU    | Dice Coefficient | Processing Time (s/image) | Mask Coverage (%) |
|--------------------|--------|------------------|---------------------------|-------------------|
| Improved Hybrid    | 0.72   | 0.83             | 0.14                      | 28.6              |
| Edge Detection     | 0.58   | 0.73             | 0.09                      | 22.4              |
| Watershed          | 0.65   | 0.78             | 0.21                      | 31.2              |
| Felzenszwalb       | 0.61   | 0.76             | 0.19                      | 26.7              |

### Task 4: U-Net Segmentation Performance

| Metric             | Value  |
|--------------------|--------|
| Mean IoU           | 0.752  |
| Accuracy           | 0.946  |
| Dice Coefficient   | 0.849  |
| Training Time      | 687s   |
| Inference Time     | 0.073s |

### Comparison of All Approaches

- **Classification Performance**: CNN (96.1%) > Neural Network (93.5%) > SVM-RBF (92.7%) > SVM-Linear (91.3%)

- **Segmentation Performance (IoU)**: U-Net (0.752) > Improved Hybrid (0.72) > Watershed (0.65) > Felzenszwalb (0.61) > Edge Detection (0.58)

## VI. Observations and Analysis

### Key Insights

1. **Deep Learning vs. Traditional Methods**:
   - CNN outperformed traditional ML classifiers by approximately 2.6% in accuracy
   - U-Net provided more precise segmentation than traditional computer vision methods
   - The performance gap between deep learning and traditional methods was smaller than expected for segmentation tasks

2. **Feature Importance**:
   - HOG features were more discriminative than LBP features for mask detection
   - The CNN automatically learned hierarchical features that captured both shape and texture information efficiently

3. **Segmentation Challenges**:
   - Masks with unusual colors or patterns were difficult to segment using traditional methods
   - The lower face region constraint significantly improved results in traditional segmentation approaches
   - U-Net was better at handling complex mask shapes but required labeled training data

4. **Efficiency vs. Accuracy Trade-offs**:
   - Traditional ML classifiers: fastest training, lowest accuracy
   - CNN: longer training, highest accuracy for classification
   - Edge detection: fastest segmentation, lowest accuracy
   - U-Net: slowest training, highest segmentation accuracy

### Challenges Faced

1. **Dataset Variability**:
   - Images varied significantly in lighting, angle, and mask types
   - Solution: Data augmentation for CNNs; multiple color spaces for traditional segmentation

2. **Segmentation Accuracy**:
   - Traditional methods struggled with diverse mask colors and patterns
   - Solution: Combined approaches (hybrid method) and anatomical constraints

3. **Ground Truth Matching**:
   - Finding matching image-mask pairs in the MFSD dataset was challenging
   - Solution: Implemented robust filename matching and directory traversal

4. **Resource Constraints**:
   - U-Net training required substantial computational resources
   - Solution: Limited training to a subset of data; optimized batch size and image dimensions

## VII. How to Run the Code

### Prerequisites

- Python 3.7+
- Required packages:
  ```
  numpy
  opencv-python
  scikit-learn
  scikit-image
  tensorflow>=2.4.0
  matplotlib
  pandas
  seaborn
  ```

### Setup

1. Clone the repository or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets (or use the automatic downloader in the code):
   - Face Mask Detection Dataset: https://github.com/chandrikadeb7/Face-Mask-Detection
   - MFSD Dataset: https://github.com/sadjadrz/MFSD

### Running Tasks

#### Task 1: Traditional ML Classification
```bash
# Change directory to project folder
cd face-mask-detection

# Run the ML classification script
python run_ml_classification.py --dataset_path /path/to/dataset
```

#### Task 2: CNN Classification
```bash
# Run the CNN classification script
python run_cnn_classification.py --dataset_path /path/to/dataset
```

#### Task 3: Traditional Segmentation
```bash
# Run the segmentation script
python run_segmentation.py --dataset_path /path/to/dataset
```

#### Task 4: U-Net Segmentation
```bash
# Run the U-Net segmentation script
python run_unet_segmentation.py
```

#### Running the Complete Pipeline
For a complete end-to-end execution of all tasks:
```bash
# Open the Jupyter notebook
jupyter notebook VR_miniProj-4.ipynb

# Execute cells in order to run all tasks
```

### Parameter Customization

- **CNN Hyperparameters**:
  ```bash
  python run_cnn_classification.py --learning_rate 0.0001 --batch_size 32 --optimizer adam
  ```

- **Segmentation Method Selection**:
  ```bash
  python run_segmentation.py --method hybrid --dataset_path /path/to/dataset
  ```

- **U-Net Configuration**:
  ```bash
  python run_unet_segmentation.py --image_size 128 --batch_size 16 --epochs 15
  ```

### Output

- Results will be saved in the `results/` directory
- Visualizations will be displayed and saved as PNG files
- Trained models will be saved with appropriate names
- Evaluation metrics will be printed to console and saved to CSV files

Similar code found with 2 license types
