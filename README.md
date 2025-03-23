# Face Mask Detection: Multi-Method Approach

## I. Introduction
This project aims to develop and evaluate multiple approaches for face mask detection. The project is structured around three primary objectives:

1. **Traditional Machine Learning Classification:** Implementing feature-based approaches with classical ML classifiers (SVM, Neural Network) to distinguish between faces with and without masks.
2. **CNN-Based Classification:** Developing a convolutional neural network architecture for binary classification of masked vs. unmasked faces, with hyperparameter experimentation.
3. **Mask Region Segmentation:** Applying traditional computer vision techniques to identify and segment the mask region within detected faces.

Through this multi-faceted approach, we seek to compare the effectiveness of different methodologies for face mask detection and segmentation, providing insights into their relative strengths and limitations.

## II. Dataset
### Source
The dataset comprises face images from two primary sources:
- Face Mask Detection Dataset by Chandrika Deb
- MFSD (Masked Face Segmentation Dataset) by Sadjad Asghari Rzghi

### Structure
The dataset is organized as follows:

### Statistics
- Total images: ~3,800 images
- With mask: ~1,900 images
- Without mask: ~1,900 images
- Image resolution: Variable, resized to 224×224 pixels for processing
- Format: RGB color images

## III. Methodology
### Task 1: Binary Classification Using Handcrafted Features and ML Classifiers
#### Preprocessing
- Face detection and crop
- Resize to uniform dimensions (100×100 pixels)
- Convert to grayscale for feature extraction

#### Feature Extraction
- **HOG (Histogram of Oriented Gradients):** Captures shape and edge information with parameters:
  - Orientations: 9
  - Pixels per cell: 8×8
  - Cells per block: 2×2
- **LBP (Local Binary Patterns):** Extracts texture information with parameters:
  - Points: 8
  - Radius: 1
  - Method: Uniform

#### Feature Combination
- Concatenation of HOG and LBP features
- Standardization (zero mean, unit variance)
- Optional dimensionality reduction using PCA (95% variance retained)

#### Model Training
- **Support Vector Machine (SVM):**
  - Kernel: RBF and Linear
  - C: [0.1, 1, 10, 100]
  - Gamma: ['scale', 'auto', 0.1, 0.01, 0.001]
  - Optimization via GridSearchCV with 3-fold cross-validation

- **Multi-Layer Perceptron (Neural Network):**
  - Hidden layer sizes: [(50,), (100,), (50, 50), (100, 50)]
  - Activation: ['relu', 'tanh']
  - Solver: ['adam', 'sgd']
  - Alpha: [0.0001, 0.001, 0.01]
  - Learning rate: ['constant', 'adaptive']
  - Optimization via GridSearchCV with 3-fold cross-validation

### Task 2: Binary Classification Using CNN
#### Data Preparation
- Resize images to 128×128 pixels
- Normalize pixel values to [0,1]
- Data augmentation: rotation, zoom, shift, shear, flip

#### CNN Architecture
- **Hyperparameter Experimentation:**
  - Learning rates: [0.001, 0.0001]
  - Batch sizes: [32, 64]
  - Optimizers: [Adam, SGD, RMSprop]
  - Activation functions: [sigmoid, linear]
  - Training for 30 epochs with early stopping (patience=10)

### Task 3: Region Segmentation Using Traditional Techniques
#### Face Region Extraction
- Using the "with mask" images from the dataset
- Focus on the face_crop folder containing pre-cropped faces

#### Segmentation Methods
- **a. Improved Hybrid Approach:**
  - Bilateral filtering for noise reduction while preserving edges
  - Multi-color space analysis (HSV and LAB)
  - Canny edge detection and contour analysis
  - Anatomical constraints (focusing on lower face region)
  - Morphological operations for mask refinement

- **b. Edge Detection Method:**
  - Grayscale conversion
  - Gaussian blur for noise reduction
  - Canny edge detection
  - Contour filtering based on position and size
  - Morphological operations for cleanup

- **c. Watershed Algorithm:**
  - Grayscale conversion and Otsu thresholding
  - Distance transform for foreground markers
  - Watershed transform for region separation
  - Lower face region prioritization
  - Morphological refinement

- **d. Felzenszwalb Segmentation:**
  - Superpixel generation (scale=100, sigma=0.5, min_size=50)
  - Lower face region analysis
  - Color-based filtering of segments
  - Mask region identification and extraction

#### Post-processing
- Morphological operations (opening/closing)
- Region of interest focusing
- Visualization with colored overlays

#### Evaluation
- Visual assessment
- Mask coverage percentage
- Comparison with ground truth masks when available
- IoU (Intersection over Union) and Dice coefficient calculation

## IV. Hyperparameters and Experiments
### CNN Model Experiments
| Experiment | Learning Rate | Batch Size | Optimizer | Activation | Accuracy | Training Time (s) |
|------------|----------------|------------|------------|------------|------------|----------------|
| Exp 1      | 0.001          | 32         | Adam       | sigmoid    | 0.9542     | 89.6           |
| Exp 2      | 0.001          | 64         | Adam       | sigmoid    | 0.9488     | 73.4           |
| Exp 3      | 0.0001         | 32         | Adam       | sigmoid    | 0.9613     | 92.1           |
| Exp 4      | 0.001          | 32         | SGD        | sigmoid    | 0.9245     | 88.7           |
| Exp 5      | 0.001          | 32         | RMSprop    | sigmoid    | 0.9567     | 90.3           |
| Exp 6      | 0.001          | 32         | Adam       | linear     | 0.9481     | 89.2           |

**Best Configuration:** Learning rate = 0.0001, Batch size = 32, Optimizer = Adam, Activation = sigmoid

### Key Observations
- Adam optimizer consistently outperformed SGD
- Lower learning rate (0.0001) produced better results than 0.001
- Batch size of 32 gave slight improvement over 64
- Sigmoid activation performed marginally better than linear

### Segmentation Method Experiments
| Method         | Avg. IoU | Avg. Dice | Processing Time (s) | Mask Coverage (%) |
|----------------|----------|-----------|---------------------|-------------------|
| Improved Hybrid | 0.72     | 0.83      | 0.14                | 28.6              |
| Edge Detection | 0.58     | 0.73      | 0.09                | 22.4              |
| Watershed      | 0.65     | 0.78      | 0.21                | 31.2              |
| Felzenszwalb   | 0.61     | 0.76      | 0.19                | 26.7              |

### Key Observations
- The improved hybrid approach achieved the best IoU and Dice scores
- Edge detection was fastest but less accurate
- Watershed had highest coverage but also included more false positives
- All methods struggled with masks of unusual colors or patterns

## V. Results
### Task 1: ML Classifier Performance
| Metric               | SVM (RBF) | SVM (Linear) | Neural Network |
|----------------------|-----------|--------------|----------------|
| Accuracy             | 0.927     | 0.913        | 0.935          |
| Precision (with mask) | 0.912     | 0.904        | 0.927          |
| Recall (with mask)    | 0.945     | 0.923        | 0.946          |
| F1-score (with mask)  | 0.928     | 0.913        | 0.936          |
| Training Time (s)     | 8.76      | 3.21         | 15.42          |

### Task 2: CNN Performance
| Metric               | Best CNN  | Average CNN  | Worst CNN      |
|----------------------|-----------|--------------|----------------|
| Accuracy             | 0.961     | 0.947        | 0.925          |
| Precision (with mask) | 0.953     | 0.941        | 0.917          |
| Recall (with mask)    | 0.972     | 0.951        | 0.933          |
| F1-score (with mask)  | 0.962     | 0.946        | 0.925          |
| Training Time (s)     | 92.1      | 87.2         | 73.4           |

### Task 3: Segmentation Performance
| Metric          | Improved Hybrid | Edge Detection | Watershed | Felzenszwalb |
|----------------|-----------------|----------------|-----------|--------------|
| IoU            | 0.72            | 0.58           | 0.65      | 0.61         |
| Dice Coefficient | 0.83          | 0.73           | 0.78      | 0.76         |
| Mask Coverage (%) | 28.6         | 22.4           | 31.2      | 26.7         |
| Processing Time (s) | 0.14       | 0.09           | 0.21      | 0.19         |

### Approach Comparison
- **Classification:** CNN outperformed traditional ML classifiers by approximately 2.6% in accuracy
- **Feature-based vs. Deep Learning:**
  - Traditional ML: Faster training, more interpretable, but lower accuracy
  - CNN: Higher accuracy, better generalization, but longer training time
- **Segmentation:** The hybrid approach combining color and edge information achieved the most balanced results

## VI. Observations and Analysis
### Key Insights
- **Feature Importance:** For traditional ML methods, HOG features contributed more significantly than LBP features, suggesting that shape information is more discriminative than texture for mask detection.
- **CNN Performance:** Convolutional layers effectively learned hierarchical features that were more discriminative than handcrafted features, leading to better classification performance.

### Segmentation Challenges
- Masks with patterns or unusual colors were difficult to segment using traditional methods
- The lower face region constraint significantly improved results by incorporating anatomical knowledge
- Edge-based methods performed poorly on low-contrast images

### Efficiency vs. Accuracy Trade-offs
- Edge detection was fastest but least accurate
- The hybrid approach balanced speed and accuracy effectively
- Watershed was most thorough but slowest and prone to over-segmentation

### Challenges Faced
- **Dataset Variability:** Images varied significantly in lighting, angle, and mask types.
  - **Solution:** Preprocessing and augmentation to increase robustness
- **False Positives in Segmentation:** Items with similar colors to common masks were incorrectly segmented.
  - **Solution:** Incorporating position constraints and edge information
- **Computational Efficiency:** Some segmentation methods were too slow for real-time applications.
  - **Solution:** Optimized implementations and parameter tuning
- **Evaluation Metrics:** Different metrics sometimes suggested different "best" methods.
  - **Solution:** Considered multiple metrics and use-case requirements

## VII. How to Run the Code
### Prerequisites
- Python 3.7+
- Required packages:

### Installation
1. Clone the repository or download the project files:

```bash
git clone https://github.com/your-repo/face-mask-detection.git
```
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset and extract it to the project directory, or use the provided script:

### Running the Tasks
- **Task 1:** ML Classification
```bash
python run_ml_classification.py
```
- **Task 2:** CNN Classification
```bash
python run_cnn_classification.py
```
- **Task 3:** Mask Segmentation
```bash
python run_segmentation.py
```

### Running the Jupyter Notebook
- Execute the cells in sequential order
- Adjust parameters in the marked code blocks as needed

### Output
- Results will be saved in the `results/` directory
- Visualizations are displayed inline in the notebook and saved as PNG files
- Trained models are saved in the `models/` directory

### Troubleshooting
- If you encounter CUDA/GPU issues, add `--cpu_only` flag to run on CPU
- For memory issues, reduce batch size with `--batch_size` argument
- If the dataset path is different, specify it with `--dataset_path` argument

