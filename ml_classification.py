import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern, hog
import time
import seaborn as sns

# Function to load and preprocess the Face Mask Detection dataset
def load_dataset(dataset_path):
    print("[INFO] Loading dataset...")
    data = []
    labels = []
    CATEGORIES = ["with_mask", "without_mask"]

    # Loop through each category folder
    for category_idx, category in enumerate(CATEGORIES):
        path = os.path.join(dataset_path, category)
        if not os.path.exists(path):
            print(f"[WARNING] Path {path} does not exist. Please check your dataset directory.")
            continue

        print(f"[INFO] Processing {category} images...")
        for img_file in os.listdir(path):
            if img_file.startswith('.'):  # Skip hidden files
                continue

            img_path = os.path.join(path, img_file)
            try:
                # Read image
                image = cv2.imread(img_path)

                if image is None:
                    print(f"[WARNING] Could not read image: {img_path}")
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Resize to consistent dimensions
                resized = cv2.resize(gray, (100, 100))

                # Store the image and its label
                data.append(resized)
                labels.append(category_idx)  # 0 for with_mask, 1 for without_mask

            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")

    # Convert lists to arrays
    data = np.array(data)
    labels = np.array(labels)

    print(f"[INFO] Loaded {len(data)} images")
    return data, labels

# Function to extract HOG features
def extract_hog_features(images):
    print("[INFO] Extracting HOG features...")
    features = []

    for image in images:
        hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)

    return np.array(features)

# Function to extract LBP features
def extract_lbp_features(images):
    print("[INFO] Extracting LBP features...")
    features = []

    for image in images:
        # Parameters: P=8 (number of points), R=1 (radius)
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
        # Normalize histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features.append(hist)

    return np.array(features)

# Function to combine features
def combine_features(hog_features, lbp_features):
    return np.hstack((hog_features, lbp_features))

# Function to train SVM classifier
def train_svm(X_train, y_train):
    print("[INFO] Training SVM classifier...")
    start_time = time.time()

    # Set up parameter grid for grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    # Initialize Grid Search
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    training_time = time.time() - start_time
    print(f"[INFO] SVM trained in {training_time:.2f} seconds")
    print(f"[INFO] Best parameters: {grid_search.best_params_}")

    return best_model, training_time

# Function to train Neural Network classifier
def train_neural_network(X_train, y_train):
    print("[INFO] Training Neural Network classifier...")
    start_time = time.time()

    # Set up parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    # Initialize Grid Search
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3, n_jobs=-1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    training_time = time.time() - start_time
    print(f"[INFO] Neural Network trained in {training_time:.2f} seconds")
    print(f"[INFO] Best parameters: {grid_search.best_params_}")

    return best_model, training_time

# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):
    print(f"[INFO] Evaluating {model_name}...")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"[INFO] {model_name} Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(y_test, predictions,
                                  target_names=["with_mask", "without_mask"],
                                  output_dict=True)
    print(classification_report(y_test, predictions,
                               target_names=["with_mask", "without_mask"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    return predictions, report, cm, accuracy

# Function to visualize results
def visualize_results(models_data, original_images, y_test):
    # Visualize confusion matrices
    plt.figure(figsize=(15, 6))

    for i, (model_name, data) in enumerate(models_data.items()):
        plt.subplot(1, 2, i+1)
        sns.heatmap(data["cm"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=["with_mask", "without_mask"],
                    yticklabels=["with_mask", "without_mask"])
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {data["accuracy"]:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

    # Visualize some predictions
    def visualize_predictions(models_data, images, true_labels, num_samples=5):
        indices = np.random.choice(range(len(images)), num_samples, replace=False)

        # Create a figure with rows for each model and columns for each sample
        fig, axes = plt.subplots(len(models_data), num_samples, figsize=(num_samples*3, len(models_data)*3))

        for row, (model_name, data) in enumerate(models_data.items()):
            predictions = data["predictions"]

            for col, idx in enumerate(indices):
                # Get the image and labels
                image = images[idx]
                true_label = "with_mask" if true_labels[idx] == 0 else "without_mask"
                pred_label = "with_mask" if predictions[idx] == 0 else "without_mask"

                # Set color based on prediction correctness
                color = "green" if true_label == pred_label else "red"

                # Display the image
                axes[row, col].imshow(image, cmap='gray')
                axes[row, col].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
                axes[row, col].axis("off")

                # For the first column, add model name to y-axis
                if col == 0:
                    axes[row, col].set_ylabel(model_name)

        plt.tight_layout()
        plt.savefig('sample_predictions.png')
        plt.show()

    # Visualize predictions
    visualize_predictions(models_data, original_images, y_test)

    # Plot feature importance for each model (if available)
    for model_name, data in models_data.items():
        if model_name == "SVM" and "linear" in str(data["model"].kernel):
            # For linear SVM, coefficients can be interpreted as feature importance
            feature_importance = np.abs(data["model"].coef_[0])
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importance)), feature_importance)
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Absolute Coefficient Value')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name}.png')
            plt.show()

def main():
    # Specify the dataset path
    dataset_path = "dataset"  # Change to your dataset path

    # Ensure dataset path exists or prompt for it
    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset path {dataset_path} does not exist.")
        dataset_path = input("Please enter the correct path to the dataset: ")

    # Load dataset
    images, labels = load_dataset(dataset_path)

    # Keep original images for visualization
    original_images = images.copy()

    # Extract features
    hog_features = extract_hog_features(images)
    lbp_features = extract_lbp_features(images)

    # Combine features
    combined_features = combine_features(hog_features, lbp_features)

    print(f"[INFO] Feature extraction complete. Feature vector shape: {combined_features.shape}")

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)

    # Apply PCA to reduce dimensionality (optional)
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    pca_features = pca.fit_transform(scaled_features)

    print(f"[INFO] PCA applied. Reduced feature vector shape: {pca_features.shape}")
    print(f"[INFO] Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        pca_features, labels, np.arange(len(images)),
        test_size=0.3, random_state=42, stratify=labels
    )

    # Train SVM model
    svm_model, svm_time = train_svm(X_train, y_train)

    # Train Neural Network model
    nn_model, nn_time = train_neural_network(X_train, y_train)

    # Evaluate SVM model
    svm_preds, svm_report, svm_cm, svm_accuracy = evaluate_model(svm_model, X_test, y_test, "SVM")

    # Evaluate Neural Network model
    nn_preds, nn_report, nn_cm, nn_accuracy = evaluate_model(nn_model, X_test, y_test, "Neural Network")

    # Get original test images for visualization
    test_images = original_images[test_idx]

    # Create a dictionary to store results for each model
    models_data = {
        "SVM": {
            "model": svm_model,
            "predictions": svm_preds,
            "report": svm_report,
            "cm": svm_cm,
            "accuracy": svm_accuracy,
            "training_time": svm_time
        },
        "Neural Network": {
            "model": nn_model,
            "predictions": nn_preds,
            "report": nn_report,
            "cm": nn_cm,
            "accuracy": nn_accuracy,
            "training_time": nn_time
        }
    }

    # Visualize results
    visualize_results(models_data, test_images, y_test)

    # Display summary statistics
    print("\n--- Summary Statistics ---")
    for model_name, data in models_data.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {data['accuracy']:.4f}")
        print(f"With Mask - Precision: {data['report']['with_mask']['precision']:.4f}, "
              f"Recall: {data['report']['with_mask']['recall']:.4f}, "
              f"F1: {data['report']['with_mask']['f1-score']:.4f}")
        print(f"Without Mask - Precision: {data['report']['without_mask']['precision']:.4f}, "
              f"Recall: {data['report']['without_mask']['recall']:.4f}, "
              f"F1: {data['report']['without_mask']['f1-score']:.4f}")
        print(f"Training Time: {data['training_time']:.2f} seconds")

if __name__ == "__main__":
    main()