import os
import numpy as np
import cv2
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import Conv2DTranspose, concatenate


def load_mfsd_dataset():
    """
    Load the Masked Face Segmentation Dataset from the provided zip file
    """
    # Check if we're in Google Colab
    try:
        from google.colab import drive
        in_colab = True
    except ImportError:
        in_colab = False

    # Define paths
    extract_path = "mfsd_dataset"

    # If in Colab, check Google Drive paths
    if in_colab:
        # Try different possible paths for the ZIP file
        possible_paths = [
            '/content/drive/MyDrive/datasets_ml/MSFD.zip',
            '/content/drive/MyDrive/MSFD.zip',
            '/content/MSFD.zip'
        ]

        zip_path = None
        for path in possible_paths:
            if os.path.exists(path):
                zip_path = path
                print(f"Found MSFD dataset at {zip_path}")
                break

        if zip_path is None:
            print("MSFD zip file not found in Google Drive.")
            return None

        # Mount Google Drive if needed
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')

        # Extract the dataset if not already extracted
        if not os.path.exists(extract_path) or not os.listdir(extract_path):
            if not extract_from_zip(zip_path, extract_path):
                return None
    else:
        # For local execution, assume the dataset is in the current directory
        zip_path = "MSFD.zip"
        if os.path.exists(zip_path):
            if not os.path.exists(extract_path) or not os.listdir(extract_path):
                if not extract_from_zip(zip_path, extract_path):
                    return None
        else:
            print(f"MSFD zip file not found at {zip_path}")
            return None

    # Now look for the specific directories after extraction
    image_dirs = []
    segmentation_dirs = []

    # Check all subdirectories
    for root, dirs, files in os.walk(extract_path):
        if "face_crop" in root:
            if "segmentation" not in root:  # Avoid counting segmentation directories twice
                image_dirs.append(root)
        if "face_crop_segmentation" in root:
            segmentation_dirs.append(root)

    # Count images and masks
    image_count = 0
    mask_count = 0

    for img_dir in image_dirs:
        for file in os.listdir(img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1

    for seg_dir in segmentation_dirs:
        for file in os.listdir(seg_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                mask_count += 1

    print(f"Found {image_count} images and {mask_count} mask files in {extract_path}")

    # Create and return dataset structure
    dataset_info = {
        "base_path": extract_path,
        "image_dirs": image_dirs,
        "segmentation_dirs": segmentation_dirs,
        "image_count": image_count,
        "mask_count": mask_count
    }

    if image_count > 0 and mask_count > 0:
        return dataset_info
    else:
        print("No valid images or masks found in the dataset")
        return None

def find_image_mask_pairs(dataset_info):
    """
    Find matching image-mask pairs in the MFSD dataset
    """
    image_dirs = dataset_info["image_dirs"]
    segmentation_dirs = dataset_info["segmentation_dirs"]

    # Collect all image paths
    image_paths = []
    for img_dir in image_dirs:
        for file in os.listdir(img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(img_dir, file))

    # Collect all mask paths
    mask_paths = []
    for seg_dir in segmentation_dirs:
        for file in os.listdir(seg_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                mask_paths.append(os.path.join(seg_dir, file))

    print(f"Found {len(image_paths)} image files and {len(mask_paths)} mask files")

    # Match image-mask pairs by filename
    matched_pairs = []

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]  # Get filename without extension

        # Look for matching mask
        for mask_path in mask_paths:
            mask_filename = os.path.basename(mask_path)
            mask_name = os.path.splitext(mask_filename)[0]

            # Check if the mask filename exactly matches the image filename
            if mask_name == img_name:
                matched_pairs.append((img_path, mask_path))
                break

    print(f"Found {len(matched_pairs)} matched image-mask pairs")

    if len(matched_pairs) > 0:
        # Show a sample pair
        img_sample, mask_sample = matched_pairs[0]
        print(f"Sample pair: {os.path.basename(img_sample)} - {os.path.basename(mask_sample)}")

    return matched_pairs

def train_unet(dataset_info, image_size=(128, 128), epochs=15):
    """
    Train U-Net for mask segmentation using the MSFD dataset structure
    """
    print("\nPart D: Mask Segmentation Using U-Net")

    # Check if dataset info is valid
    if dataset_info is None:
        print("MFSD dataset could not be loaded properly. Skipping U-Net segmentation.")
        return None

    try:
        # Get directories
        image_dirs = dataset_info["image_dirs"]
        segmentation_dirs = dataset_info["segmentation_dirs"]

        if not image_dirs or not segmentation_dirs:
            print("No valid image or segmentation directories found. Skipping U-Net segmentation.")
            return None

        print(f"Found {len(image_dirs)} image directories and {len(segmentation_dirs)} segmentation directories")

        # Find matching image-mask pairs
        matched_pairs = []

        # For each folder pair, match images with their masks
        for img_dir in image_dirs:
            # Find the corresponding segmentation directory
            folder_num = os.path.basename(os.path.dirname(img_dir))
            seg_dir_candidate = None

            for seg_dir in segmentation_dirs:
                if folder_num in seg_dir:
                    seg_dir_candidate = seg_dir
                    break

            if seg_dir_candidate:
                # Get all image files in this directory
                img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

                # For each image, find matching mask
                for img_file in img_files:
                    img_path = os.path.join(img_dir, img_file)
                    img_name = os.path.splitext(img_file)[0]

                    # Look for exact filename match in segmentation directory
                    for mask_file in os.listdir(seg_dir_candidate):
                        if mask_file.startswith(img_name) and mask_file.endswith(('.jpg', '.jpeg', '.png')):
                            mask_path = os.path.join(seg_dir_candidate, mask_file)
                            matched_pairs.append((img_path, mask_path))
                            break

        print(f"Found {len(matched_pairs)} matched image-mask pairs")

        if len(matched_pairs) == 0:
            print("No valid image-mask pairs found. Skipping U-Net training.")
            return None

        # Process a subset of the data to keep training time reasonable
        max_samples = min(500, len(matched_pairs))
        selected_pairs = matched_pairs[:max_samples]

        # Load and preprocess data
        X = []  # Images
        y = []  # Masks

        print(f"Processing {len(selected_pairs)} image-mask pairs...")
        for i, (img_path, mask_path) in enumerate(selected_pairs):
            if i % 50 == 0:
                print(f"  {i}/{len(selected_pairs)} pairs processed...")

            # Read and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, image_size)
            img = img / 255.0  # Normalize

            # Read and preprocess mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = cv2.resize(mask, image_size)
            mask = (mask > 127).astype(np.float32)  # Binarize
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension

            X.append(img)
            y.append(mask)

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Final dataset: {len(X)} valid image-mask pairs")

        if len(X) < 10:  # Not enough data
            print("Too few valid image-mask pairs for training. Skipping U-Net training.")
            return None

        # Rest of the function remains the same as before
        # (U-Net model creation, training, evaluation and visualization)

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Build U-Net model
        inputs = tf.keras.layers.Input((image_size[0], image_size[1], 3))

        # Encoder
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

        # Output
        outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=16,
            epochs=epochs,
            verbose=1
        )

        # Evaluate model performance
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        # Visualize results
        plt.figure(figsize=(15, 10))
        for i in range(min(5, len(X_test))):
            # Original image
            plt.subplot(5, 3, i*3+1)
            plt.imshow(X_test[i])
            plt.title(f"Image {i+1}")
            plt.axis('off')

            # Ground truth mask
            plt.subplot(5, 3, i*3+2)
            plt.imshow(y_test[i].squeeze(), cmap='gray')
            plt.title(f"True Mask {i+1}")
            plt.axis('off')

            # Predicted mask
            pred = model.predict(np.expand_dims(X_test[i], axis=0)).squeeze()
            plt.subplot(5, 3, i*3+3)
            plt.imshow(pred > 0.5, cmap='gray')
            plt.title(f"Predicted Mask {i+1}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('unet_predictions.png')

        # Calculate IoU for comparison with traditional methods
        def calculate_iou(y_true, y_pred):
            y_true = y_true > 0.5
            y_pred = y_pred > 0.5
            intersection = np.logical_and(y_true, y_pred)
            union = np.logical_or(y_true, y_pred)
            iou_score = np.sum(intersection) / (np.sum(union) + 1e-7)
            return iou_score

        # Evaluate IoU on test set
        test_ious = []
        for i in range(len(X_test)):
            pred = model.predict(np.expand_dims(X_test[i], axis=0)).squeeze()
            iou = calculate_iou(y_test[i].squeeze(), pred)
            test_ious.append(iou)

        mean_iou = np.mean(test_ious)
        print(f"Mean IoU: {mean_iou:.4f}")

        return model

    except Exception as e:
        import traceback
        print(f"Error in U-Net training: {e}")
        traceback.print_exc()
        return None

def download_and_extract_dataset(url, extract_path):
    """
    Download dataset from GitHub repository and extract it
    """
    # Create directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # If downloading from GitHub repo, modify URL to raw content
    if 'github.com' in url and '/tree/master/' in url:
        # Convert URL to the archive download link
        repo_url = url.split('/tree/master/')[0]
        branch = 'master'
        if '/tree/' in url:
            branch = url.split('/tree/')[1].split('/')[0]

        download_url = f"{repo_url}/archive/{branch}.zip"
    else:
        download_url = url

    print(f"Downloading dataset from {download_url}...")

    try:
        # Download the dataset
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Extract the ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"Dataset extracted to {extract_path}")

        # Return the path to extracted contents
        return extract_path
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        return None

def load_mask_detection_dataset():
    """
    Load the Face Mask Detection dataset from GitHub
    """
    dataset_url = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/master.zip"
    extract_path = "mask_detection_dataset"

    if not os.path.exists(extract_path):
        download_and_extract_dataset(dataset_url, extract_path)

    # Dataset is inside the extracted folder
    dataset_path = os.path.join(extract_path, "Face-Mask-Detection-master", "dataset")

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found. Searching for the correct path...")

        # Search for dataset folder recursively
        for root, dirs, files in os.walk(extract_path):
            if "with_mask" in dirs and "without_mask" in dirs:
                dataset_path = root
                print(f"Found dataset at {dataset_path}")
                break

    return dataset_path

def main():
    
    print("\n--- Part D: Mask Segmentation Using U-Net ---")
    # Load MFSD dataset
    print("\nLoading MFSD dataset...")
    mfsd_dataset_info = load_mfsd_dataset()

    if mfsd_dataset_info:
        unet_model = train_unet(mfsd_dataset_info)
    else:
        print("MFSD dataset could not be loaded properly. Skipping U-Net segmentation.")

if __name__ == "__main__":
    main()