import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from skimage import filters, segmentation, color, morphology
import time
import zipfile
import shutil

# Set random seed for reproducibility
np.random.seed(42)

# Function to extract the dataset if it's a zip file
def extract_dataset(zip_path):
    print("[INFO] Extracting dataset...")
    dataset_dir = os.path.dirname(zip_path)
    extracted_path = os.path.join(dataset_dir, "extracted_dataset")
    
    # Create extraction directory if it doesn't exist
    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    print(f"[INFO] Dataset extracted to {extracted_path}")
    return extracted_path

# Function to load images from the dataset
def load_mask_images(dataset_path, num_samples=30):
    print("[INFO] Loading mask images...")
    images = []
    masks = []
    filenames = []
    
    # Path to directory "1" which contains mask images
    mask_dir = os.path.join(dataset_path, "1")
    
    if not os.path.exists(mask_dir):
        print(f"[WARNING] Path {mask_dir} does not exist. Please check your dataset directory.")
        return images, masks, filenames
    
    # Get face crop images
    face_crop_dir = os.path.join(mask_dir, "face_crop")
    face_segmentation_dir = os.path.join(mask_dir, "face_crop_segmentation")
    
    if not os.path.exists(face_crop_dir) or not os.path.exists(face_segmentation_dir):
        print(f"[WARNING] Face crop or segmentation directories don't exist. Checking if they're in subdirectories...")
        
        # Check subdirectories
        subdirs = [d for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]
        for subdir in subdirs:
            if os.path.exists(os.path.join(mask_dir, subdir, "face_crop")):
                face_crop_dir = os.path.join(mask_dir, subdir, "face_crop")
            if os.path.exists(os.path.join(mask_dir, subdir, "face_crop_segmentation")):
                face_segmentation_dir = os.path.join(mask_dir, subdir, "face_crop_segmentation")
    
    # Get all image files from face_crop
    all_files = []
    if os.path.exists(face_crop_dir):
        all_files = [f for f in os.listdir(face_crop_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random samples or all if fewer than requested
    sample_size = min(num_samples, len(all_files))
    if sample_size == 0:
        print("[WARNING] No face crop images found.")
        return images, masks, filenames
    
    selected_files = np.random.choice(all_files, sample_size, replace=False)
    
    for img_file in selected_files:
        face_img_path = os.path.join(face_crop_dir, img_file)
        
        # Find corresponding segmentation mask
        mask_file = img_file  # Assume same filename, might need adjustment
        mask_img_path = os.path.join(face_segmentation_dir, mask_file)
        
        try:
            # Read face image
            face_img = cv2.imread(face_img_path)
            
            if face_img is None:
                print(f"[WARNING] Could not read image: {face_img_path}")
                continue
            
            # Convert to RGB (from BGR)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Read mask if it exists
            gt_mask = None
            if os.path.exists(mask_img_path):
                gt_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                # Ensure binary mask
                if gt_mask is not None:
                    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            
            images.append(face_img)
            masks.append(gt_mask)
            filenames.append(img_file)
            
        except Exception as e:
            print(f"[ERROR] Error processing {face_img_path}: {e}")
    
    print(f"[INFO] Loaded {len(images)} mask images")
    return images, masks, filenames

def segment_mask_improved_hybrid(images):
    print("[INFO] Segmenting masks using improved hybrid method...")
    results = []
    
    for image in enumerate(images):
        if isinstance(image, tuple):  # Handle case where enumerate is used
            _, image = image
            
        # Create a copy of the original image
        original = image.copy()
        
        # 1. PREPROCESSING
        # Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. COLOR-BASED SEGMENTATION
        # Convert to HSV and LAB color spaces for better color discrimination
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
        
        # Create masks for common mask colors
        # Light blue medical masks
        mask1 = cv2.inRange(hsv, (90, 20, 150), (130, 255, 255))
        
        # Dark blue/black masks
        mask2 = cv2.inRange(hsv, (90, 30, 0), (150, 255, 100))
        
        # White/light gray masks (based on L channel in LAB)
        mask3 = cv2.inRange(lab, (200, 114, 114), (255, 140, 140))
        
        # Combine color masks
        color_mask = cv2.bitwise_or(mask1, mask2)
        color_mask = cv2.bitwise_or(color_mask, mask3)
        
        # 3. EDGE-BASED SEGMENTATION
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect them
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 4. CONTOUR ANALYSIS
        # Find contours in the dilated edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an edge-based mask by keeping only contours in lower face region
        edge_mask = np.zeros_like(gray)
        height, width = gray.shape
        
        for contour in contours:
            # Calculate contour centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Only keep contours that are likely to be mask parts (lower half of face)
                if cY > height * 0.4:  # Adjust threshold as needed
                    # Check area to filter out tiny contours
                    area = cv2.contourArea(contour)
                    if area > 50:  # Minimum area threshold
                        cv2.drawContours(edge_mask, [contour], -1, 255, -1)
        
        # 5. COMBINE MASKS
        # Logical OR between color mask and edge mask
        combined_mask = cv2.bitwise_or(color_mask, edge_mask)
        
        # 6. POST-PROCESSING
        # Apply morphological operations to refine the mask
        # Close small holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 7. REGION OF INTEREST
        # Focus on lower half of face (where mask would be)
        upper_half_cutoff = int(height * 0.35)  # Cut off upper 35%
        mask[:upper_half_cutoff, :] = 0
        
        # 8. FINALIZE RESULTS
        # Apply mask to original image
        segmented_result = np.zeros_like(image)
        segmented_result[mask > 0] = image[mask > 0]
        
        # Create a result image with overlay
        result_overlay = image.copy()
        # Add a blue overlay to show the mask region
        result_overlay[mask > 0] = (
            result_overlay[mask > 0] * 0.7 + 
            np.array([0, 0, 200], dtype=np.uint8) * 0.3
        ).astype(np.uint8)
        
        results.append({
            'original': original,
            'segmented': segmented_result,
            'mask': mask,
            'result_overlay': result_overlay,
            'method': 'improved_hybrid'
        })
    
    return results
# Function to apply color thresholding for mask segmentation
def segment_mask_color_thresholding(images):
    print("[INFO] Segmenting masks using color thresholding...")
    results = []
    
    for i, image in enumerate(images):
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Use K-means to find the dominant colors
        pixels = hsv.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Reconstruct the image with cluster colors
        segmented = centers[labels].reshape(image.shape)
        segmented = np.uint8(segmented)
        
        # Convert back to RGB
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
        
        # Create a mask based on color range for commonly used mask colors
        # This range might need adjustment based on your dataset
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Check for light blue/light green (medical masks)
        light_mask = cv2.inRange(hsv, (80, 20, 180), (120, 255, 255))
        
        # Check for dark blue/black masks
        dark_mask = cv2.inRange(hsv, (100, 30, 0), (140, 255, 80))
        
        # Check for white/light gray
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        
        # Combine masks
        mask = cv2.bitwise_or(light_mask, dark_mask)
        mask = cv2.bitwise_or(mask, white_mask)
        
        # Apply morphology operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented_result = np.zeros_like(image)
        segmented_result[mask > 0] = image[mask > 0]
        
        # Create a result image with overlay
        result_overlay = image.copy()
        result_overlay[mask > 0] = (result_overlay[mask > 0] * 0.7).astype(np.uint8)  # Darken
        result_overlay[mask > 0, 0] = 255  # Red tint
        
        results.append({
            'original': image,
            'segmented': segmented_result,
            'mask': mask,
            'result_overlay': result_overlay,
            'method': 'color_thresholding'
        })
    
    return results

# Function to apply edge detection for mask segmentation
def segment_mask_edge_detection(images):
    print("[INFO] Segmenting masks using edge detection...")
    results = []
    
    for image in images:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect them
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        # Filter contours based on area and position
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Get contour centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Check if in lower half of face (where mask would be)
                    if cY > gray.shape[0] / 2:
                        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply morphology operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented_result = np.zeros_like(image)
        for c in range(3):
            segmented_result[:, :, c] = image[:, :, c] * (mask / 255)
        
        # Create a result image with overlay
        result_overlay = image.copy()
        for c in range(3):
            channel = result_overlay[:, :, c]
            channel[mask > 0] = channel[mask > 0] * 0.7 + (c == 0) * 77  # Red tint
        
        results.append({
            'original': image,
            'segmented': segmented_result,
            'mask': mask,
            'result_overlay': result_overlay,
            'method': 'edge_detection'
        })
    
    return results

# Function to apply watershed algorithm for mask segmentation
def segment_mask_watershed(images):
    print("[INFO] Segmenting masks using watershed algorithm...")
    results = []
    
    for image in images:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is not 0, but 1
        markers = markers + 1
        
        # Mark the unknown region with 0
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create mask
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        
        # Focus on the lower half of the face (where mask would be)
        half_height = mask.shape[0] // 2
        upper_mask = mask[:half_height, :]
        lower_mask = mask[half_height:, :]
        
        # Calculate the average intensity in each half
        if np.sum(upper_mask) > 0 and np.sum(lower_mask) > 0:
            upper_avg = np.sum(upper_mask) / np.count_nonzero(upper_mask)
            lower_avg = np.sum(lower_mask) / np.count_nonzero(lower_mask)
            
            # If lower half has more segmentation, keep it
            if lower_avg >= upper_avg:
                mask[:half_height, :] = 0
        
        # Apply morphology to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented_result = np.zeros_like(image)
        for c in range(3):
            segmented_result[:, :, c] = image[:, :, c] * (mask / 255)
        
        # Create a result image with overlay
        result_overlay = image.copy()
        for c in range(3):
            channel = result_overlay[:, :, c]
            channel[mask > 0] = channel[mask > 0] * 0.7 + (c == 2) * 77  # Blue tint
        
        results.append({
            'original': image,
            'segmented': segmented_result,
            'mask': mask,
            'result_overlay': result_overlay,
            'method': 'watershed'
        })
    
    return results

# Function to apply Felzenszwalb segmentation (replacing graph-based method)
def segment_mask_felzenszwalb(images):
    print("[INFO] Segmenting masks using Felzenszwalb algorithm...")
    results = []
    
    for image in images:
        # Apply Felzenszwalb segmentation
        segments = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        
        # Create a copy of the original image for visualization
        segment_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Find number of unique segments
        unique_segments = np.unique(segments)
        
        # Identify segments likely to be masks (in lower part of the image)
        height, width = image.shape[:2]
        half_height = height // 2
        
        # For each segment, check if it's predominantly in the lower half
        for segment_id in unique_segments:
            # Create mask for this segment
            segment_region = (segments == segment_id)
            
            # Count pixels in upper and lower half
            upper_count = np.sum(segment_region[:half_height, :])
            lower_count = np.sum(segment_region[half_height:, :])
            
            total_count = upper_count + lower_count
            
            # If segment is predominantly in lower half, include it in the mask
            if total_count > 0 and lower_count / total_count > 0.6:
                # Also check color is typical for masks
                segment_colors = image[segment_region]
                avg_color = np.mean(segment_colors, axis=0)
                
                # Simple heuristic for mask colors (blue, white, light colors)
                if (avg_color[2] > 100 or  # High blue value
                    np.mean(avg_color) > 150):  # Light color overall
                    segment_mask[segment_region] = 255
        
        # Apply morphology to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(segment_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        segmented_result = np.zeros_like(image)
        for c in range(3):
            segmented_result[:, :, c] = image[:, :, c] * (mask / 255)
        
        # Create a result image with overlay
        result_overlay = image.copy()
        for c in range(3):
            channel = result_overlay[:, :, c]
            channel[mask > 0] = channel[mask > 0] * 0.7 + (c == 1) * 77  # Green tint
        
        results.append({
            'original': image,
            'segmented': segmented_result,
            'mask': mask,
            'result_overlay': result_overlay,
            'method': 'felzenszwalb'
        })
    
    return results

# Function to compare with ground truth masks (if available)
def compare_with_ground_truth(segmentation_results, ground_truth_masks):
    print("[INFO] Comparing with ground truth masks...")
    
    # Only process images where ground truth is available
    valid_results = []
    
    for i, result in enumerate(segmentation_results):
        if i < len(ground_truth_masks) and ground_truth_masks[i] is not None:
            method = result['method']
            predicted_mask = result['mask']
            gt_mask = ground_truth_masks[i]
            
            # Resize ground truth mask if necessary
            if gt_mask.shape != predicted_mask.shape:
                gt_mask = cv2.resize(gt_mask, (predicted_mask.shape[1], predicted_mask.shape[0]))
            
            # Ensure binary
            _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Calculate metrics
            intersection = np.logical_and(predicted_mask, gt_mask)
            union = np.logical_or(predicted_mask, gt_mask)
            
            # IoU (Intersection over Union)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            
            # Dice coefficient
            dice = 2 * np.sum(intersection) / (np.sum(predicted_mask) + np.sum(gt_mask)) if (np.sum(predicted_mask) + np.sum(gt_mask)) > 0 else 0
            
            # Save metrics
            result['iou'] = iou
            result['dice'] = dice
            
            # Create comparison visualization
            comparison = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
            
            # Red: False Positive (prediction positive, ground truth negative)
            comparison[(predicted_mask > 0) & (gt_mask == 0)] = [255, 0, 0]
            
            # Green: True Positive (prediction positive, ground truth positive)
            comparison[(predicted_mask > 0) & (gt_mask > 0)] = [0, 255, 0]
            
            # Blue: False Negative (prediction negative, ground truth positive)
            comparison[(predicted_mask == 0) & (gt_mask > 0)] = [0, 0, 255]
            
            result['comparison'] = comparison
            
            valid_results.append(result)
    
    return valid_results

# Function to visualize segmentation results with ground truth comparison
def visualize_segmentation_results(segmentation_results, with_ground_truth=False):
    # Group results by method
    methods = {}
    for result in segmentation_results:
        method = result['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(result)
    
    # Number of methods
    num_methods = len(methods)
    
    # Find common images across all methods
    method_keys = list(methods.keys())
    common_originals = {}
    
    # Use first method as reference
    if method_keys:
        for result in methods[method_keys[0]]:
            common_originals[id(result['original'])] = result['original']
    
    # Take a subset for visualization
    max_samples = min(5, len(common_originals))
    sample_originals = list(common_originals.values())[:max_samples]
    
    # Create side-by-side comparison
    cols = num_methods + 1
    if with_ground_truth:
        cols += 1  # Add column for ground truth comparison
    
    plt.figure(figsize=(4 * cols, 4 * max_samples))
    
    for i, original in enumerate(sample_originals):
        # Original image
        plt.subplot(max_samples, cols, i * cols + 1)
        plt.imshow(original)
        plt.title("Original")
        plt.axis('off')
        
        # Segmentation results for each method
        for j, method in enumerate(methods.keys()):
            method_results = [r for r in methods[method] if id(r['original']) == id(original)]
            
            if method_results:
                result = method_results[0]
                plt.subplot(max_samples, cols, i * cols + j + 2)
                plt.imshow(result['result_overlay'])
                plt.title(f"{method}")
                plt.axis('off')
                
                # Ground truth comparison if available
                if with_ground_truth and 'comparison' in result:
                    plt.subplot(max_samples, cols, i * cols + j + 2 + num_methods)
                    plt.imshow(result['comparison'])
                    plt.title(f"Comparison\nIoU: {result['iou']:.2f}")
                    plt.axis('off')
            else:
                plt.subplot(max_samples, cols, i * cols + j + 2)
                plt.text(0.5, 0.5, "Segmentation Failed", horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=300)
    plt.show()
    
    # Create detailed visualizations for each method
    for method, results in methods.items():
        if not results:
            continue
        
        # Take a subset for visualization
        max_detailed = min(3, len(results))
        visualize_detailed = results[:max_detailed]
        
        cols = 4
        if with_ground_truth and 'comparison' in results[0]:
            cols = 5
        
        plt.figure(figsize=(4 * cols, 4 * max_detailed))
        
        for i, result in enumerate(visualize_detailed):
            # Original image
            plt.subplot(max_detailed, cols, i * cols + 1)
            plt.imshow(result['original'])
            plt.title("Original")
            plt.axis('off')
            
            # Segmentation mask
            plt.subplot(max_detailed, cols, i * cols + 2)
            plt.imshow(result['mask'], cmap='gray')
            plt.title("Segmentation Mask")
            plt.axis('off')
            
            # Segmented result
            plt.subplot(max_detailed, cols, i * cols + 3)
            plt.imshow(result['segmented'])
            plt.title("Segmented Result")
            plt.axis('off')
            
            # Final overlay
            plt.subplot(max_detailed, cols, i * cols + 4)
            plt.imshow(result['result_overlay'])
            plt.title(f"Final Result: {method}")
            plt.axis('off')
            
            # Ground truth comparison if available
            if with_ground_truth and 'comparison' in result:
                plt.subplot(max_detailed, cols, i * cols + 5)
                plt.imshow(result['comparison'])
                plt.title(f"GT Comparison\nIoU: {result['iou']:.2f}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'detailed_{method}_segmentation.png', dpi=300)
        plt.show()

# Function to evaluate segmentation results quantitatively
def evaluate_segmentation_results(segmentation_results, has_ground_truth=False):
    print("[INFO] Evaluating segmentation results...")
    
    # Group results by method
    methods = {}
    for result in segmentation_results:
        method = result['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(result)
    
    # Calculate metrics for each method
    method_metrics = {}
    
    for method, results in methods.items():
        # Calculate mask coverage (percentage of face area covered by mask)
        mask_coverage = []
        for result in results:
            if 'mask' in result:
                mask = result['mask']
                mask_area = np.sum(mask > 0)
                face_area = mask.shape[0] * mask.shape[1]
                coverage = mask_area / face_area if face_area > 0 else 0
                mask_coverage.append(coverage)
        
        avg_coverage = np.mean(mask_coverage) if mask_coverage else 0
        
        # Ground truth metrics if available
        iou_scores = []
        dice_scores = []
        
        if has_ground_truth:
            for result in results:
                if 'iou' in result and 'dice' in result:
                    iou_scores.append(result['iou'])
                    dice_scores.append(result['dice'])
        
        method_metrics[method] = {
            'avg_mask_coverage': avg_coverage,
            'count': len(results)
        }
        
        if iou_scores:
            method_metrics[method]['avg_iou'] = np.mean(iou_scores)
            method_metrics[method]['avg_dice'] = np.mean(dice_scores)
    
    # Create bar charts for metrics
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    methods_list = list(method_metrics.keys())
    
    if has_ground_truth:
        plt.subplot(1, 2, 1)
    
    # Average mask coverage
    coverages = [method_metrics[m]['avg_mask_coverage'] * 100 for m in methods_list]
    
    bars = plt.bar(methods_list, coverages, color=['blue', 'green', 'red', 'purple'])
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Average Mask Coverage by Method')
    plt.xlabel('Method')
    plt.ylabel('Average Coverage (%)')
    plt.ylim(0, 60)  # Give room for data labels
    
    # Ground truth metrics if available
    if has_ground_truth:
        plt.subplot(1, 2, 2)
        
        iou_scores = [method_metrics[m].get('avg_iou', 0) * 100 for m in methods_list]
        dice_scores = [method_metrics[m].get('avg_dice', 0) * 100 for m in methods_list]
        
        x = np.arange(len(methods_list))
        width = 0.35
        
        plt.bar(x - width/2, iou_scores, width, label='IoU', color='blue')
        plt.bar(x + width/2, dice_scores, width, label='Dice', color='green')
        
        plt.title('Segmentation Accuracy vs Ground Truth')
        plt.xlabel('Method')
        plt.ylabel('Score (%)')
        plt.xticks(x, methods_list)
        plt.legend()
        plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('segmentation_metrics.png', dpi=300)
    plt.show()
    
    # Print metrics
    print("\n=== Segmentation Performance Metrics ===")
    for method, metrics in method_metrics.items():
        print(f"\n{method.upper()}:")
        print(f"Average Mask Coverage: {metrics['avg_mask_coverage']*100:.2f}%")
        
        if 'avg_iou' in metrics:
            print(f"Average IoU: {metrics['avg_iou']*100:.2f}%")
            print(f"Average Dice Coefficient: {metrics['avg_dice']*100:.2f}%")
    
    return method_metrics

def main():
    # Check if the dataset is a zip file
    dataset_path = "/content/drive/MyDrive/datasets_ml/extracted_dataset/MSFD"
    
    if dataset_path.endswith('.zip'):
        dataset_path = extract_dataset(dataset_path)
    
    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path {dataset_path} does not exist.")
        return
    
    # Load images
    images, ground_truth_masks, filenames = load_mask_images(dataset_path)
    
    if not images:
        print("[ERROR] No images loaded. Please check your dataset path.")
        return
    
    # Check if ground truth masks are available
    has_ground_truth = any(mask is not None for mask in ground_truth_masks)
    print(f"[INFO] Ground truth masks available: {has_ground_truth}")
    
    # Apply different segmentation techniques
    start_time = time.time()
    color_results = segment_mask_improved_hybrid(images)
    color_time = time.time() - start_time
    
    start_time = time.time()
    edge_results = segment_mask_edge_detection(images)
    edge_time = time.time() - start_time
    
    start_time = time.time()
    watershed_results = segment_mask_watershed(images)
    watershed_time = time.time() - start_time
    
    start_time = time.time()
    felzenszwalb_results = segment_mask_felzenszwalb(images)
    felzenszwalb_time = time.time() - start_time
    
    # Combine all results
    all_results = color_results + edge_results + watershed_results + felzenszwalb_results
    
    # Compare with ground truth if available
    if has_ground_truth:
        valid_results = compare_with_ground_truth(all_results, ground_truth_masks)
    else:
        valid_results = all_results
    
    # Print processing times
    print("\n=== Processing Times ===")
    print(f"Color Thresholding: {color_time:.2f} seconds")
    print(f"Edge Detection: {edge_time:.2f} seconds")
    print(f"Watershed: {watershed_time:.2f} seconds")
    print(f"Felzenszwalb: {felzenszwalb_time:.2f} seconds")
    
    # Visualize results
    visualize_segmentation_results(valid_results, with_ground_truth=has_ground_truth)
    
    # Evaluate results
    metrics = evaluate_segmentation_results(valid_results, has_ground_truth=has_ground_truth)
    
    print("\n[INFO] Segmentation analysis complete.")
    print("[INFO] Visualization images saved.")

if __name__ == "__main__":
    main()