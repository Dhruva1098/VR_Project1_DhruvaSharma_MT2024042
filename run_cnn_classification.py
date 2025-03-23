import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import cv2
import time
import seaborn as sns
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess the Face Mask Detection dataset
def load_dataset(dataset_path, target_size=(128, 128)):
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

                # Convert to RGB (from BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize to consistent dimensions
                resized = cv2.resize(image, target_size)

                # Store the image and its label
                data.append(resized)
                labels.append(category_idx)  # 0 for with_mask, 1 for without_mask

            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")

    # Convert lists to arrays
    data = np.array(data, dtype="float32") / 255.0  # Normalize to [0,1]
    labels = np.array(labels)

    print(f"[INFO] Loaded {len(data)} images")
    print(f"[INFO] Data shape: {data.shape}")
    return data, labels

# Function to build CNN model
def build_cnn_model(input_shape, activation='sigmoid', dropout_rate=0.5):
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

    return model

# Function to train and evaluate model with different hyperparameters
def experiment_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test):
    # Hyperparameter variations
    learning_rates = [0.001]
    batch_sizes = [32]
    optimizers = ['adam']
    activations = ['sigmoid']  # linear with default threshold=0.5 is equivalent to sigmoid

    # Store results
    results = []
    models = {}
    histories = {}

    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Loop through hyperparameter combinations
    experiment_count = 1
    total_experiments = len(learning_rates) * len(batch_sizes) * len(optimizers) * len(activations)

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for optimizer_name in optimizers:
                for activation in activations:
                    # Create experiment name
                    exp_name = f"lr_{lr}_batch_{batch_size}_{optimizer_name}_{activation}"
                    print(f"\n[INFO] Experiment {experiment_count}/{total_experiments}: {exp_name}")

                    # Build model
                    model = build_cnn_model(
                        input_shape=X_train.shape[1:],
                        activation=activation
                    )

                    # Set up optimizer
                    if optimizer_name == 'adam':
                        optimizer = Adam(learning_rate=lr)
                    elif optimizer_name == 'sgd':
                        optimizer = SGD(learning_rate=lr, momentum=0.9)
                    elif optimizer_name == 'rmsprop':
                        optimizer = RMSprop(learning_rate=lr)

                    # Compile model
                    model.compile(
                        loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy']
                    )

                    # Save checkpoint for this model
                    checkpoint = ModelCheckpoint(
                        f'best_model_{exp_name}.h5',
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max'
                    )

                    # Train model
                    start_time = time.time()
                    history = model.fit(
                        datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) // batch_size,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        callbacks=[early_stopping, checkpoint]
                    )
                    training_time = time.time() - start_time

                    # Evaluate on test set
                    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                    y_pred = (model.predict(X_test) > 0.5).astype("int32")

                    # Calculate metrics
                    report = classification_report(y_test, y_pred,
                                                 target_names=["with_mask", "without_mask"],
                                                 output_dict=True)

                    # Store results
                    results.append({
                        'experiment': exp_name,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'optimizer': optimizer_name,
                        'activation': activation,
                        'accuracy': test_acc,
                        'precision_with_mask': report['with_mask']['precision'],
                        'recall_with_mask': report['with_mask']['recall'],
                        'f1_with_mask': report['with_mask']['f1-score'],
                        'precision_without_mask': report['without_mask']['precision'],
                        'recall_without_mask': report['without_mask']['recall'],
                        'f1_without_mask': report['without_mask']['f1-score'],
                        'training_time': training_time
                    })

                    # Store model and history
                    models[exp_name] = model
                    histories[exp_name] = history

                    # Update experiment counter
                    experiment_count += 1

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)

    return results_df, models, histories

# Function to visualize training history
def visualize_training_history(histories, top_n=3):
    # Get top N models based on validation accuracy
    top_models = []
    for exp_name, history in histories.items():
        max_val_acc = max(history.history['val_accuracy'])
        top_models.append((exp_name, max_val_acc))

    top_models = sorted(top_models, key=lambda x: x[1], reverse=True)[:top_n]

    # Visualize training history for top models
    plt.figure(figsize=(12, 10))

    # Accuracy
    plt.subplot(2, 1, 1)
    for exp_name, _ in top_models:
        plt.plot(histories[exp_name].history['accuracy'], label=f'Train ({exp_name})')
        plt.plot(histories[exp_name].history['val_accuracy'], label=f'Val ({exp_name})')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Loss
    plt.subplot(2, 1, 2)
    for exp_name, _ in top_models:
        plt.plot(histories[exp_name].history['loss'], label=f'Train ({exp_name})')
        plt.plot(histories[exp_name].history['val_loss'], label=f'Val ({exp_name})')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.show()

    return top_models

# Function to visualize confusion matrices
def visualize_confusion_matrices(models, X_test, y_test, top_models):
    plt.figure(figsize=(15, 5 * (len(top_models) // 2 + len(top_models) % 2)))

    for i, (exp_name, _) in enumerate(top_models):
        model = models[exp_name]

        # Make predictions
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.subplot(len(top_models) // 2 + len(top_models) % 2, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["with_mask", "without_mask"],
                    yticklabels=["with_mask", "without_mask"])
        plt.title(f'Confusion Matrix - {exp_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('cnn_confusion_matrices.png')
    plt.show()

# Function to visualize sample predictions
def visualize_predictions(models, X_test, y_test, original_images, top_models, num_samples=5):
    # Select random samples
    indices = np.random.choice(range(len(X_test)), num_samples, replace=False)

    fig, axes = plt.subplots(len(top_models), num_samples, figsize=(num_samples*3, len(top_models)*3))

    for row, (exp_name, _) in enumerate(top_models):
        model = models[exp_name]

        # Make predictions for all test samples
        predictions = (model.predict(X_test) > 0.5).astype("int32")

        for col, idx in enumerate(indices):
            # Get the image and labels
            image = original_images[idx]
            true_label = "with_mask" if y_test[idx] == 0 else "without_mask"
            pred_label = "with_mask" if predictions[idx][0] == 0 else "without_mask"

            # Set color based on prediction correctness
            color = "green" if true_label == pred_label else "red"

            # Display the image
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            axes[row, col].axis("off")

            # For the first column, add model name to y-axis
            if col == 0:
                axes[row, col].set_ylabel(exp_name)

    plt.tight_layout()
    plt.savefig('cnn_sample_predictions.png')
    plt.show()

# Function to compare CNN with ML classifiers
def compare_with_ml_classifiers(cnn_results, ml_results=None):
    # Create comparison table
    plt.figure(figsize=(12, 6))

    # If ML results are provided, create comparison
    if ml_results is not None:
        # Combine results
        comparison_data = {
            'Model': [],
            'Accuracy': [],
            'Precision (with mask)': [],
            'Recall (with mask)': [],
            'F1 (with mask)': [],
            'Training Time (s)': []
        }

        # Add CNN results (top 3)
        for i in range(min(3, len(cnn_results))):
            comparison_data['Model'].append(f"CNN - {cnn_results.iloc[i]['experiment']}")
            comparison_data['Accuracy'].append(cnn_results.iloc[i]['accuracy'])
            comparison_data['Precision (with mask)'].append(cnn_results.iloc[i]['precision_with_mask'])
            comparison_data['Recall (with mask)'].append(cnn_results.iloc[i]['recall_with_mask'])
            comparison_data['F1 (with mask)'].append(cnn_results.iloc[i]['f1_with_mask'])
            comparison_data['Training Time (s)'].append(cnn_results.iloc[i]['training_time'])

        # Add ML results
        for model_name, metrics in ml_results.items():
            comparison_data['Model'].append(model_name)
            comparison_data['Accuracy'].append(metrics['accuracy'])
            comparison_data['Precision (with mask)'].append(metrics['report']['with_mask']['precision'])
            comparison_data['Recall (with mask)'].append(metrics['report']['with_mask']['recall'])
            comparison_data['F1 (with mask)'].append(metrics['report']['with_mask']['f1-score'])
            comparison_data['Training Time (s)'].append(metrics['training_time'])

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', data=comparison_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Comparison: CNN vs ML Classifiers')
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()

        # Print comparison table
        print("Model Comparison:")
        print(comparison_df.to_string(index=False))

        return comparison_df
    else:
        # Just display CNN results
        plt.figure(figsize=(10, 6))
        sns.barplot(x='experiment', y='accuracy', data=cnn_results.head(5))
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 5 CNN Models by Accuracy')
        plt.tight_layout()
        plt.savefig('top_cnn_models.png')
        plt.show()

        return cnn_results

def main():
    # Specify the dataset path
    dataset_path = "/content/datasets/face_mask_detection"  # Change to your dataset path

    # Ensure dataset path exists or prompt for it
    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset path {dataset_path} does not exist.")
        dataset_path = input("Please enter the correct path to the dataset: ")

    # Load dataset
    data, labels = load_dataset(dataset_path)

    # Keep original images for visualization
    original_images = data.copy()

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, random_state=42, stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"[INFO] Training set: {X_train.shape[0]} images")
    print(f"[INFO] Validation set: {X_val.shape[0]} images")
    print(f"[INFO] Test set: {X_test.shape[0]} images")

    # Experiment with different hyperparameters
    results, models, histories = experiment_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test)

    # Display results
    print("\n[INFO] Hyperparameter Experiment Results (sorted by accuracy):")
    print(results.to_string(index=False))

    # Save results to CSV
    results.to_csv('cnn_hyperparameter_results.csv', index=False)
    print("[INFO] Results saved to 'cnn_hyperparameter_results.csv'")

    # Visualize training history for top models
    top_models = visualize_training_history(histories)

    # Visualize confusion matrices for top models
    visualize_confusion_matrices(models, X_test, y_test, top_models)

    # Visualize predictions for top models
    visualize_predictions(models, X_test, y_test, original_images, top_models)

    # Compare with ML classifiers (placeholder)
    # In a real scenario, you would load ML classifier results here
    ml_results = None
    # Try to load ML results from Task 1 if available
    try:
        print("[INFO] Attempting to load ML classifier results from Task 1...")
        # This is a placeholder - you would need to implement proper loading
        # of your ML classifier results from Task 1
        # ml_results = load_ml_results()
    except:
        print("[INFO] No ML classifier results found. Comparing only CNN models.")

    # Compare CNN with ML classifiers
    compare_with_ml_classifiers(results, ml_results)

    # Save best model
    best_model_name = results.iloc[0]['experiment']
    best_model = models[best_model_name]
    best_model.save('best_cnn_model.h5')
    print(f"[INFO] Best model ({best_model_name}) saved to 'best_cnn_model.h5'")

    # Summary
    print("\n[INFO] CNN Binary Classification Summary:")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {results.iloc[0]['accuracy']:.4f}")
    print(f"Hyperparameters: LR={results.iloc[0]['learning_rate']}, "
          f"Batch Size={results.iloc[0]['batch_size']}, "
          f"Optimizer={results.iloc[0]['optimizer']}, "
          f"Activation={results.iloc[0]['activation']}")

if __name__ == "__main__":
    main()