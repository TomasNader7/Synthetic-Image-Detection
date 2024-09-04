
# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
import cv2 as cv
import random

# Seaborn and Matplotlib for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D 
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, auc, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_curve

#######################################################################################################
#                                                                                                     #
#                                SUPPORT VECTOR MACHINE MODEL (SVM)                                   #
#                                                                                                     #
#######################################################################################################

# Dataset directories
train_data_dir = r"C:\Users\tomin\Documents\Synthetic Image Detection project\train"
test_data_dir = r"C:\Users\tomin\Documents\Synthetic Image Detection project\test"

img_size = 32
categories = ['REAL', 'FAKE']
max_images_per_category = 1000  # Limit the number of images loaded from each category

def load_data(data_dir, categories, img_size, max_images):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        print(f"Loading category: {category} from path: {path}")  # Debug statement
        images_loaded = 0
        for img in os.listdir(path):
            if images_loaded >= max_images:
                break
            try:
                img_path = os.path.join(path, img)
                img_array = cv.imread(img_path)
                if img_array is not None:  # Check if image is loaded correctly
                    resized_array = cv.resize(img_array, (img_size, img_size))
                    normalized_array = resized_array / 255.0
                    data.append([normalized_array, class_num])
                    images_loaded += 1
                else:
                    print(f"Failed to load image: {img_path}")
            except Exception as e:
                print(f"Error loading image: {img_path}. Error: {e}")
    return data

print("Starting data load...")

# Load and preprocess the training and testing data
training_data = load_data(train_data_dir, categories, img_size, max_images_per_category)
print(f"Training data loaded. Length: {len(training_data)}")

testing_data = load_data(test_data_dir, categories, img_size, max_images_per_category)
print(f"Testing data loaded. Length: {len(testing_data)}")

# Shuffle the data
random.shuffle(training_data)
random.shuffle(testing_data)

# Separate features (X) and labels (y) for training set
X_train = np.array([entry[0] for entry in training_data])
y_train = np.array([entry[1] for entry in training_data])

# Separate features (X) and labels (y) for testing set
X_test = np.array([entry[0] for entry in testing_data])
y_test = np.array([entry[1] for entry in testing_data])

# Display shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Flatten the image data for SVM
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Normalize features
scaler = StandardScaler() # type: ignore
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

# Initialize Support Vector Machine with probability estimates
SVM = SVC(probability=True) # type: ignore

# Define performance metrics
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='weighted')
}

# Define the sizes for training and testing
training_sizes = [0.6, 0.7, 0.8]  # Example sizes: 60%, 70%, and 80% of the dataset for training

# Iterate over different training sizes
for size in training_sizes:
    print(f"\nTraining size: {size * 100}%")

    # Split the training data according to the specified size
    X_train_split, _, y_train_split, _ = train_test_split(X_train_scaled, y_train, train_size=size, random_state=42) # type: ignore

    # Print the sizes of the training and testing sets
    print(f"Training set size: {len(X_train_split)}")
    print(f"Testing set size: {len(X_test_scaled)}")
    
    # Train the SVM model
    SVM.fit(X_train_split, y_train_split)
    y_pred_test = SVM.predict(X_test_scaled)
    
    # Print testing performance
    print("Testing Performance:")
    print(classification_report(y_test, y_pred_test))
    
    # Perform ten-fold cross-validation
    cv_results = cross_validate(SVM, X_train_scaled, y_train, cv=10, scoring=scoring_metrics) # type: ignore
    print("Cross-validation Performance:")
    for metric, scores in cv_results.items():
        print(f"{metric}: {scores.mean()} (std: {scores.std()})")

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (SVM)')
plt.show()

# Plot ROC Curve to help visualize the trade-off between the true positive rate (sensitivity) and false positive rate (1-specificity) across different thresholds.
from sklearn.metrics import auc

# Predict probabilities on the test set
y_pred_proba = SVM.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure() 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Synthetic Image Detection')
plt.legend(loc="lower right")
plt.show()