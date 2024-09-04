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
from sklearn.metrics import classification_report, auc

# Dataset directories
train_data_dir = r"C:\Users\tomin\Documents\Synthetic Image Detection project\train"
test_data_dir = r"C:\Users\tomin\Documents\Synthetic Image Detection project\test"

img_size = 32
categories = ['REAL', 'FAKE']
max_images_per_category = 10000  # Limit the number of images loaded from each category

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

# Ensure the images have the correct shape for the CNN model
X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 3)
X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 3)

#######################################################################################################
#                                                                                                     #
#                              CONVOLUTIONAL NEURAL NETWORK MODEL (CNN)                               #
#                                                                                                     #
#######################################################################################################
# Initialize Sequential model
sequential = Sequential()
  
# Adding Convolutional Layer
sequential.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
               input_shape=(img_size, img_size, 3)))

# Adding pooling layer
sequential.add(MaxPooling2D(pool_size=(2, 2)))

# Adding Another Convolutional Layer and Pooling Layer
sequential.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
sequential.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten results as Model's final output will be one-dimensional array
sequential.add(Flatten())

# Adding a Dense Layer to Reduce the Number of Features
sequential.add(Dense(units=128, activation='relu'))

# Adding Another Dense Layer to Produce the Final Output
sequential.add(Dense(units=1, activation='sigmoid'))

# Print Model's Summary
print(sequential.summary())

# Compile Sequential model 
sequential.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])


# Training the Model 
history = sequential.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluating the Model
results = sequential.evaluate(X_test, y_test)

# Extracting metrics
loss = results[0]
accuracy = results[1]
precision = results[2]
recall = results[3]
auc = results[4]

# Print metrics
print(f"Sequential Model Test accuracy: {accuracy * 100:.2f}%")
print(f"Sequential Model Test precision: {precision * 100:.2f}%")
print(f"Sequential Model Test recall: {recall * 100:.2f}%")
print(f"Sequential Model Test AUC: {auc:.2f}")

# Show how well the model is able to make predictions
predictions = sequential.predict(X_test)
predicted_classes = (predictions > 0.5).astype("int32")

# Print the actual and predicted values
print("Actual labels: ", y_test)
print("Predicted labels: ", predicted_classes.flatten())

# Check the probabilities returned by predict for the first test sample
print("\nProbabilities for the first test sample:")
for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability:.10%}')

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Generate a confusion matrix
conf_matrix = tf.math.confusion_matrix(y_test, predicted_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

# Training and Validation Accuracy Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Training and Validation Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
