<h1>Synthetic Image Detection</h1>

<p>This project implements a machine learning pipeline to detect synthetic (fake) images using two different models: a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM). The project is designed to classify images into two categories: <code>REAL</code> or <code>FAKE</code>.</p>

<h2>Project Structure</h2>

<p>The project includes the following components:</p>
<ul>
    <li><strong>Data Loading and Preprocessing:</strong> Load and preprocess images from the dataset directories.</li>
    <li><strong>CNN Model:</strong> A deep learning model to classify images using convolutional layers.</li>
    <li><strong>SVM Model:</strong> A machine learning model that uses a linear approach for image classification.</li>
    <li><strong>Model Evaluation:</strong> Evaluate the performance of both models using various metrics such as accuracy, precision, recall, and AUC (Area Under the Curve).</li>
    <li><strong>Visualization:</strong> Visualize the results using confusion matrices, ROC curves, and training/validation accuracy and loss curves.</li>
</ul>

<h2>Prerequisites</h2>

<p>Ensure you have the following libraries installed:</p>
<pre><code>pip install tensorflow numpy opencv-python seaborn matplotlib scikit-learn</code></pre>

<h2>Dataset</h2>

<p>The dataset should be organized into the following directory structure:</p>

<pre>
Synthetic Image Detection project/
│
├── train/
│   ├── REAL/
│   └── FAKE/
│
└── test/
    ├── REAL/
    └── FAKE/
</pre>

<ul>
    <li><strong>REAL:</strong> Contains real images.</li>
    <li><strong>FAKE:</strong> Contains synthetic images.</li>
</ul>

<h2>CNN Model</h2>

<h3>Model Architecture</h3>
<ul>
    <li><strong>Input Layer:</strong> 32x32 RGB images.</li>
    <li><strong>Convolutional Layers:</strong> 2 layers with ReLU activation, followed by max-pooling.</li>
    <li><strong>Flatten Layer:</strong> Flatten the output from the convolutional layers.</li>
    <li><strong>Dense Layers:</strong> Fully connected layers to produce the final output.</li>
    <li><strong>Output Layer:</strong> A single neuron with a sigmoid activation function for binary classification.</li>
</ul>

<h3>Training the CNN</h3>
<pre><code># Train the model
history = sequential.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
</code></pre>

<h3>Evaluating the CNN</h3>
<pre><code># Evaluate the model
results = sequential.evaluate(X_test, y_test)
</code></pre>

<h3>Visualizing CNN Performance</h3>
<ul>
    <li><strong>Confusion Matrix</strong></li>
    <li><strong>Training and Validation Accuracy</strong></li>
    <li><strong>Training and Validation Loss</strong></li>
</ul>

<h2>SVM Model</h2>

<h3>Preprocessing</h3>
<ul>
    <li><strong>Flattening:</strong> Images are flattened before being fed into the SVM.</li>
    <li><strong>Normalization:</strong> Data is normalized using <code>StandardScaler</code>.</li>
</ul>

<h3>Training the SVM</h3>
<pre><code># Train the SVM model
SVM.fit(X_train_split, y_train_split)
</code></pre>

<h3>Evaluating the SVM</h3>
<pre><code># Evaluate the model on test data
y_pred_test = SVM.predict(X_test_scaled)
</code></pre>

<h3>Visualizing SVM Performance</h3>
<ul>
    <li><strong>Confusion Matrix</strong></li>
    <li><strong>ROC Curve</strong></li>
</ul>

<h2>Usage</h2>

<ol>
    <li><strong>Run the CNN Model:</strong>
        <pre><code>python cnn_model.py</code></pre>
    </li>
    <li><strong>Run the SVM Model:</strong>
        <pre><code>python svm_model.py</code></pre>
    </li>
</ol>

<h2>Results</h2>

<p>Both models will output key performance metrics, including accuracy, precision, recall, and AUC, and will visualize the results through plots and confusion matrices.</p>
<ul>
    <li><strong>CNN MODEL</strong></li>
</ul>
<p>As we can observe in the Confusion Matrix for the CNN Model, the model presents a strong performance in classifying in a correct way both positive and negative instances. The model outputs 432 false positives indicating the model has good precision. Moreover, the high number of false negatives compared to false positives suggests that the model is precise, although it can miss some positive instances.</p>

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/566c62f4-5e11-43b2-aee8-9a2d70534033" alt="Confusion Matrix CNN" style="width: 50%; height: auto;">
</div>
<p>Throughout the training process, the accuracy of the validation set remains comparatively constant. It starts at about 82%, varies slightly, and ends at about 85%. Despite a minor increasing trend, validation accuracy does not increase as much as training accuracy, which could indicate that the model is starting to overfit the training set. We can observe it in the following figure:</p>
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/a62218b7-1f25-4871-a350-da4d1c3fafb8" alt="Accuracy and Loss Curve Graph CNN" style="width: 50%; height: auto;">
</div>
<p></p>
<ul>
    <li><strong>SVM MODEL</strong></li>
</ul>

<p>The Support Vector Machine classified 72 instances correctly as true positives and 743 as true negatives. On the other hand, it classified as false negatives 257 instances and 238 as false positives. Overall, this model shows a good performance in classifying instances. However, there is room for improvement as the false negatives and false positives are still relatively high if we look at it from the standpoint of achieving a very efficient model.</p>

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/43c55d0e-95a8-42db-b046-922861951a6c" alt="Confusion Matrix SVM" style="width: 50%; height: auto;">
</div>

<p>Additional insight of how the Support Vector Machine performed is also provided in the ROC Curve figure. The ROC Curve graph provides a visual representation of the trade-off between the True Positive Rate and the False Positive Rate for a binary classifier. With an AUC of 0.83, the ROC curve shows that it performs well. This suggests a high degree of accuracy in identifying synthetic images with a low false positive rate. The curve demonstrates how the model performs noticeably better than random guesses, which is essential for accurate synthetic identification of images.</p>

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/7c2cfeaa-6c9b-4696-bebd-4cc2c5bef03f" alt="ROC Curve" style="width: 50%; height: auto;">
</div>

<h2>Conclusion</h2>

<p>This project demonstrates the use of both deep learning (CNN) and traditional machine learning (SVM) methods for detecting synthetic images. The CNN model is more complex and typically yields better results, while the SVM model provides a simpler, linear approach to the problem.</p>



</body>
</html>
