Project Overview
1. Data Cleaning
The first step of the project is to ensure that only valid image files are included in the dataset. The script scans through the dataset directory, checks the image extensions, and removes any files that are not valid images (e.g., not of the types 'jpg', 'jpeg', 'png', or 'bmp').

2. Data Loading and Preprocessing
Images are loaded from the specified directory using TensorFlow's dataset utility. The images are resized to 256x256 pixels, and their pixel values are normalized (scaled to values between 0 and 1). This ensures that the model can learn effectively from the data.

3. Data Splitting
The dataset is split into three subsets: training, validation, and testing. The training set contains 60% of the data, the validation set contains 30%, and the test set contains 10%. This ensures that the model is evaluated on data it hasn't seen during training.

4. Building the Neural Network
The model is built using a Convolutional Neural Network (CNN). It consists of several convolutional layers with ReLU activation functions, followed by max-pooling and dropout layers to reduce overfitting. The model ends with a fully connected layer and a softmax activation function to output class probabilities for 8 different categories of micro-organisms.

5. Model Compilation
The model is compiled using the Adam optimizer and sparse categorical cross-entropy as the loss function. Accuracy is used as the metric to track the model's performance during training.

6. Model Training
The model is trained for 35 epochs on the training data, with the validation data used to evaluate its performance during training. This allows the model to improve its performance while avoiding overfitting.

7. Saving the Model
After training, the model is saved to a .h5 file. This allows you to load the trained model later and use it for making predictions on new images.

Usage
Run the script to train the model on the dataset.
Make predictions using the saved model by loading it and applying it to new images. This can be done using TensorFlow's model loading and prediction functions.

Conclusion
This project demonstrates a basic approach to building a CNN for classifying micro-organism images. It includes steps for cleaning the dataset, loading and preprocessing the data, building and training the model, and saving the trained model for future use.
