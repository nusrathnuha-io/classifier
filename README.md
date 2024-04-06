# ANN Species Classifier

This repository contains code for building and evaluating an Artificial Neural Network (ANN) classifier for species classification based on provided features. 
The classifier is built using TensorFlow and Keras libraries in Python.

## Dataset

The dataset used for training and evaluation is stored in a CSV file named `sample.csv`. 
It consists of features such as length-width ratio, stem height, number of leaves, and angle of leaf, along with the corresponding species label.

## Preprocessing

- The dataset is loaded into a Pandas DataFrame.
- Categorical columns are encoded using LabelEncoder.
- Features are normalized using StandardScaler.
- The dataset is split into training and testing sets using train_test_split.

## Model Architecture

The neural network model consists of multiple Dense layers with ReLU activation functions.
Dropout layers are incorporated for regularization to prevent overfitting. The output layer uses the softmax activation function for multiclass classification.

## Training

The model is trained using the Adam optimizer with sparse categorical cross-entropy loss. 
Early stopping callback is implemented to prevent overfitting. Training progress is monitored using validation split.

## Evaluation

The trained model is evaluated on both training and testing sets.
Metrics such as loss and accuracy are computed and printed.
Classification report and confusion matrix are generated to assess model performance.


## Usage
1. Load the dataset: Ensure that the dataset (`sample.csv`) is located in the same directory as the code.
2. Preprocess the data: Normalize the input features and encode the categorical target variable.
3. Split the data: Divide the dataset into training and testing sets.
4. Define the model architecture: Specify the number of features, classes, and hidden layers for the ANN model.
5. Compile the model: Choose an optimizer, loss function, and evaluation metrics for training the model.
6. Train the model: Fit the model to the training data and monitor its performance using validation data.
7. Evaluate the model: Assess the model's performance on both training and testing sets using accuracy, loss, classification report, and confusion matrix.
8. Analyze results: Interpret the model's performance metrics and identify areas for improvement.


## Results
- The model achieved an accuracy of approximately 95.45% on the training set and 90.91% on the test set.
- The classification report and confusion matrix provide insights into the model's performance across different classes.
- Further analysis and optimization can be performed to enhance the model's accuracy and generalization capabilities.


