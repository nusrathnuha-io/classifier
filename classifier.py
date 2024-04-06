import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Load the dataset
data = pd.read_csv('sample.csv')

#create a column in the data DataFrame
features = ['length_width_ratio', 'Stem_height', 'No_of_leaves', 'Angle_of_Leaf', 'Species']
data.columns = features
#inspect data
data.head()

# dataframe information
data.info()

# for each of the categorical columns, lets see the unique values
for i in data.columns:
    # print(i)
    if data[i].dtype == object:
        print(data[i].unique())

# Extract features and target variable
X = data.drop('Species', axis=1)
y = data['Species']


# Preview features
X.head()

# Normalize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix

# Define the number of features and classes based on your dataset
num_features = X_train.shape[1]  # Assuming X_train is your feature matrix
num_classes = len(np.unique(y_train))  # Assuming y_train is your label vector

# Define the neural network model
model = Sequential([
Dense(64, activation='relu', input_shape=(num_features,)),
Dropout(0.5),
Dense(64, activation='relu'),
Dropout(0.5),
Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Fit the model
history = model.fit(X_train, y_train, batch_size=256, epochs=300,
                    verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on train and test sets
train_loss, train_accuracy = model.evaluate(X_train, y_train)
print(f'Train Loss: {train_loss}')
print(f'Train Accuracy: {train_accuracy}')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Generate predictions
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Print classification report and confusion matrix
print("Test Set Classification Report:")
print(classification_report(y_test, y_pred))

print("Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Qualitative assessment
# Correct classifications
correct_indices = np.where(y_pred == y_test)[0]
correct_examples = X_test[correct_indices[:5]]  # Displaying only first 5 correct examples
print("Correctly classified examples:")
for i, example in enumerate(correct_examples):
    print(f"Example {i+1}: {example}")

# Incorrect classifications
incorrect_indices = np.where(y_pred != y_test)[0]
incorrect_examples = X_test[incorrect_indices[:5]]  # Displaying only first 5 incorrect examples
print("\nIncorrectly classified examples:")
for i, example in enumerate(incorrect_examples):
    print(f"Example {i+1}: {example}")

    # Plot the performance metrics
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.show()


