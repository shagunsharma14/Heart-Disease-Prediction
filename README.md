# Heart Disease Prediction

This project aims to predict the presence of heart disease based on various medical attributes. It utilizes machine learning techniques, particularly logistic regression, to build a predictive model. The steps involved in this project are outlined below:

## Importing the Dependencies
The necessary libraries are imported to support the implementation of the project. These include:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Collection and Processing
The dataset containing the required information is loaded into a Pandas DataFrame. Various operations are performed on the dataset to gain insights and prepare it for training the model. These operations include:
```python
# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('/content/data.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()
```

## Splitting the Features and Target
The features and target variable are separated from the dataset. The features are stored in `X`, while the target variable is stored in `Y`.
```python
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)
print(Y)
```

## Splitting the Data into Training and Test Data
The dataset is split into training data and test data using the `train_test_split` function. The split is performed with a test size of 20% and stratified sampling to maintain the proportion of the target variable in both the training and test sets.
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
```

## Model Training
The logistic regression model is instantiated and trained using the training data.
```python
model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)
```

## Model Evaluation
The accuracy of the model is evaluated using the training and test data. The accuracy score is calculated for both sets.
```python
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data: ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data: ', test_data_accuracy)
```

## Building a Predictive System
A predictive system is built using the trained model. A sample input is provided, and the model predicts whether the person has heart disease or not.
```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

