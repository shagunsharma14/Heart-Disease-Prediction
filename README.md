# Heart Disease Prediction

This project aims to predict the presence of heart disease based on various medical attributes. It utilizes machine learning techniques, particularly logistic regression, to build a predictive model. The steps involved in this project are outlined below:

## Importing the Dependencies
The necessary libraries are imported to support the implementation of the project. These include:
- `numpy` for numerical computations.
- `pandas` for data manipulation and analysis.
- `train_test_split` from `sklearn.model_selection` to split the dataset into training and testing sets.
- `LogisticRegression` from `sklearn.linear_model` to train the logistic regression model.
- `accuracy_score` from `sklearn.metrics` to evaluate the accuracy of the model.

## Data Collection and Processing
The dataset containing the required information is loaded into a Pandas DataFrame. Various operations are performed on the dataset to gain insights and prepare it for training the model. These operations include:
- Displaying the first and last five rows of the dataset.
- Determining the number of rows and columns in the dataset.
- Obtaining information about the dataset, such as column data types.
- Checking for missing values in the dataset.
- Computing statistical measures about the dataset.
- Examining the distribution of the target variable.

## Splitting the Features and Target
The features and target variable are separated from the dataset. The features are stored in `X`, while the target variable is stored in `Y`.

## Splitting the Data into Training and Test Data
The dataset is split into training data and test data using the `train_test_split` function. The split is performed with a test size of 20% and stratified sampling to maintain the proportion of the target variable in both the training and test sets.

## Model Training
The logistic regression model is instantiated and trained using the training data.

## Model Evaluation
The accuracy of the model is evaluated using the training and test data. The accuracy score is calculated for both sets.

## Building a Predictive System
A predictive system is built using the trained model. A sample input is provided, and the model predicts whether the person has heart disease or not.

Please refer to the code provided in the file for the detailed implementation.

**Note:** The accuracy of the model and the prediction results may vary depending on the dataset and the specific circumstances of the project.