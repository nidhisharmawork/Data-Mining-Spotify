#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "C:\\Users\\nidhi\\Downloads\\Data1.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the initial dataset
print("Initial Dataset (First Few Rows):")
print(df.head())

# Qualitative Summary
print("\nQualitative Summary:")
print(df.info())  # Display information about the DataFrame, including data types and non-null counts

# Handling Missing Values
print("\nHandling Missing Values:")
print("Number of missing values per column:")
print(df.isnull().sum())  # Display the count of missing values for each column

# Drop rows with missing values
df.dropna(inplace=True)

# Variable-specific processing (if needed)
# For example, converting categorical variables to numerical, handling outliers, etc.

# Rename columns
df.rename(columns={'Payroll#': 'Payroll', 'Promoted?': 'Promoted', 'Ftime': 'Full-Time'}, inplace=True)

# Check for Duplicates
print("\nCheck for Duplicates:")
print("Number of duplicate rows:", df.duplicated(subset='Employee Id').sum())

# Remove duplicate rows based on 'Employee Id'
df.drop_duplicates(subset='Employee Id', inplace=True)

# Display the first few rows of the pre-processed dataset
print("\nPre-processed Dataset (First Few Rows):")
print(df.head())




# In[26]:


# Specify the correct column names in your dataset
features = ['Field Experience', 'Training', 'Full-Time', 'IsMale', 'AnnualSalary', 'AnnualBenefits']
target = 'Promoted'

# Extract features and target variable
X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess features (e.g., scaling for kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification Tree
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

# Predictions on the test set
tree_predictions = tree_classifier.predict(X_test)

# Evaluate the Decision Tree model
print("Decision Tree Model:")
print("Accuracy:", accuracy_score(y_test, tree_predictions))
print("Classification Report:")
print(classification_report(y_test, tree_predictions))

# k-Nearest Neighbors (kNN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# Predictions on the scaled test set
knn_predictions = knn_classifier.predict(X_test_scaled)

# Evaluate the kNN model
print("\nk-Nearest Neighbors (kNN) Model:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Classification Report:")
print(classification_report(y_test, knn_predictions))

#Random Forests
from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test, rf_predictions))


# In[27]:


# Specify relevant features and target variable
features = ['Field Experience', 'Training', 'Full-Time', 'IsMale', 'AnnualSalary', 'AnnualBenefits']
target = 'Promoted'

# Extract features and target variable
X = df[features]
y = df[target]

# Convert categorical variables to dummy/indicator variables (if needed)
X = pd.get_dummies(X, columns=['Full-Time', 'IsMale'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_train_scaled, y_train)

# Predictions on the test set
logit_predictions = logit_model.predict(X_test_scaled)

# Evaluate the Logistic Regression model
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, logit_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logit_predictions))
print("Classification Report:")
print(classification_report(y_test, logit_predictions)) 

