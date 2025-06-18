# Importing required libraries for data handling, modeling, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing machine learning tools from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Read the dataset
df = pd.read_csv("KDDTrain_filtered.csv")

# Check for categorical columns (object dtype)
categorical_columns = df.select_dtypes(include=['object']).columns

# Print them to understand what needs encoding
print("Categorical Columns:", categorical_columns)

# Convert categorical features to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Separate features and target after encoding
X = df_encoded.drop("label_normal.", axis=1, errors="ignore")  # adjust this if your label becomes one-hot too
y = df["label"]  # or use encoded label if needed

# Encode label column separately
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
