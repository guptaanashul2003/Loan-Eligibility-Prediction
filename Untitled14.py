#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Project 2: Simple Linear Regression on Housing Prices

# -------------------------------
# Step 1: Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Step 2: Load Dataset
# -------------------------------
# Using sklearn's built-in Boston Housing dataset (but it's deprecated),
# so we will use California Housing dataset as modern replacement.
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# -------------------------------
# Step 3: Data Exploration
# -------------------------------
print("\nSummary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Step 4: Features and Target
# -------------------------------
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 5: Model Training
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Predictions & Evaluation
# -------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)

# Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# In[ ]:




