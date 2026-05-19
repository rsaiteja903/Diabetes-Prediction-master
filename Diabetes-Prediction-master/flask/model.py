#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. Load dataset
dataset = pd.read_csv('diabetes.csv')

# We will use 4 features:
# 1. Glucose (col 1)
# 2. Insulin (col 4)
# 3. BMI (col 5)
# 4. Age (col 7)
X = dataset.iloc[:, [1, 4, 5, 7]].values  # Glucose, Insulin, BMI, Age
y = dataset.iloc[:, 8].values             # Outcome

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=dataset['Outcome']
)

# 3. Build pipeline: scaler + classifier
model = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('svc', SVC(kernel='linear', random_state=42))
])

# 4. Train
model.fit(X_train, y_train)

# Optional: check accuracy
print("Test accuracy:", model.score(X_test, y_test))

# 5. Save the whole pipeline
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Quick manual test
sample = np.array([[86, 120, 26.6, 31]])  # [Glucose, Insulin, BMI, Age]
print("Sample prediction:", model.predict(sample))