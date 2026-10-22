# ================================
# Crop Production Prediction Model
# Decision Tree with Pruning
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

import tkinter as tk
from tkinter import messagebox


# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("data.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# ================================
# 2. DATA CLEANING
# ================================

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

# Drop rows where Production is missing
df = df.dropna(subset=['Production'])

# Fill remaining missing values
df = df.fillna(0)

# ================================
# 3. ENCODE CATEGORICAL DATA
# ================================

le = LabelEncoder()

categorical_cols = ['State', 'District', 'Season', 'Crop']

for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# ================================
# 4. DEFINE FEATURES & TARGET
# ================================

# Remove duplicate 'Season' if exists twice
if df.columns.tolist().count('Season') > 1:
    df = df.loc[:, ~df.columns.duplicated()]

X = df.drop(['Production'], axis=1)
y = df['Production']

# ================================
# 5. TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 6. TRAIN INITIAL MODEL
# ================================

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nBefore Pruning:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ================================
# 7. APPLY PRUNING
# ================================

path = model.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

models = []

for alpha in alphas:
    m = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    m.fit(X_train, y_train)
    models.append(m)

# ================================
# 8. SELECT BEST MODEL
# ================================

best_model = None
best_score = -np.inf

for m in models:
    y_pred = m.predict(X_test)
    score = r2_score(y_test, y_pred)

    if score > best_score:
        best_score = score
        best_model = m

print("\nAfter Pruning:")
print("Best R2 Score:", best_score)

# ================================
# 9. FINAL EVALUATION
# ================================

y_pred = best_model.predict(X_test)

print("\nFinal Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ================================
# 10. PREDICT NEW DATA
# ================================

# Example: Take last row as sample
sample = X.iloc[-1].values.reshape(1, -1)

prediction = best_model.predict(sample)

print("\nPredicted Production for Sample:", prediction[0])

# ================================
# 11. FEATURE IMPORTANCE (OPTIONAL)
# ================================

import matplotlib.pyplot as plt

importance = best_model.feature_importances_
feature_names = X.columns

# Sort features
indices = np.argsort(importance)[-10:]  # Top 10 features

plt.figure()
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Importance")
plt.show()
print(df.head())
print(df.dtypes)