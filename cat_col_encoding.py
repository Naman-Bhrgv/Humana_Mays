from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import copy

#folder_path = "Training_final/Additional Features.csv"  # Replace with your folder path

folder_path = "Training_final"
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        print(f"Processing file: {filename}")
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                l = df[col]
                print("DISCRETE COLUMN")
                print(col)
                print("Number of unique values: ", len(l.unique()))
                print("Unique values: ", l.unique())
                null_count = df[col].isnull().sum()
                total_count = len(df[col])
                null_percentage = (null_count / total_count) * 100
                print(f"Number of null values in column '{col}': {null_count}")
                print(f"Percentage of null values in column '{col}': {null_percentage:.2f}%")
                variance = df[col].value_counts(normalize=True)
                print(f"Variance in column '{col}': {variance}")
                print("-"*10)
