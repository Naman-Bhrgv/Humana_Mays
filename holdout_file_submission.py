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

folder_path = "Holdout_final"  # Replace with your folder path


# Read selected_features.txt and store all the strings in a list
selected_features = []
with open('selected_features.txt', 'r') as file:
    selected_features = [line.strip() for line in file]

#print(selected_features)
c=0
for filename in os.listdir(folder_path):
    print(c)
    print(filename)
    df=pd.read_csv(os.path.join(folder_path, filename))
    #print(df.shape)
    # Drop all the features from df that do not lie in selected_features list
    df = df[[col for col in df.columns if col in selected_features]]

    if c==0:
        prev_df=df
    else:
        prev_df=pd.merge(prev_df, df, on='id', how='inner')
    del df
    c+=1

    import pickle

# Load the saved XGBoost model from the pickle file
model_filename = 'xgboost_model.pkl'

print(len(prev_df.columns)==len(selected_features))


with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Ensure 'id' column is not used for predictions
X_holdout = prev_df.drop(columns=['id'])
X_holdout = X_holdout[model.feature_names_in_]
# Make predictions using the loaded model
y_prob = model.predict_proba(X_holdout)[:, 1]

# Add predictions to the DataFrame
prev_df['SCORE'] = y_prob

# Create an additional column RANK based on prev_df['SCORE']
prev_df['RANK'] = prev_df['SCORE'].rank(ascending=False, method='first').astype(int)

final_sub=prev_df[['id','RANK','SCORE']]
del prev_df
# Save the DataFrame with predictions to a new CSV file
output_filename = 'holdout_predictions.csv'
final_sub.to_csv(output_filename, index=False)

print(f"Predictions saved to '{output_filename}'")