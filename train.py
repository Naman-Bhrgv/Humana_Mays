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

folder_path = "Training_final"  # Replace with your folder path
c=0
for filename in os.listdir(folder_path):
    print(c)
    print(filename)
    df=pd.read_csv(os.path.join(folder_path, filename), nrows=1069533)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            
            df = df.drop(col, axis=1)
        
    
    if filename=="humana_mays_target_members.csv":
      tgt=df['preventive_visit_gap_ind']
      
      df_1=df[['id','preventive_visit_gap_ind']]
      df=df.drop('preventive_visit_gap_ind', axis=1)
    
    selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
    selector.fit(df)

    # Get the boolean mask of selected features
    selected_features_mask = selector.get_support()

    # Filter your DataFrame based on the mask
    df = df.loc[:, selected_features_mask]
    
    if c==0:
        prev_df=df

    else:
        prev_df=pd.merge(prev_df, df, on='id', how='inner')

    if filename=="humana_mays_target_members.csv":

        prev_df=pd.merge(prev_df, df_1, on='id', how='inner')
    
    del df
    c+=1

prev_df = prev_df.dropna()

print(prev_df.shape)

exit()

X = prev_df.drop(columns=['preventive_visit_gap_ind'])
y = prev_df['preventive_visit_gap_ind']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_tr_id=X_train['id']
x_te_id=X_test['id']

X_train=X_train.drop(columns=['id'])
X_test=X_test.drop(columns=['id'])

print("Data splitting done.")
del X

del y

print("RF feature selection started.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:50]  # Top 50 features
print(indices)

X_train = X_train.iloc[:, indices]  

print("Relevant Features-")

print(X_train.columns)

# Save list of X_train columns to text file
with open('selected_features.txt', 'w') as f:
    for column in X_train.columns:
        f.write(f"{column}\n")

print("Relevant features saved to 'selected_features.txt'")


X_test = X_test.iloc[:, indices]  

print("Hyperparameter Tuning-")
param_grid={            'n_estimators': [75 ,100, 150, 200, 250, 300],
                          'eta':[0.2,0.1,0.3,0.5],
                          'gamma':[0.1,0.15,0.01,0.05,0.2],
                          'max_depth':[8,9,10,6]
                          ,'subsample':[0.6,0.5,0.7,0.9,1],'colsample_bytree':[0.8,0.7,0.6,0.5,1]}

xgb_cl = xgb.XGBClassifier()

cv=RandomizedSearchCV(estimator=xgb_cl,param_distributions=param_grid,n_iter=10,cv=3,scoring='f1')

cv.fit(X_train, y_train)
best_params= cv.best_params_


print("Training Xgboost model-")
# Create an XGBoost classifier

#best_xgb_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
#best_xgb_model.fit(X_train, y_train)

model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='auc')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(f'AUC Score: {roc_auc:.2f}')

# Save XGBoost model as pickle file
import pickle

# Define the filename for the pickle file
model_filename = 'xgboost_model.pkl'

# Save the model to a file
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"XGBoost model saved as '{model_filename}'")

#Disparity
df1=pd.read_csv("Training_final/humana_mays_target_member_details.csv")

df1=df1[['id','race','sex_cd']]

X_test['id']=x_te_id

df=pd.merge(X_test, df1, on='id', how='inner')

del df1

# Calculate disparity scores for race and sex
def calculate_disparity_score(df, y_prob, group_column):
    df['predicted_prob'] = y_prob
    disparity_scores = {}
    overall_mean = df['predicted_prob'].mean()
    
    for group in df[group_column].unique():
        group_mean = df[df[group_column] == group]['predicted_prob'].mean()
        disparity_scores[group] = group_mean / overall_mean
    
    return disparity_scores

race_disparity = calculate_disparity_score(df, y_prob, 'race')
sex_disparity = calculate_disparity_score(df, y_prob, 'sex_cd')

print("Race Disparity Scores:")
for race, score in race_disparity.items():
    print(f"{race}: {score:.4f}")

print("\nSex Disparity Scores:")
for sex, score in sex_disparity.items():
    print(f"{sex}: {score:.4f}")

# Calculate weights
def calculate_weights(disparity_scores):
    return {group: 1 / score for group, score in disparity_scores.items()}

race_weights = calculate_weights(race_disparity)
sex_weights = calculate_weights(sex_disparity)

# Apply weights to the dataset
df['race_weight'] = df['race'].map(race_weights)
df['sex_weight'] = df['sex_cd'].map(sex_weights)

# Calculate combined weight
df['combined_weight'] = (df['race_weight'] + df['sex_weight']) / 2

print("\nSample of weights applied:")
print(df[['id', 'race', 'sex_cd', 'race_weight', 'sex_weight', 'combined_weight']].head())

# Calculate overall disparity score
overall_disparity_score = df['combined_weight'].mean()
print(f"\nOverall Disparity Score: {overall_disparity_score:.4f}")


