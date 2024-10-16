import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import roc_auc_score  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
"""
# -*- coding: utf-8 -*-
"""

import pandas as pd
import os
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import roc_auc_score  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score

selected_features=['days_since_last_login', 'login_pmpm_ct', 'login_count_4', 'rwjf_premature_death_rate', 'rwjf_preventable_ip_rate', 'rwjf_teen_births_rate', 'rwjf_poor_phy_hlth_days', 'rwjf_poor_men_hlth_days', 'rwjf_income_inequ_ratio', 'rwjf_std_infect_rate', 'rwjf_population', 'rwjf_median_house_income', 'rwjf_premature_mortality', 'rwjf_injury_deaths_rate', 'rwjf_social_associate_rate', 'rwjf_life_expectancy', 'rwjf_healthcare_cost', 'nonpar_deduct_pmpm_cost', 'nonpar_ds_clm', 'oontwk_ds_clm', 'total_allowed_pmpm_cost', 'total_net_paid_pmpm_cost', 'total_deduct_pmpm_cost', 'cnt_cp_emails_0', 'cnt_cp_emails_1', 'cnt_cp_emails_2', 'cnt_cp_emails_3', 'cnt_cp_emails_4', 'cnt_cp_emails_5', 'cnt_cp_emails_6', 'cnt_cp_emails_7', 'cnt_cp_emails_8', 'cnt_cp_emails_9', 'cnt_cp_emails_10', 'cnt_cp_emails_11', 'cnt_cp_print_1', 'cnt_cp_print_3', 'cnt_cp_print_4', 'cnt_cp_print_5', 'cnt_cp_print_6', 'cnt_cp_print_7', 'cnt_cp_print_8', 'cnt_cp_print_9', 'cnt_cp_print_10', 'cnt_cp_print_11', 'cnt_cp_vat_0', 'cnt_cp_vat_1', 'cnt_cp_vat_2', 'cnt_cp_vat_3', 'cnt_cp_vat_4', 'cnt_cp_vat_5', 'cnt_cp_vat_6', 'cnt_cp_vat_7', 'cnt_cp_vat_8', 'cnt_cp_vat_9', 'cnt_cp_vat_10', 'cnt_cp_vat_11', 'cnt_cp_emails_pmpm_ct', 'cnt_cp_webstatement_pmpm_ct', 'plan_benefit_package_id', 'pbp_segment_id', 'rx_overall_deduct_pmpm_cost', 'rx_overall_gpi_pmpm_ct', 'rx_overall_dist_gpi6_pmpm_ct', 'rx_tier_3_pmpm_ct', 'rx_tier_4_pmpm_ct', 'rx_days_since_last_script', 'riskarr_upside', 'fci_score', 'dcsi_score']

folder_path="Training_final"
df_tr=pd.read_csv("Training_final/humana_mays_target_members.csv")
l=df_tr['id'].unique()
df_tr=df_tr[['id','preventive_visit_gap_ind']]
rel_id=l

# Load the CSV file in chunks (optional if the file is large)
chunksize = 10000  # specify a suitable chunk size based on your memory capacity


# Read the CSV file in chunks and filter by ids in the list l

prev_df=pd.DataFrame()
for filename in os.listdir(folder_path):

    filtered_rows = []
    print(filename)

    for chunk in pd.read_csv(os.path.join(folder_path, filename), chunksize=chunksize):
        filtered_chunk = chunk[chunk['id'].isin(rel_id)]  # filter rows where id is in list l
        filtered_rows.append(filtered_chunk)

    # Concatenate the filtered chunks into a single DataFrame
    filtered_df = pd.concat(filtered_rows)
    # Remove rows with duplicate id in filtered_df
    filtered_df = filtered_df.drop_duplicates(subset='id')
    print("Filtered df shape: ",filtered_df.shape)

    # Keep only features in filtered_df which belong to selected_features list
    filtered_df = filtered_df[['id'] + [col for col in filtered_df.columns if col in selected_features]]

    if prev_df.empty:
        prev_df=filtered_df
    else:
        prev_df=pd.merge(prev_df,filtered_df,on=['id'],how='inner')
        print("Prev df shape: ",prev_df.shape)


prev_df=pd.merge(prev_df,df_tr,on=['id'],how='inner')
print(prev_df.shape)
# Sample 99238 rows randomly from prev_df

for col in prev_df.columns:
    if not pd.api.types.is_numeric_dtype(prev_df[col]):
        print("Categorical column-",col)
        le = LabelEncoder()
        prev_df[col] = le.fit_transform(prev_df[col].astype(str))
        # Save the label encoder
        with open(f'label_encoder_{col}.pkl', 'wb') as le_file:
            pickle.dump(le, le_file)

from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

# Prepare the data

y = prev_df['preventive_visit_gap_ind']
X = prev_df.drop(columns=['id','preventive_visit_gap_ind'])

scaler = StandardScaler()

# Fit and transform the DataFrame
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

from sklearn.utils import shuffle

# Shuffle X and y
X, y = shuffle(X, y, random_state=42)

# Sample 20% of the rows
sample_size = int(0.2 * len(X))
X_sampled = X.sample(n=sample_size, random_state=42)
y_sampled = y.loc[X_sampled.index]

print("Hyperparameter Tuning-")
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, KFold

# Define the parameter grid for LightGBM

param_grid_lgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 150],
    'max_depth': [3, 5, 7],
    'num_leaves': [20, 30, 40],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.01, 0.1, 1],
}

# Initialize the LightGBM classifier
lgb_cl = lgb.LGBMClassifier()

# Perform RandomizedSearchCV to find the best hyperparameters
cv_lgb = RandomizedSearchCV(estimator=lgb_cl, param_distributions=param_grid_lgb, n_iter=80, cv=3, scoring='roc_auc', random_state=42,verbose=2)
cv_lgb.fit(X_sampled, y_sampled)
best_params_lgb = cv_lgb.best_params_

print("Best parameters found for LightGBM: ", best_params_lgb)


