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

folder_path = "Training_final" 
"""
# Load the feature importance data
feature_importance_df = pd.read_csv('feature_importance_0910.csv')

# Sort the features by importance in descending order and select the top 50
top_50_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(100)
# Print the top 50 features
selected_features=top_50_features['Feature'].tolist()

# Save the selected features to a text file
with open('selected_features_mic.txt', 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print("Selected features saved to 'selected_features_mic.txt'")
"""

selected_features=['days_since_last_login' 'login_pmpm_ct' 'login_count_4'
 'rwjf_premature_death_rate' 'rwjf_preventable_ip_rate'
 'rwjf_teen_births_rate' 'rwjf_poor_phy_hlth_days'
 'rwjf_poor_men_hlth_days' 'rwjf_income_inequ_ratio'
 'rwjf_std_infect_rate' 'rwjf_population' 'rwjf_median_house_income'
 'rwjf_premature_mortality' 'rwjf_injury_deaths_rate'
 'rwjf_social_associate_rate' 'rwjf_life_expectancy'
 'rwjf_healthcare_cost' 'nonpar_deduct_pmpm_cost' 'nonpar_ds_clm'
 'oontwk_ds_clm' 'total_allowed_pmpm_cost' 'total_net_paid_pmpm_cost'
 'total_deduct_pmpm_cost' 'cnt_cp_emails_0' 'cnt_cp_emails_1'
 'cnt_cp_emails_2' 'cnt_cp_emails_3' 'cnt_cp_emails_4' 'cnt_cp_emails_5'
 'cnt_cp_emails_6' 'cnt_cp_emails_7' 'cnt_cp_emails_8' 'cnt_cp_emails_9'
 'cnt_cp_emails_10' 'cnt_cp_emails_11' 'cnt_cp_print_1' 'cnt_cp_print_3'
 'cnt_cp_print_4' 'cnt_cp_print_5' 'cnt_cp_print_6' 'cnt_cp_print_7'
 'cnt_cp_print_8' 'cnt_cp_print_9' 'cnt_cp_print_10' 'cnt_cp_print_11'
 'cnt_cp_vat_0' 'cnt_cp_vat_1' 'cnt_cp_vat_2' 'cnt_cp_vat_3'
 'cnt_cp_vat_4' 'cnt_cp_vat_5' 'cnt_cp_vat_6' 'cnt_cp_vat_7'
 'cnt_cp_vat_8' 'cnt_cp_vat_9' 'cnt_cp_vat_10' 'cnt_cp_vat_11'
 'cnt_cp_emails_pmpm_ct' 'cnt_cp_webstatement_pmpm_ct'
 'plan_benefit_package_id' 'pbp_segment_id' 'rx_overall_deduct_pmpm_cost'
 'rx_overall_gpi_pmpm_ct' 'rx_overall_dist_gpi6_pmpm_ct'
 'rx_tier_3_pmpm_ct' 'rx_tier_4_pmpm_ct' 'rx_days_since_last_script'
 'riskarr_upside' 'fci_score' 'dcsi_score']

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

y=prev_df['preventive_visit_gap_ind']
X = prev_df.drop(columns=['id','preventive_visit_gap_ind'])
del prev_df
# Remove one feature from X if its correlation with another feature is greater than 0.8
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
X = X.drop(columns=to_drop)
# Consider only the first 50 features from X
print(X.shape)
exit()
X = X.iloc[:, :50]

# Write all features in X to a text file
#with open('selected_features_mic_filtered.txt', 'w') as f:
#    for feature in X.columns:
#        f.write(f"{feature}\n")




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load your data
# Replace this with your actual data loading
# X is your feature set, y is your target variable ('t

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
catboost_model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)

# Train the base models
print("Training base models")
rf_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

# Get probability predictions from the base models (for the training set)
rf_train_preds = rf_model.predict_proba(X_train)[:, 1]  # Probability of class 1
lgbm_train_preds = lgbm_model.predict_proba(X_train)[:, 1]
catboost_train_preds = catboost_model.predict_proba(X_train)[:, 1]

# Stack predictions as features for meta-learner
stacked_train = np.column_stack((rf_train_preds, lgbm_train_preds, catboost_train_preds))

# Initialize and train the meta-learner (Logistic Regression)
print("Training meta-learner")
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_train)

# Get probability predictions from the base models (for the test set)
rf_test_preds = rf_model.predict_proba(X_test)[:, 1]
lgbm_test_preds = lgbm_model.predict_proba(X_test)[:, 1]
catboost_test_preds = catboost_model.predict_proba(X_test)[:, 1]

# Stack predictions for the test set
stacked_test = np.column_stack((rf_test_preds, lgbm_test_preds, catboost_test_preds))

# Get final predictions from the meta-model
print("Getting final predictions from the meta-model")
meta_test_preds = meta_model.predict_proba(stacked_test)[:, 1]

# Evaluate the model using AUC-ROC score
auc_roc = roc_auc_score(y_test, meta_test_preds)
print(f"AUC-ROC score of the Meta-Ensemble Stacking Model: {auc_roc:.4f}")

