# TAMU Humana Mays Healthcare Analytics Competition

This repository is our team submission for Humana Mays Healthcare Analytics competition where we had to predict the probability of Medicare insurance customer skipping Primary Centre Physician visit. <br>
Key contributions- <br>
- Achieved 59th rank among 280+ teams in predicting patients skipping primary physician visit. <br>
- Achieved AUC score of 0.71 and disparity score of 0.98through Xgboost model trained over data of 1.5 million Medicare Insurance customers. <br>

File Description- <br>

- cat_col_encoding.py - Analyze categorical columns across the dataset and encode relevant columns. <br>
- eda.ipynb - Handle missing values in the dataset. <br>
- feat_corr.py- Remove highly correlated features. <br> 
- holdout_pred.ipynb- Make predictions on holdout dataset. <br>
- light_gbm.py - Build & train lightgbm model. <br>
- train.py - Feature Selection & training Xgboost model. <br>
- xgboost_model_2.pkl - saved xgboost model for final prediction.
