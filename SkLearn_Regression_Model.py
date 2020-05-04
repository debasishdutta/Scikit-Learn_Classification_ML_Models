#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Downloads/Python Code Library/SkLearn Model Codes/Datasets/Regression_Model_Dataset.csv" 
global_id_var = 'id'
global_dep_var = 'AvgBill'
global_test_split = 0.2
global_k_fold_cv = 5
global_seed = 1234

# Model Configurations (Linear Regression (OLS, Lasso, Ridge, Elasticnet), Random Forest, Gradient Boosting)
linear_reg_fit_intercept = [True, False]
linear_reg_normalize = [True, False]
linear_reg_alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
linear_reg_l1_ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
random_forest_n_tree = [50,100, 200, 300, 400, 500]
random_forest_max_depth = [3, 4, 5]
random_forest_min_sample_split = [8, 10, 12]
random_forest_min_sample_leaf = [4, 5, 6]
gbm_max_depth = [3, 4, 5]
gbm_min_sample_leaf = [4, 5, 6]
gbm_n_tree = [50,100, 200, 300, 400, 500]
gbm_learning_rate = [0.05,0.1,0.2,0.3,0.4,0.5]
xgb_min_child_weight = [1,3,6,9]
xgb_gamma = [0.5, 1, 1.5, 2, 5]
xgb_subsample = [0.6, 0.8, 1.0]
xgb_max_depth = [3,4,5]
xgb_learning_rate = [0.05,0.1,0.2,0.3,0.4,0.5]
xgb_n_estimators = [50,100,200,300,500]


# In[ ]:


### IMPORT ALL NECCESSARY PACKAGES ###

import warnings
from time import *
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


# In[ ]:


### USER DEFINED FUNCTION: RAW DATA IMPORT ###

def data_import(source_name, id_var, dep_var):
    
    import_start_time = time()
    
    print("\nKindly Follow The Log For Tracing The Modelling Process\n")
    
    df = pd.read_csv(global_source_name)
    
    df_x = df[df.columns[~df.columns.isin([id_var,dep_var])]]
    df_y = df.loc[:,dep_var]
    
    numeric_cols = df_x.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_x.select_dtypes(include=['object']).columns.tolist()
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)
    print("\nTime To Perform Data Import: %.3f Seconds\n" % import_elapsed_time)
    
    final_data_import = [df_x,df_y,numeric_cols,categorical_cols]
        
    return(final_data_import)


# In[ ]:


### USER DEFINED FUNCTION: TRAIN & TEST SAMPLE CREATION USING RANDOM SAMPLING ###

def random_sampling(x, y, split):
    
    sampling_start_time = time()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split, random_state=1000)
    
    sampling_end_time = time()
    sampling_elapsed_time = (sampling_end_time - sampling_start_time)
    print("\nTime To Perform Random Sampling For Train & Test Set: %.3f Seconds\n" % sampling_elapsed_time)
    
    final_sampling = [x_train, x_test, y_train, y_test]
        
    return(final_sampling)


# In[ ]:


### USER DEFINED FUNCTION: LINEAR REGRESSION (OLS) ###

def model_ols_linear_reg(train_x,
                         train_y, 
                         test_x, 
                         test_y, 
                         num_col_list, 
                         cat_col_list,
                         n_cv,
                         param_fit_intercept,
                         param_normalize):
    
    print("\nStarting Linear Regression(OLS) Model Devlopment\n")
    
    ols_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('ols', LinearRegression(n_jobs = -1))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'ols__fit_intercept':param_fit_intercept,
                        'ols__normalize': param_normalize}    
  
    # Model Development #
    ols_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    ols_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - ols_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, ols_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(ols_cv_model.score(test_x, test_y),2)*100
    
    ols_end_time = time()
    ols_elapsed_time = round(ols_end_time - ols_start_time,2)
    print("\nTime To Develop Linear Regression (OLS) Model: %.3f Seconds\n" % ols_elapsed_time)
    
    ols_model_stat = pd.DataFrame({"Model Name" : ["Linear Regression (OLS)"],
                                   "MAPE(%)": mape,
                                   "MSE": mse, 
                                   "RMSE": rmse,
                                   "R-Square(%)": r_sqr,
                                   "Time (Sec.)": ols_elapsed_time})
    final_result = (ols_cv_model,ols_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: LINEAR REGRESSION (LASSO) ###

def model_lasso_linear_reg(train_x,
                         train_y, 
                         test_x, 
                         test_y, 
                         num_col_list, 
                         cat_col_list,
                         n_cv,
                         param_fit_intercept,
                         param_normalize,
                         param_alpha):
    
    print("\nStarting Linear Regression(Lasso) Model Devlopment\n")
    
    lasso_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('lasso', Lasso(max_iter=100))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'lasso__fit_intercept': param_fit_intercept,
                        'lasso__normalize': param_normalize,
                        'lasso__alpha': param_alpha}    
  
    # Model Development #
    lasso_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    lasso_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - lasso_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, lasso_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(lasso_cv_model.score(test_x, test_y),2)*100
    
    lasso_end_time = time()
    lasso_elapsed_time = round(lasso_end_time - lasso_start_time,2)
    print("\nTime To Develop Linear Regression (Lasso) Model: %.3f Seconds\n" % lasso_elapsed_time)
    
    lasso_model_stat = pd.DataFrame({"Model Name" : ["Linear Regression (LASSO)"],
                                     "MAPE(%)": mape,
                                     "MSE": mse, 
                                     "RMSE": rmse,
                                     "R-Square(%)": r_sqr,
                                     "Time (Sec.)": lasso_elapsed_time})
    final_result = (lasso_cv_model,lasso_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: LINEAR REGRESSION (RIDGE) ###

def model_ridge_linear_reg(train_x,
                           train_y,
                           test_x, 
                           test_y, 
                           num_col_list, 
                           cat_col_list,
                           n_cv,
                           param_fit_intercept,
                           param_normalize,
                           param_alpha):
    
    print("\nStarting Linear Regression(Ridge) Model Devlopment\n")
    
    ridge_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('ridge', Ridge(max_iter=100))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'ridge__fit_intercept': param_fit_intercept,
                        'ridge__normalize': param_normalize,
                        'ridge__alpha': param_alpha,
                        'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}    
  
    # Model Development #
    ridge_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    ridge_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - ridge_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, ridge_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(ridge_cv_model.score(test_x, test_y),2)*100
    
    ridge_end_time = time()
    ridge_elapsed_time = round(ridge_end_time - ridge_start_time,2)
    print("\nTime To Develop Linear Regression (Ridge) Model: %.3f Seconds\n" % ridge_elapsed_time)
    
    ridge_model_stat = pd.DataFrame({"Model Name" : ["Linear Regression (RIDGE)"],
                                     "MAPE(%)": mape,
                                     "MSE": mse, 
                                     "RMSE": rmse,
                                     "R-Square(%)": r_sqr,
                                     "Time (Sec.)": ridge_elapsed_time})
    final_result = (ridge_cv_model,ridge_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: LINEAR REGRESSION (ELASTICNET) ###

def model_elasticnet_linear_reg(train_x,
                           train_y,
                           test_x, 
                           test_y, 
                           num_col_list, 
                           cat_col_list,
                           n_cv,
                           param_fit_intercept,
                           param_normalize,
                           param_alpha,
                           param_l1_ratio):
    
    print("\nStarting Linear Regression(ElasticNet) Model Devlopment\n")
    
    en_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('en', ElasticNet(max_iter=100))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'en__fit_intercept': param_fit_intercept,
                        'en__normalize': param_normalize,
                        'en__alpha': param_alpha,
                        'en__l1_ratio': param_l1_ratio}    
  
    # Model Development #
    en_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    en_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - en_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, en_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(en_cv_model.score(test_x, test_y),2)*100
    
    en_end_time = time()
    en_elapsed_time = round(en_end_time - en_start_time,2)
    print("\nTime To Develop Linear Regression (ElasticNet) Model: %.3f Seconds\n" % en_elapsed_time)
    
    en_model_stat = pd.DataFrame({"Model Name" : ["Linear Regression (ELASTICNET)"],
                                  "MAPE(%)": mape,
                                  "MSE": mse, 
                                  "RMSE": rmse,
                                  "R-Square(%)": r_sqr,
                                  "Time (Sec.)": en_elapsed_time})
    final_result = (en_cv_model,en_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: RANDOM FOREST MODEL ###

def model_random_forest(train_x, 
                        train_y, 
                        test_x, 
                        test_y, 
                        num_col_list, 
                        cat_col_list,
                        n_cv,
                        param_tree,
                        param_min_leaf_sample,
                        param_min_split_sample,
                        param_max_depth):
    
    print("\nStarting Random Forest Model Devlopment\n")
    
    rf_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('rf', RandomForestRegressor())])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'rf__n_estimators':param_tree,
                        'rf__max_features':['auto','sqrt','log2'],
                        'rf__min_samples_leaf':param_min_leaf_sample,
                        'rf__min_samples_split':param_min_split_sample,
                        'rf__max_depth':param_max_depth}
    
    # Model Development #
    rf_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    rf_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - rf_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, rf_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(rf_cv_model.score(test_x, test_y),2)*100

    rf_end_time = time()
    rf_elapsed_time = round(rf_end_time - rf_start_time,2)
    print("\nTime To Develop Random Foreest Model: %.3f Seconds\n" % rf_elapsed_time)
    
    rf_model_stat = pd.DataFrame({"Model Name" : ["Random Forest"],
                                  "MAPE(%)": mape,
                                  "MSE": mse, 
                                  "RMSE": rmse,
                                  "R-Square(%)": r_sqr,
                                  "Time (Sec.)": rf_elapsed_time})
    final_result = (rf_cv_model,rf_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: GRADIENT BOOSTING MODEL ###

def model_gradient_boosting(train_x,
                            train_y, 
                            test_x, 
                            test_y, 
                            num_col_list, 
                            cat_col_list,
                            n_cv,
                            param_max_depth,
                            param_min_Sample_leaf,
                            param_n_tree,
                            param_lr):
    
    print("\nStarting Gradient Boosting Model Devlopment\n")
    
    gbm_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('gbm', GradientBoostingRegressor(random_state=0))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'gbm__loss':['ls', 'lad', 'huber', 'quantile'],
                        'gbm__learning_rate': param_lr,
                        'gbm__n_estimators':param_n_tree,
                        'gbm__criterion':['friedman_mse','mae'],
                        'gbm__min_samples_leaf':param_min_Sample_leaf,
                        'gbm__max_depth':param_max_depth,
                        'gbm__max_features':['auto','sqrt','log2']}
    
    # Model Development #
    gbm_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    gbm_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - gbm_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, gbm_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(gbm_cv_model.score(test_x, test_y),2)*100
    
    gbm_end_time = time()
    gbm_elapsed_time = round(gbm_end_time - gbm_start_time,2)
    print("\nTime To Develop Gradient Boosting Model: %.3f Seconds\n" % gbm_elapsed_time)
    
    gbm_model_stat = pd.DataFrame({"Model Name" : ["Gradient Boosting"],
                                    "MAPE(%)": mape,
                                    "MSE": mse, 
                                    "RMSE": rmse,
                                    "R-Square(%)": r_sqr,
                                    "Time (Sec.)": gbm_elapsed_time})
    final_result = (gbm_cv_model,gbm_model_stat)
    
    return(final_result)


# In[ ]:


### USER DEFINED FUNCTION: XGBOOST MODEL ###

def model_xgboost(train_x,
                  train_y,
                  test_x,
                  test_y,
                  num_col_list,
                  cat_col_list,
                  n_cv,
                  param_min_child_weight,
                  param_gamma,
                  param_subsample,
                  param_max_depth,
                  param_learning_rate,
                  param_n_estimators):
    
    print("\nStarting XGBoost Model Devlopment\n")
    
    xgb_start_time = time()
    
    # Preprocessing Step For Numeric Variables #
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
                                          ('scaler', StandardScaler(with_mean = True, with_std = True))])
    # Preprocessing Step For Categorical Variables #
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # Combining All Preprocessing Step #
    final_preprocessor = ColumnTransformer(transformers=[('preprocessing_num_col', numeric_transformer, num_col_list),
                                                         ('preprocessing_cat_col', categorical_transformer, cat_col_list)])
    # Modelling Pipeline Creation #
    final_pipeline = Pipeline(steps=[('preprocessor', final_preprocessor),
                                     ('xgb', XGBRegressor(random_state=0, objective='reg:squarederror'))])
                                     
    # Hyper Parameter Tuning #
    hyper_parameters = {'xgb__min_child_weight': param_min_child_weight,
                        'xgb__gamma': param_gamma,
                        'xgb__subsample': param_subsample,
                        'xgb__max_depth': param_max_depth,
                        'xgb__learning_rate': param_learning_rate,
                        'xgb__n_estimators': param_n_estimators}
    
    # Model Development #
    xgb_cv_model = GridSearchCV(final_pipeline, param_grid=hyper_parameters, cv=n_cv, scoring='r2')
    xgb_cv_model.fit(train_x, train_y)
    
    mape = round(np.mean(np.abs((test_y - xgb_cv_model.predict(test_x)) / test_y)) * 100,2)
    mse = round(np.sqrt(mean_squared_error(test_y, xgb_cv_model.predict(test_x))),2)
    rmse = round(sqrt(mse),2)
    r_sqr = round(xgb_cv_model.score(test_x, test_y),2)*100
    
    xgb_end_time = time()
    xgb_elapsed_time = round(xgb_end_time - xgb_start_time,2)
    print("\nTime To Develop XGBoost Model: %.3f Seconds\n" % xgb_elapsed_time)
    
    xgb_model_stat = pd.DataFrame({"Model Name" : ["XGBoost"],
                                    "MAPE(%)": mape,
                                    "MSE": mse, 
                                    "RMSE": rmse,
                                    "R-Square(%)": r_sqr,
                                    "Time (Sec.)": xgb_elapsed_time})
    final_result = (xgb_cv_model,xgb_model_stat)
    
    return(final_result)

# In[ ]:


### SCRIPT EXECUTION ###

warnings.filterwarnings("ignore")

# Data Import #
result_import = data_import(global_source_name, global_id_var, global_dep_var)

# Random Sampling of Test & Train Data #
result_sampling = random_sampling(result_import[0], result_import[1], global_test_split)

# Linear Regression (OLS) Model #
result_ols_model = model_ols_linear_reg(result_sampling[0],
                                        result_sampling[2],
                                        result_sampling[1],
                                        result_sampling[3],
                                        result_import[2],
                                        result_import[3],
                                        global_k_fold_cv,
                                        linear_reg_fit_intercept,
                                        linear_reg_normalize)

# Linear Regression (Lasso) Model #
result_lasso_model = model_lasso_linear_reg(result_sampling[0],
                                            result_sampling[2],
                                            result_sampling[1],
                                            result_sampling[3],
                                            result_import[2],
                                            result_import[3],
                                            global_k_fold_cv,
                                            linear_reg_fit_intercept,
                                            linear_reg_normalize,
                                            linear_reg_alpha)

# Linear Regression (Ridge) Model #
result_ridge_model = model_ridge_linear_reg(result_sampling[0],
                                            result_sampling[2],
                                            result_sampling[1],
                                            result_sampling[3],
                                            result_import[2],
                                            result_import[3],
                                            global_k_fold_cv,
                                            linear_reg_fit_intercept,
                                            linear_reg_normalize,
                                            linear_reg_alpha)

# Linear Regression (ElasticNet) Model #
result_elasticnet_model = model_elasticnet_linear_reg(result_sampling[0],
                                                      result_sampling[2],
                                                      result_sampling[1],
                                                      result_sampling[3],
                                                      result_import[2],
                                                      result_import[3],
                                                      global_k_fold_cv,
                                                      linear_reg_fit_intercept,
                                                      linear_reg_normalize,
                                                      linear_reg_alpha,
                                                      linear_reg_l1_ratio)

# Random Forest Model #
result_rf_model = model_random_forest(result_sampling[0],
                                      result_sampling[2],
                                      result_sampling[1],
                                      result_sampling[3],
                                      result_import[2],
                                      result_import[3],
                                      global_k_fold_cv,
                                      random_forest_n_tree,
                                      random_forest_min_sample_leaf,
                                      random_forest_min_sample_split,
                                      random_forest_max_depth)

# Gradient Boosting Machine Model #
result_gbm_model = model_gradient_boosting(result_sampling[0],
                                        result_sampling[2],
                                        result_sampling[1],
                                        result_sampling[3],
                                        result_import[2],
                                        result_import[3],
                                        global_k_fold_cv,
                                        gbm_max_depth,
                                        gbm_min_sample_leaf,
                                        gbm_n_tree,
                                        gbm_learning_rate)

# XGBoost Model #
result_xgb_model = model_xgboost(result_sampling[0],
                                 result_sampling[2],
                                 result_sampling[1],
                                 result_sampling[3],
                                 result_import[2],
                                 result_import[3],
                                 global_k_fold_cv,
                                 xgb_min_child_weight,
                                 xgb_gamma,
                                 xgb_subsample,
                                 xgb_max_depth,
                                 xgb_learning_rate,
                                 xgb_n_estimators)

# Collecting All Model Output #
print("\n++++++ Overall Model Summary ++++++\n")
all_model_summary = pd.DataFrame()

all_model_summary = all_model_summary.append(result_ols_model[1],ignore_index=True).append(result_lasso_model[1],ignore_index=True).append(result_ridge_model[1],ignore_index=True).append(result_elasticnet_model[1],ignore_index=True).append(result_rf_model[1],ignore_index=True).append(result_gbm_model[1],ignore_index=True).append(result_xgb_model[1],ignore_index=True)
display(all_model_summary)

print("\n++++++ Process Completed ++++++\n")