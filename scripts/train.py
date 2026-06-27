import os 
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from src.data_loader import X_train, X_test, y_train, y_test 
from src.model import ridge_train_pipeline, random_forest_pipeline, gradient_boosting_pipeline

def train_ridge_baseline_model():
    ridge = ridge_train_pipeline()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Ridge Baseline test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

def train_random_forest_pipeline():
    rf = random_forest_pipeline()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Random Forest test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

def train_gradient_boosting_pipeline():
    gb = gradient_boosting_pipeline()
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Gradient Boosting test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


    
    
