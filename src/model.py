# pyrefly: ignore [missing-import]
from sklearn.linear_model import Ridge 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def ridge_train_pipeline():
    ridge_baseline = Ridge(alpha=1.0)
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', ridge_baseline)
    ])
    return ridge_pipeline

def random_forest_pipeline():
    random_forest = RandomForestRegressor(n_estimators=300,       
    max_depth=15,           
    min_samples_leaf=5,   
    min_samples_split=10,   
    max_features=0.7,       
    n_jobs=-1,              
    random_state=42)
    random_forest_pipeline = Pipeline([
        ('regression', random_forest)
    ])
    return random_forest_pipeline

def gradient_boosting_pipeline():
    gradient_boosting = GradientBoostingRegressor(    n_estimators=500,      
    max_depth=5,             
    learning_rate=0.05,    
    subsample=0.8,           
    min_samples_leaf=5,
    random_state=42)
    gradient_boosting_pipeline = Pipeline([
        ('regression', gradient_boosting)
    ])
    return gradient_boosting_pipeline
