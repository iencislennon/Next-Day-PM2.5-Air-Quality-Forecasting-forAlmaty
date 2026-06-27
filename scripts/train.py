from sklearn.linear_model import _ridge
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from src.data_loader import PM25DataLoder
from src.model import ridge_train_pipeline, random_forest_pipeline, gradient_boosting_pipeline
from src.utils import setup_logger, regression_plots as save_regression_plots

logger = setup_logger(name="TrainPipeline")

loader = PM25DataLoder()
X_train, X_test, y_train, y_test = loader.train_test_split()

def train_ridge_baseline_model():
    ridge = ridge_train_pipeline()
    logger.info("Fitting Scikit-Learn Model...")
    ridge.fit(X_train, y_train)
    output_dir = Path('/Users/ansartleubayev/Documents/20 projects /models')
    model_filename = output_dir / 'ridge_baseline.pkl'

    with open(model_filename, 'wb') as file:
        pickle.dump(ridge, file)
    logger.info("Model pipeline serialized to disk successfully.")


    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    save_regression_plots(y_test, y_pred)
    logger.info("Evaluation plots saved to models/plots/")
    logger.info(f"Ridge Baseline test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


def train_random_forest_pipeline():
    rf = random_forest_pipeline()
    logger.info("Fitting Scikit-Learn Model...")
    rf.fit(X_train, y_train)

    output_dir = Path('/Users/ansartleubayev/Documents/20 projects /models')
    model_filename = output_dir / 'random_forest.pkl'

    with open(model_filename, 'wb') as file:
        pickle.dump(rf, file)
    logger.info("Model pipeline serialized to disk successfully.")


    
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    save_regression_plots(y_test, y_pred)
    logger.info("Evaluation plots saved to models/plots/")
    logger.info(f"Random Forest test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

def train_gradient_boosting_pipeline():
    gb = gradient_boosting_pipeline()
    logger.info("Fitting Scikit-Learn Model...")
    gb.fit(X_train, y_train)

    output_dir = Path('/Users/ansartleubayev/Documents/20 projects /models')
    model_filename = output_dir / 'gradient_boosting.pkl'

    with open(model_filename, 'wb') as file:
        pickle.dump(gb, file)
    logger.info("Model pipeline serialized to disk successfully.")

    y_pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    save_regression_plots(y_test, y_pred)
    logger.info("Evaluation plots saved to models/plots/")
    logger.info(f"Gradient Boosting test results: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


if __name__ == '__main__':
    train_ridge_baseline_model()
    train_random_forest_pipeline()
    train_gradient_boosting_pipeline()

    
    
