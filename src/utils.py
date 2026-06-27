import os 
import sys 
import logging

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

def setup_logger(name='ml pipeline', log_file='training.log'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # to prevent duplicates if fucntion will be called multiple times 
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # console handlers 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger 

def regression_plots(y_true, y_pred, output_dir: str = "models/plots"):
    """
    Generates and saves visual evaluations of the model predictions.
    Essential for analyzing heteroscedasticity and residual patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Actual vs Predicted values
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=axes[0], color='teal')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_title("Actual vs. Predicted Values")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    
    # Plot 2: Residual Plot (Errors)
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=axes[1], color='crimson')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_title("Residual Plot (Errors)")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals (True - Pred)")
    
    # Save fig
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()