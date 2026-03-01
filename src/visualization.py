import matplotlib.pyplot as plt
import os

def plot_predictions(dates, y_true, y_pred, symbol, save_path=None):
    """
    Plots Actual vs Predicted prices.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_true, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted Price', color='orange', linestyle='--', alpha=0.8)
    
    plt.title(f'Stock Price Prediction: {symbol}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plots the residuals (errors).
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, color='gray', edgecolor='black', alpha=0.7)
    plt.title('Residuals Distribution', fontsize=14)
    plt.xlabel('Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Residuals plot saved to {save_path}")
    plt.show()
