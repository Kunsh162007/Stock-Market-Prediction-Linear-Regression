import matplotlib.pyplot as plt
import os

def plot_predictions(dates, y_true, y_pred, symbol, save_path=None, show=False):
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
    
    if show:
        plt.show()
    plt.close()

def plot_residuals(y_true, y_pred, save_path=None, show=False):
    """
    Plots the residuals (errors) with detailed statistics.
    """
    residuals = y_true - y_pred
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add vertical line for mean error
    plt.axvline(mean_err, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_err:.2f}')
    
    plt.title('Residuals Distribution (Actual - Predicted)', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add info box
    info_text = f"Stats:\nMean: {mean_err:.2f}\nStd Dev: {std_err:.2f}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Detailed residuals plot saved to {save_path}")
    
    if show:
        plt.show()
    plt.close()
