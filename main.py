import os
from src.data_ingestion import fetch_stock_data
from src.preprocessing import preprocess_data
from src.model import StockPredictor
from src.visualization import plot_predictions, plot_residuals
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Configuration
    SYMBOL = "^GSPC" # S&P 500
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"
    DATA_PATH = "data/sp500_data.csv"
    MODEL_PATH = "exports/stock_model.joblib"
    PLOT_PATH = "exports/sp500_predictions.png"
    RESIDUALS_PATH = "exports/residuals_dist.png"

    # 1. Fetch Data
    if not os.path.exists(DATA_PATH):
        data = fetch_stock_data(SYMBOL, START_DATE, END_DATE, DATA_PATH)
    else:
        print(f"Loading data from {DATA_PATH}...")
        data = pd.read_csv(DATA_PATH)

    if data is None or data.empty:
        print("Failed to get data.")
        return

    # 2. Preprocess Data
    print("Preprocessing data...")
    X, y, dates, scaler = preprocess_data(data)
    
    # Split into Train and Test
    # For time-series, we usually split by date, not randomly.
    # We'll take the last 20% for testing.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = dates.iloc[split_idx:]

    # 3. Modeling
    predictor = StockPredictor()
    predictor.train(X_train, y_train)
    
    # Evaluate
    train_preds = predictor.predict(X_train)
    test_preds = predictor.predict(X_test)
    
    metrics = predictor.evaluate(y_test, test_preds)
    print("\nModel Evaluation Metrics (Test Set):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 4. Save Model
    predictor.save_model(MODEL_PATH)

    # 5. Visualizations
    print("\nGenerating plots...")
    plot_predictions(test_dates, y_test, test_preds, SYMBOL, PLOT_PATH)
    plot_residuals(y_test, test_preds, RESIDUALS_PATH)
    
    print("\nProject execution complete. Check 'exports/' for results.")

if __name__ == "__main__":
    main()
