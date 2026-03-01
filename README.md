# Stock Market Prediction using Linear Regression

A professional machine learning project that predicts stock market trends (S&P 500) using historical data from Yahoo Finance and a Linear Regression model.

## Features

- **Automated Data Ingestion**: Fetches real-time historical data using `yfinance`.
- **Feature Engineering**: Implements lagged variables and Moving Averages (SMA) for improved prediction.
- **Modular Architecture**: Clean project structure suitable for professional showcases.
- **Visualizations**: Generates plots for Actual vs. Predicted prices and residual analysis.

## Project Structure

```text
Linear_Regression/
├── data/               # Raw and processed datasets
├── exports/            # Generated plots and saved models
├── src/                # Source code modules
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model.py
│   └── visualization.py
├── notebooks/          # Optional EDA notebooks
├── main.py             # Main execution script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed.

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Usage

Run the main script to fetch data, train the model, and generate plots:

```bash
python main.py
```

## Results

The model predicts the next day's closing price based on the previous 5 days of price history and moving averages.

- **Prediction Plot**: Saved in `exports/sp500_predictions.png`
- **Error Distribution**: Saved in `exports/residuals_dist.png`

## Showcase

This project demonstrates:

- End-to-end Machine Learning pipeline design.
- Data collection from APIs.
- Time-series feature engineering.
- Professional coding standards for GitHub and LinkedIn portfolios.
