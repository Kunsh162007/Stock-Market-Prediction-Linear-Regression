from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """
        Trains the Linear Regression model.
        """
        print("Training model...")
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """
        Makes predictions using the trained model.
        """
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluates the model performance.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }
        return metrics

    def save_model(self, path):
        """
        Saves the trained model to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads a trained model from a file.
        """
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
