from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the linear regression model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Linear Regression: {e}")
            return False
    
    def predict(self, X_test):
        """Make predictions (rounded to nearest integer for classification)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X_test)
        # Round predictions to nearest integer for classification
        return np.round(predictions).astype(int)
    
    def get_feature_importance(self, feature_names):
        """Get feature coefficients as importance"""
        if not self.is_trained:
            return None
        
        coefficients = self.model.coef_
        importance_dict = dict(zip(feature_names, np.abs(coefficients)))
        return importance_dict
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Linear Regression',
            'type': 'Regression (adapted for classification)',
            'description': 'Linear model that finds the best linear relationship between features and target',
            'pros': ['Simple and interpretable', 'Fast training', 'No hyperparameters'],
            'cons': ['Assumes linear relationship', 'Sensitive to outliers', 'Not ideal for classification']
        }

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    # Data load karo
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Data scale kar lo (achha result ke liye)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model banao aur train karo
    model = LinearRegressionModel()
    model.train(X_train, y_train)

    # Prediction karo
    preds = model.predict(X_test)
    print("Predictions:", preds)

    # MSE calculate karo
    mse = mean_squared_error(y_test, preds)
    print("Mean Squared Error:", mse)

    # Feature importance print karo
    importance = model.get_feature_importance(feature_names)
    print("Feature Importance:", importance)

    # Model info bhi print karo
    info = model.get_model_info()
    print("Model Info:", info)
