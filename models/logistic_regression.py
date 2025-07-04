from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
            return False
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, feature_names):
        """Get feature coefficients as importance"""
        if not self.is_trained:
            return None
        
        # For multiclass, take the mean of absolute coefficients across classes
        if len(self.model.coef_) > 1:
            coefficients = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            coefficients = np.abs(self.model.coef_[0])
        
        importance_dict = dict(zip(feature_names, coefficients))
        return importance_dict
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Logistic Regression',
            'type': 'Classification',
            'description': 'Linear classifier using logistic function for probability estimation',
            'pros': ['Good for binary/multiclass classification', 'Provides probabilities', 'Interpretable'],
            'cons': ['Assumes linear decision boundary', 'May struggle with complex patterns']
        }

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegressionModel()
    model.train(X_train, y_train)

    preds = model.predict(X_test)
    print("Predictions:", preds)

    probs = model.predict_proba(X_test)
    print("Probabilities:\n", probs)

    importance = model.get_feature_importance(feature_names)
    print("Feature Importance:\n", importance)

    info = model.get_model_info()
    print("Model Info:\n", info)
