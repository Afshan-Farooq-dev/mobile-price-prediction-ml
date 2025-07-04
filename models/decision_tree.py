from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the decision tree model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Decision Tree: {e}")
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
        """Get feature importance from the tree"""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        return importance_dict
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Decision Tree',
            'type': 'Classification',
            'description': 'Tree-based model that makes decisions using if-else conditions',
            'pros': ['Easy to interpret', 'Handles non-linear patterns', 'No scaling required'],
            'cons': ['Prone to overfitting', 'Can be unstable', 'Biased toward features with more levels']
        }
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Data load karo (Iris dataset, famous example)
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Data ko training aur testing mein baanto (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model banao aur train karo
    model = DecisionTreeModel()
    model.train(X_train, y_train)

    # Predictions banao test data pe
    preds = model.predict(X_test)
    print("Predictions:", preds)

    # Har class ke liye probabilities nikalo
    probs = model.predict_proba(X_test)
    print("Probabilities:\n", probs)

    # Feature importance check karo
    importance = model.get_feature_importance(feature_names)
    print("Feature Importance:\n", importance)

