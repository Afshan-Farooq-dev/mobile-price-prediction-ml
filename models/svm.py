from sklearn.svm import SVC
import numpy as np

class SVMModel:
    def __init__(self):
        self.model = SVC(
            kernel='rbf',
            random_state=42,
            probability=True,  # Enable probability estimates
            C=1.0   #C=1.0: Ye ek penalty parameter hai. Zyada C matlab model training mein galtiyon ko kam bardasht karega, lekin overfitting ho sakta hai. Kam C matlab model zyada simple hoga.
        )
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the SVM model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training SVM: {e}")
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
        """SVM doesn't provide direct feature importance, return None"""
        return None
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Support Vector Machine (SVM)',
            'type': 'Classification',
            'description': 'Finds optimal hyperplane to separate classes with maximum margin',
            'pros': ['Effective in high dimensions', 'Memory efficient', 'Versatile with different kernels'],
            'cons': ['Slow on large datasets', 'Sensitive to feature scaling', 'No direct feature importance']
        }