from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        )
        self.is_trained = False


#         self.model = RandomForestClassifier(
#     n_estimators=100,  # 100 trees banayein
#     random_state=42,   # result repeatable banane ke liye seed
#     max_depth=10,      # har tree ka maximum depth 10 (jitna gehra tree, utna complex)
#     n_jobs=-1          # jitne CPU cores hain, sab use karo (speed ke liye)
# )
# self.is_trained = False  # Abhi model train nahi hua
    
    def train(self, X_train, y_train):
        """Train the random forest model"""
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training Random Forest: {e}")
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
        """Get feature importance from the forest"""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        return importance_dict
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Random Forest',
            'type': 'Ensemble Classification',
            'description': 'Ensemble of decision trees with voting mechanism',
            'pros': ['Reduces overfitting', 'Handles missing values', 'Provides feature importance'],
            'cons': ['Less interpretable than single tree', 'Can be memory intensive', 'May overfit with very noisy data']
        }