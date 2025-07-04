import json
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model and return metrics"""
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        training_time = time.time() - start_time
        
        # Store results
        result = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': float(training_time)
        }
        
        self.results[model_name] = result
        return result
    
    def get_all_results(self):
        """Get all model results"""
        return self.results
    
    def save_results(self, file_path):
        """Save results to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
    
    def load_results(self, file_path):
        """Load results from JSON file"""
        try:
            with open(file_path, 'r') as f:
                self.results = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def compare_models(self):
        """Compare all models and return sorted by accuracy"""
        if not self.results:
            return []
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        return sorted_results
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        return best_model