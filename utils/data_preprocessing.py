import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')

    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def basic_info(self, data):
        """Get basic information about the dataset"""
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'target_distribution': data['price_range'].value_counts().to_dict() if 'price_range' in data.columns else None
        }
        return info
    
    def preprocess_data(self, data, target_column='price_range'):
        """Preprocess the data for machine learning"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_imputed),
            columns=X.columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_importance_data(self, data):
        """Get data for feature importance analysis"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        return correlation_matrix