# models/__init__.py
"""
Machine Learning Models Module

This module contains implementations of various machine learning algorithms
for the mobile price classification project.
"""

__version__ = "1.0.0"
__author__ = "Mobile Price Classification Project"

# Import all model classes
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .svm import SVMModel
from .ann import ANNModel

__all__ = [
    'LinearRegressionModel',
    'LogisticRegressionModel', 
    'DecisionTreeModel',
    'RandomForestModel',
    'SVMModel',
    'ANNModel'
]

# utils/__init__.py
"""
Utilities Module

This module contains utility classes for data preprocessing and model evaluation.
"""

__version__ = "1.0.0"
__author__ = "Mobile Price Classification Project"

# # Import utility classes
# from .data_preprocessing import DataPreprocessor
# from .model_evaluator import ModelEvaluator

__all__ = [
    'DataPreprocessor',
    'ModelEvaluator'
]