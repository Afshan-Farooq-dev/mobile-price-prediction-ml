# Mobile Price Classification Project

A comprehensive machine learning project comparing different algorithms for mobile price classification using the Kaggle Mobile Price Classification dataset.

## 📁 Project Structure

```
mobile_price_classification/
├── data/
│   └── train.csv                    # Dataset (download from Kaggle)
├── models/
│   ├── __init__.py
│   ├── linear_regression.py         # Linear Regression implementation
│   ├── logistic_regression.py       # Logistic Regression implementation
│   ├── decision_tree.py             # Decision Tree implementation
│   ├── random_forest.py             # Random Forest implementation
│   ├── svm.py                       # Support Vector Machine implementation
│   └── ann.py                       # Artificial Neural Network implementation
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data preprocessing utilities
│   └── model_evaluator.py          # Model evaluation utilities
├── results/
│   └── model_results.json          # Stored model results
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create the project directory
mkdir mobile_price_classification
cd mobile_price_classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to [Kaggle Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
2. Download the `train.csv` file
3. Place it in the `data/` folder

### 3. Create Project Files

Create all the files as shown in the project structure with the provided code.

### 4. Run the Application

```bash
streamlit run app.py
```

## 📊 Features

### Data Analysis
- **Data Overview**: Explore dataset characteristics, distributions, and correlations
- **Missing Value Analysis**: Identify and handle missing data
- **Feature Correlation**: Visualize relationships between features

### Machine Learning Models
- **Linear Regression**: Simple baseline model
- **Logistic Regression**: Probabilistic classification
- **Decision Tree**: Rule-based interpretable model
- **Random Forest**: Ensemble method for robust predictions
- **SVM**: Support Vector Machine for complex patterns
- **ANN**: Artificial Neural Network for deep learning

### Model Evaluation
- **Accuracy, Precision, Recall, F1-Score**: Comprehensive metrics
- **Training Time**: Performance comparison
- **Feature Importance**: Understanding model decisions
- **Visual Comparisons**: Interactive charts and graphs

### Algorithm Analysis
- **Detailed Comparisons**: When to use each algorithm
- **Dataset Recommendations**: Best algorithms for different data types
- **Performance vs Complexity**: Trade-off analysis

## 📈 Model Performance

The application provides comprehensive comparison of all models including:

- Performance metrics comparison
- Training time analysis
- Feature importance visualization
- Best model recommendations

## 🔍 Algorithm Guide

### When to Use Each Algorithm

| Algorithm | Best For | Dataset Size | Interpretability | Speed |
|-----------|----------|--------------|------------------|-------|
| Linear Regression | Baseline, Linear relationships | Any | High | Very Fast |
| Logistic Regression | Binary/Multi-class classification | Small-Large | High | Fast |
| Decision Tree | Non-linear patterns, Rules | Small-Medium | Very High | Fast |
| Random Forest | Robust predictions | Medium-Large | Medium | Medium |
| SVM | High-dimensional data | Small-Medium | Low | Slow |
| ANN | Complex patterns | Large | Very Low | Slow |

### Dataset Type Recommendations

- **Small Dataset (< 1000)**: Logistic Regression, Decision Tree, SVM
- **Medium Dataset (1000-10000)**: Random Forest, SVM, Logistic Regression
- **Large Dataset (> 10000)**: Random Forest, ANN, Logistic Regression
- **High Dimensional**: SVM, Random Forest, ANN
- **Need Interpretability**: Decision Tree, Logistic Regression
- **Non-linear Patterns**: Random Forest, ANN, SVM

## 📝 Usage Instructions

### 1. Data Overview Page
- View dataset statistics and distributions
- Explore feature correlations
- Understand data quality

### 2. Model Training Page
- Click "Train All Models" to run all algorithms
- Monitor training progress
- View individual model results

### 3. Results Comparison Page
- Compare all models side by side
- View performance metrics
- Analyze feature importance
- Identify best performing model

### 4. Algorithm Analysis Page
- Learn about each algorithm
- Get recommendations for your use case
- Understand performance vs complexity trade-offs

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow**: Neural network implementation
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

### Model Implementations
Each model is implemented as a separate class with:
- `train()`: Training method
- `predict()`: Prediction method
- `get_feature_importance()`: Feature analysis (where applicable)
- `get_model_info()`: Algorithm information

### Data Preprocessing
- **Missing Value Handling**: Mean imputation
- **Feature Scaling**: StandardScaler normalization
- **Train-Test Split**: 80-20 split with stratification

## 📊 Expected Results

Based on the mobile price classification dataset, typical performance ranges:

- **Random Forest**: 85-90% accuracy
- **ANN**: 85-88% accuracy
- **SVM**: 80-85% accuracy
- **Logistic Regression**: 75-80% accuracy
- **Decision Tree**: 70-75% accuracy
- **Linear Regression**: 65-70% accuracy (adapted for classification)

## 🔧 Customization

### Adding New Models
1. Create a new file in `models/` directory
2. Implement the same interface as existing models
3. Add to the models dictionary in `app.py`

### Modifying Evaluation Metrics
- Edit `model_evaluator.py` to add new metrics
- Update visualization code in `app.py`

### Dataset Changes
- Modify `data_preprocessing.py` for different datasets
- Update feature names and target variables

## 📖 Learning Objectives

This project helps you understand:

1. **Data Preprocessing**: Cleaning and preparing data for ML
2. **Model Selection**: Choosing appropriate algorithms
3. **Performance Evaluation**: Comprehensive model assessment
4. **Algorithm Comparison**: Understanding trade-offs
5. **Practical Implementation**: Real-world ML project structure

## 🤝 Contributing

Feel free to:
- Add new algorithms
- Improve visualizations
- Enhance documentation
- Fix bugs or issues

## 📄 License

This project is for educational purposes. Dataset credit goes to the original Kaggle contributors.

## 🙋‍♂️ Support

If you encounter issues:
1. Check that all files are in correct directories
2. Verify dataset is downloaded and placed correctly
3. Ensure all dependencies are installed
4. Check Python version compatibility (3.7+)

---

**Happy Learning! 🎓**