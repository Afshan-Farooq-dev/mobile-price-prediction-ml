import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utils classes from utils module
from utils import DataPreprocessor, ModelEvaluator

# Import all models from models module
from models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    DecisionTreeModel,
    RandomForestModel,
    SVMModel,
    ANNModel
)

# Page configuration
st.set_page_config(
    page_title="Mobile Price Classification",
    page_icon="📱",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title and description
st.title("📱 Mobile Price Classification Project")
st.markdown("Compare different machine learning algorithms for mobile price prediction")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Overview", "Model Training", "Results Comparison", "Algorithm Analysis"]
)

# Data loading section
@st.cache_data
def load_and_preprocess_data():
    preprocessor = DataPreprocessor()
    
    # Try to load data
    data_path = "data/train.csv"
    if os.path.exists(data_path):
        data = preprocessor.load_data(data_path)
        if data is not None:
            return data, preprocessor
    
    return None, None

# Load data
data, preprocessor = load_and_preprocess_data()

if data is None:
    st.error("⚠️ Please place the 'train.csv' file in the 'data/' folder")
    st.info("Download the dataset from: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification")
    st.stop()

st.session_state.data_loaded = True

# Page 1: Data Overview
if page == "Data Overview":
    st.header("📊 Data Overview")
    
    # Basic information
    info = preprocessor.basic_info(data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", info['shape'][0])
    with col2:
        st.metric("Features", info['shape'][1] - 1)  # Excluding target
    with col3:
        st.metric("Classes", len(info['target_distribution']) if info['target_distribution'] else 0)
    with col4:
        st.metric("Missing Values", sum(info['missing_values'].values()))
    
    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(data.head())
    
    # # Target distribution
    # if info['target_distribution']:
    #     st.subheader("Price Range Distribution")
    #     fig = px.bar(
    #         x=list(info['target_distribution'].keys()),
    #         y=list(info['target_distribution'].values()),
    #         title="Distribution of Price Ranges",
    #         labels={'x': 'Price Range', 'y': 'Count'}
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
    
    # # Correlation heatmap
    # st.subheader("Feature Correlation")
    # correlation_matrix = preprocessor.get_feature_importance_data(data)
    
    # fig, ax = plt.subplots(figsize=(12, 8))
    # sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
    # plt.title("Feature Correlation Matrix")
    # st.pyplot(fig)

# Page 2: Model Training
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.button("Train All Models", type="primary"):
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(data)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegressionModel(),
            'Logistic Regression': LogisticRegressionModel(),
            'Decision Tree': DecisionTreeModel(),
            'Random Forest': RandomForestModel(),
            'SVM': SVMModel(),
            'ANN': ANNModel()
        }
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train each model
        trained_models = {}
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                # Train model
                success = model.train(X_train, y_train)
                
                if success:
                    # Evaluate model
                    result = evaluator.evaluate_model(model, X_test, y_test, name)
                    trained_models[name] = {
                        'model': model,
                        'result': result
                    }
                    st.success(f"✅ {name} trained successfully - Accuracy: {result['accuracy']:.3f}")
                else:
                    st.error(f"❌ Failed to train {name}")
                    
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(models))
        
        # Store results in session state
        st.session_state.trained_models = trained_models
        st.session_state.evaluator = evaluator
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.feature_names = list(X_train.columns)
        st.session_state.models_trained = True
        
        status_text.text("Training completed!")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        evaluator.save_results('results/model_results.json')
        
        st.success("🎉 All models trained successfully!")

# Page 3: Results Comparison
elif page == "Results Comparison":
    st.header("📈 Results Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the 'Model Training' page")
    else:
        evaluator = st.session_state.evaluator
        results = evaluator.get_all_results()
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        # Display results table
        st.subheader("Model Performance Comparison")
        st.dataframe(comparison_df.round(4))
        
        # Accuracy comparison chart
        fig = px.bar(
            comparison_df,
            x=comparison_df.index,
            y='accuracy',
            title="Model Accuracy Comparison",
            color='accuracy',
            color_continuous_scale='viridis',
            labels={'x': 'Models', 'y': 'Accuracy'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # # Multi-metric comparison
        # st.subheader("Multi-Metric Comparison")
        # metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # fig = go.Figure()
        # for metric in metrics:
        #     fig.add_trace(go.Bar(
        #         name=metric.replace('_', ' ').title(),
        #         x=comparison_df.index,
        #         y=comparison_df[metric],
        #     ))
        
        # fig.update_layout(
        #     title="Comparison of All Metrics",
        #     xaxis_title="Models",
        #     yaxis_title="Score",
        #     barmode='group'
        # )
        # st.plotly_chart(fig, use_container_width=True)
        
        # Training time comparison
        fig = px.bar(
            comparison_df,
            x=comparison_df.index,
            y='training_time',
            title="Training Time Comparison",
            color='training_time',
            color_continuous_scale='reds',
            labels={'x': 'Models', 'y': 'Training Time (seconds)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (for applicable models)
        st.subheader("Feature Importance Analysis")
        
        trained_models = st.session_state.trained_models
        feature_names = st.session_state.feature_names
        
        models_with_importance = []
        for name, model_data in trained_models.items():
            model = model_data['model']
            importance = model.get_feature_importance(feature_names)
            if importance is not None:
                models_with_importance.append((name, importance))
        
        if models_with_importance:
            # Create subplot for feature importance
            n_models = len(models_with_importance)
            
            for i, (model_name, importance) in enumerate(models_with_importance):
                if i % 2 == 0:
                    col1, col2 = st.columns(2)
                
                # Sort features by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:10]  # Top 10 features
                
                fig = px.bar(
                    x=[f[1] for f in top_features],
                    y=[f[0] for f in top_features],
                    orientation='h',
                    title=f"{model_name} - Top 10 Features",
                    labels={'x': 'Importance', 'y': 'Features'}
                )
                
                if i % 2 == 0:
                    col1.plotly_chart(fig, use_container_width=True)
                else:
                    col2.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        best_model = evaluator.get_best_model()
        if best_model:
            st.success(f"🏆 Best Performing Model: **{best_model[0]}** with {best_model[1]['accuracy']:.3f} accuracy")

# Page 4: Algorithm Analysis
elif page == "Algorithm Analysis":
    st.header("🔍 Algorithm Analysis & Recommendations")
    
    # Algorithm information
    algorithms_info = {
        'Linear Regression': {
            'best_for': ['Regression problems', 'Linear relationships', 'Baseline models'],
            'dataset_size': 'Any size',
            'interpretability': 'High',
            'training_speed': 'Very Fast',
            'when_to_use': 'When you need a simple baseline or suspect linear relationships'
        },
        'Logistic Regression': {
            'best_for': ['Binary/Multi-class classification', 'Probability estimates', 'Linear decision boundaries'],
            'dataset_size': 'Small to Large',
            'interpretability': 'High',
            'training_speed': 'Fast',
            'when_to_use': 'Classification with interpretable results and probability estimates'
        },
        'Decision Tree': {
            'best_for': ['Non-linear patterns', 'Categorical features', 'Rule-based decisions'],
            'dataset_size': 'Small to Medium',
            'interpretability': 'Very High',
            'training_speed': 'Fast',
            'when_to_use': 'When you need interpretable rules and handle mixed data types'
        },
        'Random Forest': {
            'best_for': ['Robust predictions', 'Feature importance', 'Handling overfitting'],
            'dataset_size': 'Medium to Large',
            'interpretability': 'Medium',
            'training_speed': 'Medium',
            'when_to_use': 'When you want good performance with minimal tuning'
        },
        'SVM': {
            'best_for': ['High-dimensional data', 'Non-linear patterns', 'Small datasets'],
            'dataset_size': 'Small to Medium',
            'interpretability': 'Low',
            'training_speed': 'Slow',
            'when_to_use': 'High-dimensional data or when you need robust classification'
        },
        'ANN': {
            'best_for': ['Complex patterns', 'Large datasets', 'Non-linear relationships'],
            'dataset_size': 'Large',
            'interpretability': 'Very Low',
            'training_speed': 'Slow',
            'when_to_use': 'Complex patterns with large datasets and computational resources'
        }
    }
    
    # Display algorithm comparison table
    st.subheader("Algorithm Characteristics")
    
    comparison_data = []
    for algo, info in algorithms_info.items():
        comparison_data.append({
            'Algorithm': algo,
            'Dataset Size': info['dataset_size'],
            'Interpretability': info['interpretability'],
            'Training Speed': info['training_speed'],
            'Best For': ', '.join(info['best_for'][:2])  # Show first 2 items
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Detailed analysis for each algorithm
    st.subheader("Detailed Algorithm Analysis")
    
    selected_algo = st.selectbox("Select algorithm for detailed analysis:", list(algorithms_info.keys()))
    
    if selected_algo:
        info = algorithms_info[selected_algo]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best suited for:**")
            for item in info['best_for']:
                st.write(f"• {item}")
        
        with col2:
            st.write("**Characteristics:**")
            st.write(f"• **Dataset Size:** {info['dataset_size']}")
            st.write(f"• **Interpretability:** {info['interpretability']}")
            st.write(f"• **Training Speed:** {info['training_speed']}")
        
        st.write(f"**When to use:** {info['when_to_use']}")
        
        # Get model-specific information if available
        if st.session_state.models_trained:
            trained_models = st.session_state.trained_models
            if selected_algo in trained_models:
                model_info = trained_models[selected_algo]['model'].get_model_info()
                
                st.subheader(f"{selected_algo} - Detailed Information")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pros:**")
                    for pro in model_info['pros']:
                        st.write(f"✅ {pro}")
                
                with col2:
                    st.write("**Cons:**")
                    for con in model_info['cons']:
                        st.write(f"❌ {con}")
    
    # Dataset type recommendations
    st.subheader("📋 Dataset Type Recommendations")
    
    recommendations = {
        'Small Dataset (< 1000 samples)': ['Logistic Regression', 'Decision Tree', 'SVM'],
        'Medium Dataset (1000-10000 samples)': ['Random Forest', 'SVM', 'Logistic Regression'],
        'Large Dataset (> 10000 samples)': ['Random Forest', 'ANN', 'Logistic Regression'],
        'High Dimensional Data': ['SVM', 'Random Forest', 'ANN'],
        'Need Interpretability': ['Decision Tree', 'Logistic Regression', 'Linear Regression'],
        'Non-linear Patterns': ['Random Forest', 'ANN', 'SVM'],
        'Fast Prediction Required': ['Logistic Regression', 'Decision Tree', 'Linear Regression'],
        'Probability Estimates Needed': ['Logistic Regression', 'Random Forest', 'ANN']
    }
    
    for scenario, algos in recommendations.items():
        with st.expander(f"🎯 {scenario}"):
            st.write("**Recommended algorithms:**")
            for i, algo in enumerate(algos, 1):
                st.write(f"{i}. {algo}")
    
    
# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📱 Mobile Price Classification**")


# Instructions
if not st.session_state.models_trained:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Quick Start:**")
    st.sidebar.markdown("1. Check Data Overview")
    st.sidebar.markdown("2. Train Models")
    st.sidebar.markdown("3. Compare Results")
    st.sidebar.markdown("4. Analyze Algorithms")