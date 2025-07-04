"""
Test script for Housing Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from utils import (
    load_data, preprocess_data, get_models, evaluate_model,
    train_and_evaluate_models, get_hyperparameter_grids
)


def test_load_data():
    """Test data loading functionality."""
    df = load_data()
    
    # Check if data is loaded correctly
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] == 14  # 13 features + 1 target
    assert 'MEDV' in df.columns
    assert len(df.columns) == 14
    
    # Check for missing values
    assert df.isnull().sum().sum() == 0
    
    print("Data loading test passed")


def test_preprocess_data():
    """Test data preprocessing functionality."""
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Check shapes
    assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
    assert X_train.shape[1] == 13  # 13 features
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    
    # Check if data is scaled (mean should be close to 0)
    assert abs(X_train.mean()) < 1e-10
    assert abs(X_test.mean()) < 1
    
    print("Data preprocessing test passed")


def test_get_models():
    """Test model initialization."""
    models = get_models()
    
    assert len(models) == 3
    assert 'Linear Regression' in models
    assert 'Random Forest' in models
    assert 'Gradient Boosting' in models
    
    print("Model initialization test passed")


def test_evaluate_model():
    """Test model evaluation functionality."""
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Test with Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Check if results contain required metrics
    assert 'MSE' in results
    assert 'R2' in results
    assert 'CV_R2_Mean' in results
    assert 'CV_R2_Std' in results
    
    # Check if values are reasonable
    assert results['MSE'] > 0
    assert results['R2'] <= 1
    
    print("Model evaluation test passed")


def test_hyperparameter_grids():
    """Test hyperparameter grid configuration."""
    param_grids = get_hyperparameter_grids()
    
    assert len(param_grids) == 3
    assert 'Linear Regression' in param_grids
    assert 'Random Forest' in param_grids
    assert 'Gradient Boosting' in param_grids
    
    # Check if each model has at least 3 hyperparameters
    for model_name, params in param_grids.items():
        assert len(params) >= 3, f"{model_name} should have at least 3 hyperparameters"
    
    print("Hyperparameter grids test passed")


def test_train_and_evaluate_models():
    """Test complete training and evaluation workflow."""
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Check if all models are evaluated
    assert len(results) == 3
    
    # Check if all models have reasonable performance
    for model_name, metrics in results.items():
        assert metrics['R2'] > 0, f"{model_name} should have positive R² score"
        assert metrics['MSE'] > 0, f"{model_name} should have positive MSE"
    
    print("Complete training and evaluation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running tests for Housing Price Prediction...")
    print("=" * 50)
    
    test_load_data()
    test_preprocess_data()
    test_get_models()
    test_evaluate_model()
    test_hyperparameter_grids()
    test_train_and_evaluate_models()
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests() 