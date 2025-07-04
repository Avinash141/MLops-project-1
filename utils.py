"""
Utility functions for Housing Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any


def load_data() -> pd.DataFrame:
    """
    Load Boston Housing dataset from the specified URL.
    
    Returns:
        pd.DataFrame: DataFrame containing the Boston Housing data
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    
    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # Target variable
    
    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the data by splitting into train/test sets and scaling features.
    
    Args:
        df: DataFrame containing the data
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) arrays
    """
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of regression models to evaluate.
    
    Returns:
        Dict containing model names and instances
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    return models


def evaluate_model(model: Any, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate a model and return performance metrics.
    
    Args:
        model: Trained model instance
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        Dict containing performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    return {
        'MSE': mse,
        'R2': r2,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std()
    }


def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        Dict containing results for each model
    """
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{name} - MSE: {results[name]['MSE']:.4f}, R²: {results[name]['R2']:.4f}")
    
    return results


def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for each model.
    
    Returns:
        Dict containing hyperparameter grids for each model
    """
    param_grids = {
        'Linear Regression': {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
            'positive': [True, False]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    return param_grids


def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for all models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dict containing best models and their parameters
    """
    models = get_models()
    param_grids = get_hyperparameter_grids()
    best_models = {}
    
    for name, model in models.items():
        print(f"Tuning hyperparameters for {name}...")
        
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_models[name] = {
            'model': grid_search.best_estimator_,
            'params': grid_search.best_params_,
            'score': grid_search.best_score_
        }
        
        print(f"{name} - Best R²: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    return best_models


def evaluate_tuned_models(best_models: Dict[str, Any], X_train: np.ndarray, 
                         X_test: np.ndarray, y_train: np.ndarray, 
                         y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate tuned models and return performance metrics.
    
    Args:
        best_models: Dict containing best models from hyperparameter tuning
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        Dict containing performance metrics for each tuned model
    """
    results = {}
    
    for name, model_info in best_models.items():
        model = model_info['model']
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"{name} (Tuned) - MSE: {results[name]['MSE']:.4f}, R²: {results[name]['R2']:.4f}")
    
    return results


def plot_results(results: Dict[str, Dict[str, float]], title: str = "Model Performance Comparison"):
    """
    Plot model performance comparison.
    
    Args:
        results: Dict containing model results
        title: Title for the plot
    """
    # Extract model names and metrics
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE plot
    ax1.bar(models, mse_values, color='skyblue', alpha=0.7)
    ax1.set_title('Mean Squared Error (MSE)')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² plot
    ax2.bar(models, r2_values, color='lightgreen', alpha=0.7)
    ax2.set_title('R² Score')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_results_to_csv(results: Dict[str, Dict[str, float]], filename: str):
    """
    Save results to CSV file.
    
    Args:
        results: Dict containing model results
        filename: Name of the CSV file to save
    """
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(filename, index=True)
    print(f"Results saved to {filename}")


def display_results_table(results: Dict[str, Dict[str, float]]):
    """
    Display results in a formatted table.
    
    Args:
        results: Dict containing model results
    """
    df = pd.DataFrame.from_dict(results, orient='index')
    print("\nModel Performance Results:")
    print("=" * 80)
    print(df.round(4))
    print("=" * 80) 