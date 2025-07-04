"""
Main script for Housing Price Prediction - MLOps Assignment 1
This script implements regression models with and without hyperparameter tuning
"""

import os
import argparse
from utils import (
    load_data, preprocess_data, train_and_evaluate_models,
    hyperparameter_tuning, evaluate_tuned_models, plot_results,
    save_results_to_csv, display_results_table
)


def main():
    """Main function to run the housing price prediction workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Housing Price Prediction with Regression Models')
    parser.add_argument('--mode', choices=['basic', 'hyperparameter', 'both'], 
                       default='both', help='Mode to run: basic, hyperparameter, or both')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to CSV files')
    parser.add_argument('--plot', action='store_true', 
                       help='Display performance plots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HOUSING PRICE PREDICTION - MLOPS ASSIGNMENT 1")
    print("=" * 80)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {', '.join(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    # Display basic statistics
    print("\nDataset Info:")
    print(df.describe())
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Basic regression models
    if args.mode in ['basic', 'both']:
        print("\n" + "=" * 80)
        print("2. BASIC REGRESSION MODELS")
        print("=" * 80)
        
        results_basic = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        display_results_table(results_basic)
        
        if args.save_results:
            save_results_to_csv(results_basic, 'basic_regression_results.csv')
        
        if args.plot:
            plot_results(results_basic, "Basic Regression Models - Performance Comparison")
    
    # Hyperparameter tuning
    if args.mode in ['hyperparameter', 'both']:
        print("\n" + "=" * 80)
        print("3. HYPERPARAMETER TUNING")
        print("=" * 80)
        
        # Perform hyperparameter tuning
        best_models = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate tuned models
        print("\nEvaluating tuned models...")
        results_tuned = evaluate_tuned_models(best_models, X_train, X_test, y_train, y_test)
        display_results_table(results_tuned)
        
        if args.save_results:
            save_results_to_csv(results_tuned, 'hyperparameter_tuned_results.csv')
        
        if args.plot:
            plot_results(results_tuned, "Hyperparameter Tuned Models - Performance Comparison")
        
        # Display best parameters
        print("\nBest Hyperparameters:")
        print("-" * 40)
        for model_name, model_info in best_models.items():
            print(f"{model_name}:")
            for param, value in model_info['params'].items():
                print(f"  {param}: {value}")
            print(f"  CV Score: {model_info['score']:.4f}")
            print()
    
    # Comparison between basic and tuned models
    if args.mode == 'both':
        print("\n" + "=" * 80)
        print("4. COMPARISON: BASIC vs HYPERPARAMETER TUNED")
        print("=" * 80)
        
        # Compare results
        print("\nPerformance Improvement Summary:")
        print("-" * 50)
        
        for model_name in results_basic.keys():
            basic_r2 = results_basic[model_name]['R2']
            tuned_r2 = results_tuned[model_name]['R2']
            improvement = ((tuned_r2 - basic_r2) / basic_r2) * 100
            
            print(f"{model_name}:")
            print(f"  Basic R²: {basic_r2:.4f}")
            print(f"  Tuned R²: {tuned_r2:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            print()
    
    # Find best overall model
    if args.mode in ['hyperparameter', 'both']:
        best_model_name = max(results_tuned.keys(), key=lambda x: results_tuned[x]['R2'])
        best_r2 = results_tuned[best_model_name]['R2']
        best_mse = results_tuned[best_model_name]['MSE']
        
        print("\n" + "=" * 80)
        print("5. BEST MODEL SUMMARY")
        print("=" * 80)
        print(f"Best Model: {best_model_name}")
        print(f"R² Score: {best_r2:.4f}")
        print(f"MSE: {best_mse:.4f}")
        print(f"Best Parameters: {best_models[best_model_name]['params']}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main() 