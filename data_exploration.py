"""
Data Exploration Script for Boston Housing Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

def explore_dataset():
    """Explore the Boston Housing dataset."""
    print("=" * 60)
    print("BOSTON HOUSING DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of samples: {df.shape[0]}")
    
    # Feature descriptions
    feature_descriptions = {
        'CRIM': 'Per capita crime rate by town',
        'ZN': 'Proportion of residential land zoned for lots over 25,000 sq.ft.',
        'INDUS': 'Proportion of non-retail business acres per town',
        'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
        'NOX': 'Nitric oxides concentration (parts per 10 million)',
        'RM': 'Average number of rooms per dwelling',
        'AGE': 'Proportion of owner-occupied units built prior to 1940',
        'DIS': 'Weighted distances to five Boston employment centres',
        'RAD': 'Index of accessibility to radial highways',
        'TAX': 'Full-value property-tax rate per $10,000',
        'PTRATIO': 'Pupil-teacher ratio by town',
        'B': '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
        'LSTAT': '% lower status of the population',
        'MEDV': 'Median value of owner-occupied homes in $1000s (Target)'
    }
    
    print("\n" + "=" * 60)
    print("FEATURE DESCRIPTIONS")
    print("=" * 60)
    for feature, description in feature_descriptions.items():
        print(f"{feature:10}: {description}")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("BASIC STATISTICS")
    print("=" * 60)
    print(df.describe())
    
    # Missing values
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✓ No missing values found!")
    
    # Correlation analysis
    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGET VARIABLE (MEDV)")
    print("=" * 60)
    correlations = df.corr()['MEDV'].sort_values(ascending=False)
    print(correlations)
    
    # Data types
    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    
    return df

def create_visualizations(df):
    """Create visualizations for the dataset."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Distribution of target variable
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['MEDV'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of House Prices (MEDV)')
    plt.xlabel('Median House Value ($1000s)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    
    # 3. Scatter plot: RM vs MEDV
    plt.subplot(2, 2, 3)
    plt.scatter(df['RM'], df['MEDV'], alpha=0.6, color='green')
    plt.xlabel('Average Number of Rooms (RM)')
    plt.ylabel('Median House Value (MEDV)')
    plt.title('House Price vs Number of Rooms')
    plt.grid(True, alpha=0.3)
    
    # 4. Scatter plot: LSTAT vs MEDV
    plt.subplot(2, 2, 4)
    plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.6, color='red')
    plt.xlabel('% Lower Status Population (LSTAT)')
    plt.ylabel('Median House Value (MEDV)')
    plt.title('House Price vs Lower Status Population')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance visualization
    plt.figure(figsize=(12, 6))
    correlations = df.corr()['MEDV'].drop('MEDV').sort_values(ascending=True)
    
    colors = ['red' if x < 0 else 'green' for x in correlations]
    plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
    plt.yticks(range(len(correlations)), correlations.index)
    plt.xlabel('Correlation with House Price (MEDV)')
    plt.title('Feature Correlation with House Prices')
    plt.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, v in enumerate(correlations.values):
        plt.text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                va='center', ha='left' if v > 0 else 'right')
    
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plots for key features
    plt.figure(figsize=(15, 10))
    
    key_features = ['CRIM', 'RM', 'LSTAT', 'PTRATIO', 'NOX', 'DIS']
    
    for i, feature in enumerate(key_features, 1):
        plt.subplot(2, 3, i)
        plt.boxplot(df[feature])
        plt.title(f'Box Plot: {feature}')
        plt.ylabel(feature)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizations created and saved!")

def main():
    """Main function to run data exploration."""
    # Explore dataset
    df = explore_dataset()
    
    # Create visualizations
    create_visualizations(df)
    
    # Summary insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("1. Strong positive correlation between RM (rooms) and house prices")
    print("2. Strong negative correlation between LSTAT (lower status %) and house prices")
    print("3. Crime rate (CRIM) has negative correlation with house prices")
    print("4. Pupil-teacher ratio (PTRATIO) negatively affects house prices")
    print("5. The target variable (MEDV) follows roughly normal distribution")
    print("6. No missing values in the dataset")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR MODELING")
    print("=" * 60)
    print("1. Feature scaling is important due to different ranges")
    print("2. RM and LSTAT are likely to be the most important features")
    print("3. Consider polynomial features for RM")
    print("4. Monitor for outliers in CRIM and other features")
    print("5. Cross-validation is crucial for reliable model evaluation")

if __name__ == "__main__":
    main() 