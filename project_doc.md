MLOps Assignment 1 - Housing Price Prediction Project Documentation

Project Overview

Project Name: Housing Price Prediction using Machine Learning  
Assignment: MLOps Assignment 1  
Student: Avinash Singh  
GitHub Repository**: https://github.com/Avi2001-dot/MLops-project-1  
Roll Number: G24Ai1027  

Project Objective
Design, implement, and automate a complete machine learning workflow to predict house prices using classical machine learning models with proper MLOps practices including version control, CI/CD pipelines, and automated testing.

---

## Assignment Requirements Fulfilled

### Core Requirements
- Environment Setup: Conda environment with requirements.txt
- GitHub Repository: Command-line setup (no web upload)
- Three Regression Models: Linear Regression, Random Forest, Gradient Boosting
- Hyperparameter Tuning: Minimum 3 hyperparameters per model
- Performance Comparison: MSE and R² metrics
- Modular Code Structure: All functions in utils.py
- CI/CD Pipeline: GitHub Actions automation
- Branch Strategy: main, reg, hyper branches
- Complete Documentation: Comprehensive project documentation

### Technical Specifications
- Programming Language: Python 3.8+
- ML Framework: scikit-learn
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn
- Testing: Custom test suite
- CI/CD: GitHub Actions
- Version Control: Git with proper branching strategy

---

## Project Architecture

### Repository Structure
```
MLops-project-1/
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI/CD pipeline
├── utils.py                   # Utility functions for ML workflow
├── regression.py              # Main regression analysis script
├── data_exploration.py        # Data exploration and visualization
├── test_regression.py         # Test suite for validation
├── setup.py                   # Automated setup script
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
├── INSTRUCTIONS.md            # Setup and execution guide
└── project_doc.md             # This comprehensive documentation
```

### Branching Strategy
- main`: Production-ready code with merged features
- reg`: Basic regression models implementation
- hyper: Hyperparameter tuning implementation

---

## Dataset Information

### Boston Housing Dataset
- Source: http://lib.stat.cmu.edu/datasets/boston
- Samples: 506 housing records
- Features: 13 input features + 1 target variable
- Target: MEDV (Median home value in $1000s)

### Features Description
| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate by town |
| ZN | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS | Proportion of non-retail business acres per town |
| CHAS | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| NOX | Nitric oxides concentration (parts per 10 million) |
| RM | Average number of rooms per dwelling |
| AGE | Proportion of owner-occupied units built prior to 1940 |
| DIS | Weighted distances to five Boston employment centres |
| RAD | Index of accessibility to radial highways |
| TAX | Full-value property-tax rate per $10,000 |
| PTRATIO| Pupil-teacher ratio by town |
| B | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
| LSTAT | % lower status of the population |
| MEDV | Median value of owner-occupied homes in $1000s *(Target)* |

### Data Preprocessing
- Train-Test Split: 80-20 split with random state 42
- Feature Scaling: StandardScaler for all numerical features
- Missing Values: None (dataset is clean)
- Data Validation: Comprehensive checks for data integrity

---

## Machine Learning Models

### Model 1: Linear Regression
```python
LinearRegression()
```
Hyperparameters Tuned:
- `fit_intercept`: [True, False]
- `copy_X`: [True, False]  
- `positive`: [True, False]

Performance:
- R² Score: ~0.67
- MSE: ~24.29
- Cross-validation R²: ~0.72

### Model 2: Random Forest Regressor
```python
RandomForestRegressor(random_state=42)
```
Hyperparameters Tuned:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [3, 5, 7, None]
- `min_samples_split`: [2, 5, 10]

Performance:
- R² Score: ~0.89
- MSE: ~7.91
- Cross-validation R²: ~0.83

### Model 3: Gradient Boosting Regressor
```python
GradientBoostingRegressor(random_state=42)
```
Hyperparameters Tuned:
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 5, 7]

Performance:
- R² Score: ~0.92
- MSE: ~6.21
- Cross-validation R²: ~0.85

---

## Performance Analysis

### Model Comparison
| Model | MSE | R² Score | CV R² Mean | CV R² Std |
|-------|-----|----------|------------|-----------|
| Linear Regression | 24.29 | 0.6688 | 0.7244 | 0.0635 |
| Random Forest | 7.91 | 0.8921 | 0.8262 | 0.0475 |
| Gradient Boosting | 6.21 | 0.9153 | 0.8511 | 0.0546 |

### **Key Insights**
1. **Best Performing Model**: Gradient Boosting Regressor
2. **Significant Improvement**: Random Forest and Gradient Boosting significantly outperform Linear Regression
3. **Consistency**: All models show consistent performance across cross-validation folds
4. **Feature Importance**: RM (rooms) and LSTAT (lower status %) are most predictive

### **Hyperparameter Tuning Impact**
- **Linear Regression**: Minimal improvement (~1-2%)
- **Random Forest**: Moderate improvement (~5-8%)
- **Gradient Boosting**: Significant improvement (~8-12%)

---

## 🔧 **Technical Implementation**

### **Core Functions (utils.py)**
- `load_data()`: Loads Boston Housing dataset from CMU repository
- `preprocess_data()`: Handles train-test split and feature scaling
- `get_models()`: Returns dictionary of regression models
- `evaluate_model()`: Computes performance metrics
- `hyperparameter_tuning()`: Performs grid search optimization
- `plot_results()`: Creates performance visualization plots

### **Main Workflow (regression.py)**
- Command-line interface with multiple execution modes
- Supports basic regression, hyperparameter tuning, or both
- Automated results saving and visualization
- Comprehensive performance comparison

### **Data Exploration (data_exploration.py)**
- Dataset overview and statistical analysis
- Feature correlation analysis
- Visualization of key relationships
- Distribution analysis and outlier detection

### **Testing Suite (test_regression.py)**
- Unit tests for data loading and preprocessing
- Model initialization and evaluation tests
- Hyperparameter grid validation
- End-to-end workflow testing

---

## **CI/CD Pipeline**

### **GitHub Actions Workflow (.github/workflows/ci.yml)**
```yaml
name: Housing Price Prediction CI/CD Pipeline

on:
  push:
    branches: [ main, reg, hyper ]
  pull_request:
    branches: [ main ]
```

### **Pipeline Features**
- **Multi-Python Testing**: Python 3.8 and 3.9
- **Automated Testing**: Runs complete test suite
- **Code Quality**: Linting with flake8
- **Code Formatting**: Black formatter checks
- **Artifact Management**: Stores results as artifacts
- **Deployment**: Automated deployment on main branch

### **Pipeline Steps**
1. **Environment Setup**: Install dependencies
2. **Code Quality**: Linting and formatting checks
3. **Data Validation**: Test data loading functionality
4. **Model Testing**: Basic regression model validation
5. **Hyperparameter Testing**: Advanced model tuning (non-main branches)
6. **Artifact Storage**: Save results for analysis
7. **Deployment**: Deploy complete analysis (main branch only)

---

## **Setup and Usage**

### **Environment Setup**
```bash
# Create conda environment
conda create -n housing-regression python=3.8 -y

# Activate environment
conda activate housing-regression

# Install dependencies
pip install -r requirements.txt
```

### **Execution Commands**
```bash
# Run complete analysis
python regression.py --mode both --save-results

# Run basic regression only
python regression.py --mode basic --save-results

# Run hyperparameter tuning only
python regression.py --mode hyperparameter --save-results

# Run data exploration
python data_exploration.py

# Run tests
python test_regression.py
```

### **Git Workflow**
```bash
# Clone repository
git clone https://github.com/Avi2001-dot/MLops-project-1.git

# Check branches
git branch -a

# Switch to specific branch
git checkout reg    # Basic regression
git checkout hyper  # Hyperparameter tuning
git checkout main   # Complete implementation
```

---

## **Results and Outputs**

### **Generated Files**
- `basic_regression_results.csv`: Basic model performance metrics
- `hyperparameter_tuned_results.csv`: Tuned model performance metrics
- `dataset_exploration.png`: Data visualization plots
- `feature_correlations.png`: Feature importance analysis
- `feature_boxplots.png`: Feature distribution analysis

### **Performance Metrics**
- **Mean Squared Error (MSE)**: Lower values indicate better performance
- **R² Score**: Higher values indicate better fit (maximum 1.0)
- **Cross-validation R²**: Average performance across multiple folds
- **Standard Deviation**: Consistency measure across folds

---

## **Key Achievements**

### **MLOps Best Practices**
1. **Version Control**: Proper Git branching strategy
2. **Automated Testing**: Comprehensive test suite
3. **CI/CD Integration**: GitHub Actions pipeline
4. **Code Quality**: Linting and formatting standards
5. **Documentation**: Complete project documentation
6. **Reproducibility**: Fixed random seeds and environment management

### **Machine Learning Excellence**
1. **Model Diversity**: Three different algorithm families
2. **Hyperparameter Optimization**: Grid search with cross-validation
3. **Performance Evaluation**: Multiple metrics and validation strategies
4. **Feature Engineering**: Proper preprocessing and scaling
5. **Data Validation**: Comprehensive data quality checks

---

## **Troubleshooting Guide**

### **Common Issues**
1. **Module Not Found**: Ensure conda environment is activated
2. **Network Errors**: Check internet connection for data loading
3. **Permission Denied**: Verify GitHub credentials and repository access
4. **Dependency Conflicts**: Use exact versions from requirements.txt

### **Solutions**
```bash
# Environment issues
conda deactivate && conda activate housing-regression

# Package issues
pip install --upgrade pip && pip install -r requirements.txt

# Git issues
git config --global user.name "YourUsername"
git config --global user.email "your.email@example.com"
```

---

## 📚 **References and Resources**

### **Dataset Reference**
- Boston Housing Dataset: http://lib.stat.cmu.edu/datasets/boston
- Scikit-learn Documentation: https://scikit-learn.org/stable/

### **Technical Documentation**
- Python: https://docs.python.org/3/
- Pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/stable/contents.html
- GitHub Actions: https://docs.github.com/en/actions

### **MLOps Resources**
- Git Best Practices: https://git-scm.com/doc
- CI/CD Pipelines: https://github.com/features/actions
- Machine Learning Workflows: https://ml-ops.org/

---

## 👨‍💻 **Author Information**

**Student**: Avinash Singh  
**Course**: Machine Learning Operations (MLOps)  
**Assignment**: Assignment 1 - Housing Price Prediction  
**Institution**: [Your Institution Name]  
**Date**: January 2025  

**GitHub Repository**: https://github.com/Avi2001-dot/MLops-project-1  
**Contact**: [Your Email Address]  

---

## 📜 **License**

This project is created for educational purposes as part of MLOps coursework. All code and documentation are available under the MIT License.

---

## 🎉 **Conclusion**

This project successfully demonstrates a complete MLOps workflow for machine learning model development, including:

- **Comprehensive ML Pipeline**: From data loading to model evaluation
- **Advanced Model Optimization**: Hyperparameter tuning with cross-validation
- **Production-Ready Code**: Modular, tested, and documented
- **Automated CI/CD**: GitHub Actions integration
- **Best Practices**: Version control, testing, and documentation

The Gradient Boosting Regressor achieved the best performance with an R² score of 0.92, demonstrating the effectiveness of ensemble methods for regression tasks. The complete MLOps workflow ensures reproducibility, maintainability, and scalability of the machine learning solution.

**Project Status**: **COMPLETED SUCCESSFULLY**

---

*This documentation serves as a comprehensive guide for the MLOps Housing Price Prediction project, covering all technical aspects, implementation details, and usage instructions.* 