# Housing Price Prediction - MLOps Assignment 1

## GitHub Repository
**Repository Link**: https://github.com/Avi2001-dot/HousingRegression

## Project Overview
This project implements a complete machine learning workflow to predict house prices using classical machine learning models on the Boston Housing dataset.

## Assignment Requirements
- Compare minimum three regression models
- Implement hyperparameter tuning (minimum 3 hyperparameters per model)
- Create automated CI/CD pipeline using GitHub Actions
- Follow modular programming approach
- Performance comparison using MSE/R² metrics

## Repository Structure
```
HousingRegression/
|-- .github/workflows/
|   |-- ci.yml
|-- utils.py
|-- regression.py
|-- requirements.txt
|-- README.md
```

## Branches
- **main**: Contains the merged code from both reg and hyper branches
- **reg**: Basic regression models implementation
- **hyper**: Regression models with hyperparameter tuning

## Dataset
Using Boston Housing dataset loaded from: http://lib.stat.cmu.edu/datasets/boston

## Setup Instructions
1. Create conda environment: `conda create -n housing-regression python=3.8`
2. Activate environment: `conda activate housing-regression`
3. Install requirements: `pip install -r requirements.txt`
4. Run the regression analysis: `python regression.py`

## Models Implemented
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Performance Metrics
- Mean Squared Error (MSE)
- R² Score
- Cross-validation scores

## Author
Avinash Singh - Roll No: G24AI1027