# Housing Price Prediction - MLOps Assignment 1
## Complete Setup and Execution Guide

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip
- Git
- Internet connection for data loading

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Create conda environment
conda create -n housing-regression python=3.8 -y

# Activate environment
conda activate housing-regression

# Install required packages
pip install -r requirements.txt
```

#### Option B: Using pip
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter pytest flake8 black
```

### 2. Project Structure Overview
```
assignment1/
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI/CD pipeline
├── utils.py                   # Utility functions for ML workflow
├── regression.py              # Main regression analysis script
├── data_exploration.py        # Data exploration and visualization
├── test_regression.py         # Test suite for validation
├── setup.py                   # Automated setup script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── INSTRUCTIONS.md            # This file
```

### 3. Running the Project

#### Quick Start
```bash
# Run complete analysis (both basic and hyperparameter tuning)
python regression.py --mode both --save-results

# Run only basic regression
python regression.py --mode basic --save-results

# Run only hyperparameter tuning
python regression.py --mode hyperparameter --save-results
```

#### Data Exploration
```bash
# Explore dataset and create visualizations
python data_exploration.py
```

#### Running Tests
```bash
# Run all tests to verify functionality
python test_regression.py
```

### 4. Git Repository Setup

#### Initialize Repository
```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Setup main branch with README"
```

#### Create reg Branch
```bash
# Create and switch to reg branch
git checkout -b reg

# Add regression-specific files
git add utils.py regression.py

# Commit regression implementation
git commit -m "Add basic regression models implementation"
```

#### Create hyper Branch
```bash
# Create and switch to hyper branch
git checkout -b hyper

# Add hyperparameter tuning files
git add .

# Commit hyperparameter tuning implementation
git commit -m "Add hyperparameter tuning implementation"
```

#### Merge Branches
```bash
# Switch to main branch
git checkout main

# Merge reg branch
git merge reg

# Merge hyper branch
git merge hyper
```

### 5. GitHub Setup

#### Create GitHub Repository
1. Go to GitHub.com
2. Click "New Repository"
3. Name it "HousingRegression"
4. Make it public
5. Don't initialize with README (we already have one)

#### Push to GitHub
```bash
# Add GitHub remote
git remote add origin https://github.com/Avi2001-dot/HousingRegression.git

# Push main branch
git push -u origin main

# Push other branches
git push origin reg
git push origin hyper
```

### 6. Automated Setup (Alternative)

Use the provided setup script for automated configuration:

```bash
# Run complete setup
python setup.py --all

# Or run individual components
python setup.py --setup-env        # Setup conda environment
python setup.py --setup-git        # Setup git repository
python setup.py --explore-data     # Run data exploration
python setup.py --run-tests        # Run tests
python setup.py --complete-analysis # Run complete analysis
```

### 7. Expected Outputs

#### Files Generated
- `basic_regression_results.csv` - Results from basic models
- `hyperparameter_tuned_results.csv` - Results from tuned models
- `dataset_exploration.png` - Dataset visualization
- `feature_correlations.png` - Feature correlation plot
- `feature_boxplots.png` - Feature distribution plots

#### Performance Metrics
- **Mean Squared Error (MSE)**: Lower is better
- **R² Score**: Higher is better (max 1.0)
- **Cross-validation R²**: Average performance across folds

#### Expected Results
Based on the Boston Housing dataset:
- Linear Regression: R² ≈ 0.67
- Random Forest: R² ≈ 0.89
- Gradient Boosting: R² ≈ 0.92

### 8. GitHub Actions CI/CD

The project includes automated CI/CD pipeline that:
- Runs tests on Python 3.8 and 3.9
- Performs code linting with flake8
- Tests data loading and model training
- Archives results as artifacts
- Deploys complete analysis on main branch

### 9. Assignment Requirements Checklist

- ✅ Conda environment setup
- ✅ Requirements.txt file
- ✅ GitHub repository with command line setup
- ✅ Three regression models (Linear, Random Forest, Gradient Boosting)
- ✅ Hyperparameter tuning (3+ parameters per model)
- ✅ Performance comparison (MSE/R²)
- ✅ Modular code structure with utils.py
- ✅ GitHub Actions CI/CD pipeline
- ✅ Three branches (main, reg, hyper)
- ✅ Proper documentation

### 10. Troubleshooting

#### Common Issues
1. **Module not found**: Ensure environment is activated
2. **Network error**: Check internet connection for data loading
3. **Permission denied**: Use appropriate Git credentials
4. **Merge conflicts**: Follow Git best practices

#### Solutions
```bash
# If environment issues
conda deactivate
conda activate housing-regression

# If package issues
pip install --upgrade pip
pip install -r requirements.txt

# If git issues
git status
git log --oneline
```

### 11. Report Documentation

For your assignment report, document:
1. **Setup Process**: Screenshots of environment setup
2. **Model Performance**: Include MSE and R² metrics
3. **Hyperparameter Impact**: Show improvement percentages
4. **Visualizations**: Include generated plots
5. **GitHub Repository**: Link to your public repository
6. **CI/CD Pipeline**: Screenshots of GitHub Actions

### 12. Next Steps

1. ✅ Complete the implementation
2. ✅ Run all tests and verify functionality
3. ✅ Set up GitHub repository with all branches
4. ✅ Push code to GitHub
5. ✅ Verify CI/CD pipeline works
6. 📝 Create assignment report (PDF)
7. 📝 Include GitHub repository link in report
8. 📝 Document all steps and results

### Contact Information
- **Student**: Avinash Singh
- **Assignment**: MLOps Assignment 1
- **Repository**: https://github.com/Avi2001-dot/HousingRegression

---
*This project demonstrates a complete MLOps workflow for regression analysis with proper version control, automated testing, and CI/CD pipeline.* 