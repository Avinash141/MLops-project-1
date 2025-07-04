"""
Setup script for Housing Price Prediction - MLOps Assignment 1
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Success")
            if result.stdout:
                print(result.stdout)
        else:
            print("✗ Error")
            if result.stderr:
                print(result.stderr)
            
    except Exception as e:
        print(f"✗ Exception: {e}")

def setup_environment():
    """Set up the conda environment and install requirements."""
    print("=" * 60)
    print("SETTING UP ENVIRONMENT")
    print("=" * 60)
    
    # Create conda environment
    run_command(
        "conda create -n housing-regression python=3.8 -y",
        "Creating conda environment 'housing-regression'"
    )
    
    # Install requirements
    run_command(
        "conda run -n housing-regression pip install -r requirements.txt",
        "Installing Python packages"
    )
    
    print("\n✓ Environment setup complete!")
    print("Activate the environment with: conda activate housing-regression")

def run_data_exploration():
    """Run data exploration."""
    print("=" * 60)
    print("RUNNING DATA EXPLORATION")
    print("=" * 60)
    
    run_command(
        "python data_exploration.py",
        "Exploring dataset and creating visualizations"
    )

def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    run_command(
        "python test_regression.py",
        "Running regression tests"
    )

def run_basic_regression():
    """Run basic regression models."""
    print("=" * 60)
    print("RUNNING BASIC REGRESSION")
    print("=" * 60)
    
    run_command(
        "python regression.py --mode basic --save-results",
        "Training basic regression models"
    )

def run_hyperparameter_tuning():
    """Run hyperparameter tuning."""
    print("=" * 60)
    print("RUNNING HYPERPARAMETER TUNING")
    print("=" * 60)
    
    run_command(
        "python regression.py --mode hyperparameter --save-results",
        "Performing hyperparameter tuning"
    )

def run_complete_analysis():
    """Run complete analysis."""
    print("=" * 60)
    print("RUNNING COMPLETE ANALYSIS")
    print("=" * 60)
    
    run_command(
        "python regression.py --mode both --save-results",
        "Running complete analysis with both basic and tuned models"
    )

def setup_git_repository():
    """Set up Git repository with proper branching structure."""
    print("=" * 60)
    print("SETTING UP GIT REPOSITORY")
    print("=" * 60)
    
    # Initialize git repository
    run_command("git init", "Initializing Git repository")
    
    # Add files to staging
    run_command("git add .", "Adding files to staging area")
    
    # Initial commit
    run_command(
        'git commit -m "Initial commit: Setup main branch with README"',
        "Creating initial commit"
    )
    
    # Create and checkout reg branch
    run_command("git checkout -b reg", "Creating reg branch")
    
    # Add regression files
    run_command("git add utils.py regression.py", "Adding regression files")
    run_command(
        'git commit -m "Add basic regression models implementation"',
        "Committing regression implementation"
    )
    
    # Create and checkout hyper branch
    run_command("git checkout -b hyper", "Creating hyper branch")
    
    # Add hyperparameter tuning files
    run_command("git add .", "Adding hyperparameter tuning files")
    run_command(
        'git commit -m "Add hyperparameter tuning implementation"',
        "Committing hyperparameter tuning"
    )
    
    # Merge branches back to main
    run_command("git checkout main", "Switching to main branch")
    run_command("git merge reg", "Merging reg branch to main")
    run_command("git merge hyper", "Merging hyper branch to main")
    
    print("\n✓ Git repository setup complete!")
    print("Branches created: main, reg, hyper")

def display_project_structure():
    """Display the project structure."""
    print("=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    
    structure = """
    HousingRegression/
    |-- .github/workflows/
    |   |-- ci.yml
    |-- utils.py
    |-- regression.py
    |-- data_exploration.py
    |-- test_regression.py
    |-- setup.py
    |-- requirements.txt
    |-- README.md
    """
    print(structure)

def main():
    """Main function to run setup tasks."""
    parser = argparse.ArgumentParser(description='Setup script for Housing Price Prediction')
    parser.add_argument('--setup-env', action='store_true', 
                       help='Set up conda environment')
    parser.add_argument('--setup-git', action='store_true', 
                       help='Set up Git repository with branches')
    parser.add_argument('--explore-data', action='store_true', 
                       help='Run data exploration')
    parser.add_argument('--run-tests', action='store_true', 
                       help='Run all tests')
    parser.add_argument('--basic-regression', action='store_true', 
                       help='Run basic regression models')
    parser.add_argument('--hyperparameter-tuning', action='store_true', 
                       help='Run hyperparameter tuning')
    parser.add_argument('--complete-analysis', action='store_true', 
                       help='Run complete analysis')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tasks')
    
    args = parser.parse_args()
    
    # Display project structure
    display_project_structure()
    
    if args.all:
        setup_environment()
        setup_git_repository()
        run_data_exploration()
        run_tests()
        run_complete_analysis()
    else:
        if args.setup_env:
            setup_environment()
        
        if args.setup_git:
            setup_git_repository()
        
        if args.explore_data:
            run_data_exploration()
        
        if args.run_tests:
            run_tests()
        
        if args.basic_regression:
            run_basic_regression()
        
        if args.hyperparameter_tuning:
            run_hyperparameter_tuning()
        
        if args.complete_analysis:
            run_complete_analysis()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Activate environment: conda activate housing-regression")
    print("2. Run analysis: python regression.py --mode both --save-results")
    print("3. Create GitHub repository and push code")
    print("4. Document results in your assignment report")

if __name__ == "__main__":
    main() 