"""
Requirements file for running the unit tests
Save as: test_requirements.txt
"""
# Core testing framework
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Scientific computing
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
polars>=0.16.0

# Machine learning
scikit-learn>=1.0.0

# Molecular dynamics and chemistry
MDAnalysis>=2.0.0
mdtraj>=1.9.0
openmm>=7.5.0
parmed>=3.4.0
openbabel-wheel>=3.1.0
rdkit>=2022.03.1
biopython>=1.79

# Visualization (optional for tests but may be imported)
matplotlib>=3.4.0
seaborn>=0.11.0

# Progress bars
tqdm>=4.60.0

# Other dependencies
pydantic>=1.8.0
pyyaml>=5.4.0
numba>=0.55.0
parsl>=1.2.0

"""
Test runner script
Save as: run_tests.py
"""
#!/usr/bin/env python
"""
Test runner script for the molecular dynamics library
Usage: python run_tests.py [options]
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_dir=".", coverage=False, verbose=False, specific_test=None):
    """
    Run pytest with specified options
    
    Args:
        test_dir: Directory containing tests
        coverage: Enable coverage reporting
        verbose: Enable verbose output
        specific_test: Run only a specific test file or test
    """
    cmd = ["pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    if specific_test:
        cmd.append(specific_test)
    else:
        cmd.append(test_dir)
    
    # Run tests
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run unit tests for the molecular dynamics library")
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--test", "-t",
        type=str,
        help="Run specific test file or test (e.g., test_autocluster.py or test_autocluster.py::TestGenericDataloader)"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"])
        return
    
    print("Running tests...")
    exit_code = run_tests(
        coverage=args.coverage,
        verbose=args.verbose,
        specific_test=args.test
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

"""
Makefile for running tests
Save as: Makefile
"""
.PHONY: test test-verbose test-coverage test-specific install-test-deps clean

# Run all tests
test:
	python run_tests.py

# Run tests with verbose output
test-verbose:
	python run_tests.py --verbose

# Run tests with coverage
test-coverage:
	python run_tests.py --coverage --verbose

# Run specific test file
test-specific:
	@echo "Usage: make test-specific TEST=test_filename.py"
	python run_tests.py --test $(TEST)

# Install test dependencies
install-test-deps:
	python run_tests.py --install-deps

# Clean up generated files
clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run all test suites individually
test-all-individual:
	python -m pytest test_autocluster.py -v
	python -m pytest test_fingerprinter.py -v
	python -m pytest test_sasa.py -v
	python -m pytest test_ipsae.py -v

"""
GitHub Actions CI configuration
Save as: .github/workflows/test.yml
"""
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r test_requirements.txt
    
    - name: Run tests with coverage
      run: |
        python run_tests.py --coverage --verbose
    
    - name: Upload coverage reports to Codecov
      uses: codecov/codecov-action@v3
      env:
        CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
