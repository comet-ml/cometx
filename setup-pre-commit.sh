#!/bin/bash

# Setup script for pre-commit hooks

echo "Setting up pre-commit hooks..."

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Run pre-commit on all files (optional)
echo "Running pre-commit on all files..."
pre-commit run --all-files

echo "Pre-commit setup complete!"
echo ""
echo "To run pre-commit manually:"
echo "  pre-commit run --all-files"
echo ""
echo "To run specific hooks:"
echo "  pre-commit run black --all-files"
echo "  pre-commit run flake8 --all-files"
echo "  pre-commit run mypy --all-files"
