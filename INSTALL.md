# Installation Guide

This guide will help you set up the cometx project for development and testing.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cometx
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

## Running Tests

### Basic Test Suite
```bash
pytest tests
```

### With Coverage
```bash
pytest tests --cov=cometx
```

### Integration Tests
To run integration tests that require Comet ML credentials:
```bash
COMET_USER=your_username pytest tests
```

## Development Setup

For additional development tools:
```bash
pip install -r requirements-dev.txt
```

## Environment Variables

The following environment variables may be required for certain tests:

- `COMET_USER`: Your Comet ML username (for integration tests)
- `COMET_API_KEY`: Your Comet ML API key (for integration tests)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've installed the package in development mode with `pip install -e .`

2. **Missing dependencies**: Run `pip install -r requirements.txt` to install all required packages

3. **Test failures**: Some tests require Comet ML credentials. Set the `COMET_USER` environment variable or the tests will be skipped gracefully.

### Python Version

This project requires Python 3.6 or higher. Python 3.12 is recommended for the best experience.
