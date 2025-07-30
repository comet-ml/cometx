# Pre-commit Setup for CometX

This document describes the pre-commit setup for the CometX repository.

## What's Included

The pre-commit configuration includes the following hooks:

### Automatic Hooks (run on every commit)
- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting (compatible with Black)
- **flake8**: Linting (with Black-compatible settings)
- **bandit**: Security checks
- **pre-commit hooks**: Various general checks:
  - Merge conflict detection
  - YAML/JSON validation
  - Large file detection
  - Case conflict detection
  - Docstring placement
  - AST validation
  - Debug statement detection
  - End-of-file fixing
  - Trailing whitespace removal
  - TOML validation
  - VCS permalink checking
  - Mixed line ending detection
  - Requirements.txt sorting

### Manual Hooks (run only when explicitly called)
- **mypy**: Type checking (basic mode)
- **pydocstyle**: Documentation style checking

## Setup Instructions

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. (Optional) Run the setup script:
   ```bash
   ./setup-pre-commit.sh
   ```

## Usage

### Automatic Checks
The automatic hooks will run every time you commit. If any fail, the commit will be blocked until the issues are fixed.

### Manual Checks
To run the manual hooks:

```bash
# Run all manual hooks
pre-commit run --all-files --hook-stage manual

# Run specific manual hooks
pre-commit run mypy --all-files
pre-commit run pydocstyle --all-files
```

### Running Specific Hooks
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run isort --all-files
```

## Configuration Files

- `.pre-commit-config.yaml`: Main pre-commit configuration
- `.flake8`: Flake8 linting configuration
- `pyproject.toml`: Black, isort, and mypy configuration

## Current Status

✅ **Working Hooks:**
- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- bandit (security checks)
- All pre-commit general hooks

⚠️ **Manual Hooks:**
- mypy (type checking) - set to manual due to existing type issues
- pydocstyle (documentation) - set to manual due to extensive docstring issues

## Known Issues

1. **Line Length Violations**: Some files exceed the 88-character line limit
2. **Type Annotations**: Some files have syntax errors in type annotations
3. **Documentation**: Many functions and classes lack proper docstrings
4. **Security Warnings**: Bandit found some issues (mostly in tests)

## Next Steps

To improve the code quality:

1. **Fix line length violations**: Run `pre-commit run black --all-files` to auto-format
2. **Fix type annotations**: Address mypy errors in `cometx/framework/comet/download_manager.py`
3. **Add docstrings**: Gradually add proper documentation to functions and classes
4. **Address security issues**: Review and fix bandit warnings (especially the `eval` usage in `cometx/utils.py`)

## Disabling Hooks

If you need to bypass pre-commit hooks temporarily:

```bash
git commit --no-verify -m "Your commit message"
```

**Note**: This should only be used in emergencies, not as a regular practice.
