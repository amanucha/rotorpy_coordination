.PHONY: help setup run test clean lint format

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "rotorpy - Available commands:"
	@echo ""
	@echo "  make setup   - Create venv and install rotorpy package"
	@echo "  make run     - Run coordination_with_mpc.py"
	@echo "  make clean   - Remove venv, build artifacts, and cache"
	@echo ""
	@echo "Manual activation: source venv/bin/activate"

# Install package from pyproject.toml
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: pyproject.toml
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing rotorpy package..."
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo "✅ Setup complete!"

# Install with development dependencies (if defined in pyproject.toml)
setup-dev: setup
	$(PIP) install -e ".[dev]"

# Run the MPC coordination script
run: setup
	$(PYTHON) examples/coordination_with_mpc.py

# Run without MPC
run-simple: setup
	$(PYTHON) examples/rotorpy_without_mpc.py

# Run tests (if you have a tests directory)
test: setup
	$(PYTHON) -m pytest tests/ -v

# Clean everything
clean:
	rm -rf $(VENV)
	rm -rf build/
	rm -rf rotorpy.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Clean only cache, keep venv
clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
