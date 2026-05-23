PYTHON_SYS = python3
VENV = venv
BIN = $(VENV)/bin
DEPS_STAMP = $(VENV)/.deps_installed

PYTHON = $(BIN)/python
PIP = $(BIN)/pip
MYPY = $(BIN)/mypy
FLAKE8 = $(BIN)/flake8

# Main files
MAIN = call_me_maybe.py

# MyPy Flags
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

# Terminal colors
GREEN = \033[0;32m
CYAN  = \033[0;36m
NC    = \033[0m

.PHONY: all install run debug clean lint lint-strict

all: run

install: $(DEPS_STAMP)

$(BIN)/python:
	@echo "$(CYAN)Checking virtual environment...$(NC)"
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON_SYS) -m venv $(VENV); \
		echo "$(GREEN)Virtual environment created/$(VENV)$(NC)"; \
	fi

$(DEPS_STAMP): requirements.txt Makefile | $(BIN)/python
	@echo "$(CYAN)Installing/Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(DEPS_STAMP)

# 2. Program execution with venv
run: install
	@if [ ! -f "$(BIN)/python" ]; then echo "Error: Execute 'make install' first."; exit 1; fi
	$(PYTHON) $(MAIN)

# 3. Debug mode with pdb
debug: install
	$(PYTHON) -m pdb $(MAIN)

clean:
	@echo "$(CYAN)Cleaning temporary files and the virtual environment...$(NC)"
# find . -name "*.txt" ! -name "requirements.txt" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf $(VENV)
	@echo "$(GREEN)Clean completed.$(NC)"

# 5. Linter
lint: install
	@echo "Executing flake8..."
	$(FLAKE8) . --exclude=$(VENV)
	@echo "Executing mypy..."
	$(MYPY) . $(MYPY_FLAGS) --exclude $(VENV)

# 6. Strict linter
lint-strict: install
	$(FLAKE8) . --exclude=$(VENV)
	$(MYPY) . --strict --exclude $(VENV)