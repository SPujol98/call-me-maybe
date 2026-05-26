# Compiler / Tools
UV = uv

# Main files
MAIN = -m src

# MyPy Flags
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

# Terminal colors
GREEN = \033[0;32m
CYAN  = \033[0;36m
NC    = \033[0m

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	$(UV) sync

run: install
	$(UV) run python $(MAIN)

debug: install
	$(UV) run python -m pdb $(MAIN)

clean:
	@echo "$(CYAN)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "$(GREEN)Clean completed.$(NC)"

lint: install
	@echo "$(CYAN)Executing flake8...$(NC)"
	$(UV) run flake8 .
	@echo "$(CYAN)Executing mypy...$(NC)"
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict: install
	@echo "$(CYAN)Executing flake8...$(NC)"
	$(UV) run flake8 .
	@echo "$(CYAN)Executing mypy strict...$(NC)"
	$(UV) run mypy . --strict