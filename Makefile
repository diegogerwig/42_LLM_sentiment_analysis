PYTHON := python3
PIP := pip
SRC_DIR := ./src
MODEL_DIR := ./models/sentiment_model
DATA_DIR := ./data
VENV := $(HOME)/.venvs/sentiment_analysis
VENV_BIN := $(VENV)/bin
ENV_FILE := .env

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make train      - Run the training notebook/script"
	@echo "  make upload     - Upload model to Hugging Face (requires .env)"
	@echo "  make run        - Run the Streamlit app"
	@echo "  make clean      - Remove virtual environment and cached files"

check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Error: .env file not found. Please create it with your HF_TOKEN"; \
		exit 1; \
	fi

setup:
	@echo "Creating project structure..."
	mkdir -p $(SRC_DIR)
	mkdir -p $(MODEL_DIR)
	mkdir -p $(DATA_DIR)
	@echo "Setting up virtual environment..."
	mkdir -p $(HOME)/.venvs  
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	@echo "Verifying critical dependencies..."
	$(VENV_BIN)/pip install python-dotenv streamlit transformers torch --no-deps
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Creating template .env file..."; \
		echo "HF_TOKEN=your-token-here" > $(ENV_FILE); \
		echo "Please update .env with your Hugging Face token"; \
	fi

train:
	@echo "Running training script..."
	$(VENV_BIN)/python $(SRC_DIR)/sentiment-analysis_v0.py

upload: check-env
	@echo "Uploading model to Hugging Face..."
	$(VENV_BIN)/python $(SRC_DIR)/upload_model.py

run: check-env
	@echo "Starting Streamlit app..."
	$(VENV_BIN)/streamlit run $(SRC_DIR)/app_main.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

requirements:
	@echo "streamlit>=1.24.0" > requirements.txt
	@echo "transformers>=4.30.0" >> requirements.txt
	@echo "torch>=2.0.0" >> requirements.txt
	@echo "pandas>=1.5.0" >> requirements.txt
	@echo "huggingface-hub>=0.16.4" >> requirements.txt
	@echo "python-dotenv>=0.19.0" >> requirements.txt
	@echo "jupyter" >> requirements.txt
	@echo "flake8" >> requirements.txt
	@echo "pytest" >> requirements.txt
	@echo "scikit-learn" >> requirements.txt
	@echo "numpy" >> requirements.txt
	@echo "tqdm" >> requirements.txt

deploy: setup upload run

.PHONY: setup train upload run clean help check-env deploy