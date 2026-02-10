#!/bin/bash

# Define project directories
PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run training
echo "Starting model training..."
python src/train.py

# Run dashboard
echo "Launching dashboard..."
streamlit run src/app.py --server.headless true
