#!/bin/bash

# Stop execution on error
set -e

# Step 1: Update the system and install necessary packages
echo "Updating system and installing prerequisites..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential

# Step 2: Create and activate a virtual environment
echo "Creating Python virtual environment..."
python3 -m ven env
source env/bin/activate

# Step 3: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Step 5: Verify the installation
echo "Verifying installations..."
python -c "import cupy; print(f'Cupy version: {cupy.__version__}')"

# Step 6: Provide completion message
echo "Environment setup complete. Use 'source env/bin/activate' to activate the virtual environment."
