#!/bin/bash

# Update the package list and ensure pip is up-to-date
echo "Updating package list and pip..."
sudo apt-get update
python3 -m pip install --upgrade pip

# Install libraries
echo "Installing required Python libraries..."
python3 -m pip install shap torch torchvision matplotlib torchaudio

# Verify installation
echo "Verifying installation..."
python3 -m pip show shap torch torchvision matplotlib torchaudio | grep "Name\|Version"

echo "Installation completed successfully."