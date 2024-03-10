#!/bin/bash

echo "Setting up environment..."

# Function to simulate a loading bar
loading_bar() {
    echo -n "Starting process: "
    for i in {1..50}; do
        echo -n "█"
        sleep 0.1
    done
    echo " Done."
}

# Step 1: Create a Python virtual environment named ex00
python3 -m venv ex00 && echo "Virtual environment created."

# Simulate loading (not actual progress)
loading_bar

# Step 2: Activate the virtual environment
source ex00/bin/activate && echo "Virtual environment activated."

# Simulate loading (not actual progress)
loading_bar

# Step 3: Install dependencies from requirements.txt
pip3 install -r requirements.txt && echo "Dependencies installed."

# Simulate loading (not actual progress)
loading_bar

# Step 4: Launch Jupyter Lab on port 8891
echo "Launching Jupyter Lab..."
jupyter lab --port 8891 &
sleep 2 # Give Jupyter a moment to start

echo "Jupyter Lab should now be running on port 8891."