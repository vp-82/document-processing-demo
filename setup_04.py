import subprocess
import sys

# Function to run a shell command
def run_shell_command(command):
    try:
        # Print the command that is going to be run
        print(f"Running command: {command}")
        # Run the command and get the output
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        # Print the output of the command
        print("Command output:", result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        # If there was an error running the command, print the error
        print("Error running command:", e.output.decode('utf-8'))

# Function to install a Python package using pip
def install(package):
    print(f"Installing {package}...")
    # Run the pip install command
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Function to install a list of dependencies
def install_dependencies():
    # List of dependencies to install
    dependencies = [
        "pytesseract",
        "pillow",
        "transformers",
        "scikit-learn",
        "pandas",
        "datasets",
        "wandb",
        "python-dotenv",
    ]

    # Loop over each dependency and try to install it
    for dependency in dependencies:
        try:
            install(dependency)
            print(f"{dependency} installed successfully.")
        except Exception as e:
            # If there was an error installing the dependency, print the error
            print(f"Error installing {dependency}: {e}")

# Function to import the necessary Python libraries
def import_libraries():
    print("Importing libraries...")
    # Declare the libraries as global variables
    global os, pytesseract, Image, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, confusion_matrix, classification_report, pd, load_dataset

    # Import the necessary Python libraries
    import os
    import pytesseract
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import confusion_matrix, classification_report
    import pandas as pd
    from datasets import load_dataset

    print("Libraries imported successfully.")

# Main entry point for the script
if __name__ == "__main__":
    # List of Linux commands to run
    linux_commands = [
        "sudo apt-get update",
        "sudo apt-get install software-properties-common -y",
        "sudo add-apt-repository ppa:alex-p/tesseract-ocr -y",
        "sudo apt-get update",
        "sudo apt-get install tesseract-ocr -y",
    ]
    # Run each Linux command
    for command in linux_commands:
        run_shell_command(command)

    # Install the dependencies
    install_dependencies()
    # Import the Python libraries
    import_libraries()
