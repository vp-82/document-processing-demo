import subprocess
import sys

def run_shell_command(command):
    try:
        print(f"Running command: {command}")
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print("Command output:", result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("Error running command:", e.output.decode('utf-8'))

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_dependencies():
    dependencies = [
        "pytesseract",
        "pillow",
        "transformers",
        "scikit-learn",
        "pandas",
        "datasets",
    ]

    for dependency in dependencies:
        try:
            install(dependency)
            print(f"{dependency} installed successfully.")
        except Exception as e:
            print(f"Error installing {dependency}: {e}")

def import_libraries():
    print("Importing libraries...")
    global os, pytesseract, Image, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, confusion_matrix, classification_report, pd, load_dataset

    import os
    import pytesseract
    from PIL import Image
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.metrics import confusion_matrix, classification_report
    import pandas as pd
    from datasets import load_dataset

    print("Libraries imported successfully.")

if __name__ == "__main__":
    linux_commands = [
        "sudo apt-get update",
        "sudo apt-get install software-properties-common -y",
        "sudo add-apt-repository ppa:alex-p/tesseract-ocr -y",
        "sudo apt-get update",
        "sudo apt-get install tesseract-ocr -y",
    ]
    for command in linux_commands:
        run_shell_command(command)

    install_dependencies()
    import_libraries()
