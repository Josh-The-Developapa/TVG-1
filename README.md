# TumourVision G1 (TVG-1)

TumourVision G1 (TVG-1) is a powerful Convolutional Neural Network (CNN) designed to classify brain MRI scans for brain tumor detection. The model is built using PyTorch and leverages Torch, TorchVision, and Matplotlib libraries to achieve accurate and insightful results. Current structure supports approx 22 million parameters

TVG-1 currently runs with a 56.45% Accuracy. Still in active development

_Note_ - `CLOD.webp` in full is Convolutional Layers' Output Dimensions. Just as an abbreviation for easy file naming

## Overview

TVG-1 excels at identifying brain tumors in MRI scans, providing a crucial diagnostic tool for medical professionals. With advanced convolutional neural network architecture and meticulous training, TVG-1 has achieved an impressive accuracy rate of 56.45% on test data.

## Features

- State-of-the-art CNN architecture using PyTorch
- Utilizes Torch and TorchVision for efficient neural network operations
- Generates informative visualizations using Matplotlib
- Accurate brain tumor classification with 56.45% Accuracy, aiming for 80%+

## Dataset

The dataset used for training and testing TVG-1 was sourced from two Kaggle datasets:

- [Brain Tumour Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor)

## Usage

1. Clone the repository: `git clone https://github.com/Josh-The-Developapa/TVG-1.git`
2. Navigate to the project directory: `cd TVG-1`
3. Set up a Python virtual environment (recommended).
4. Install the required dependencies: `pip install -r requirements.txt`
5. Prepare your dataset following the project's data structure.
6. Run the TVG-1 model and observe the results.
7. Customize and fine-tune the model according to your needs.

## Example

```bash
# Clone the repository
git clone https://github.com/Josh-The-Developapa/TVG-1.git

# Navigate to the project directory
cd TVG-1

# Set up a Python virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Designed and developed by **Joshua Mukisa**
