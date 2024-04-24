# TumourVision G2 (TVG-2)

<!-- add more illustrations -->
<!-- make this a PyPI module -->
<!-- later add X accuracy -->
<!-- may require logistic regression analysis -->

TumourVision G2 (TVG-2), an advancement over [TVG-1](https://github.com/Josh-The-Developapa/TVG-1.git), incorporates [PolyTrend](https://github.com/asiimwemmanuel/polytrend) methodologies to improve brain tumor detection in MRI scans. Utilizing a Convolutional Neural Network (CNN) architecture developed with PyTorch, TVG-2 classifies brain MRI scans for tumor detection. It employs Torch, TorchVision, and Matplotlib libraries for analysis and visualization. With around 22 million parameters, TVG-2 provides a robust computational framework. While TVG-1 achieves 56.45% accuracy, TVG-2 is in active development, enhancing diagnostic capabilities for medical professionals.

> `CLOD_formulas.webp` in full is Convolutional Layers' Output Dimensions. Just as an abbreviation for easy file naming

## Features

- Integration of PolyTrend's polynomial trend analysis for data preprocessing and feature extraction.
- Utilization of polynomial regression techniques to refine CNN predictions and improve classification accuracy.
- Seamless integration with existing TVG-1 architecture for easy adoption and deployment.


## Dataset

<!-- GET MORE DATA -->

The dataset used for training and testing TVG-1 was sourced from two Kaggle datasets:

- [Brain Tumour Dataset](https://www.kaggle.com/datasets/denizkavi1/brain-tumor)

## Installation

You can install "TumorVision" via pip by running:

```bash
pip install tumorvision
```

Alternatively, if you prefer to install it from the source code, you can follow these steps:

1. Clone the TVG-2 repository:
```bash
git clone https://github.com/Josh-The-Developapa/TumorVision.git
```

2. Navigate to the project directory:
```bash
cd TumorVision
```

3. Set up a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix or MacOS
venv\Scripts\activate      # On Windows
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

5. Follow additional setup instructions provided in the README files of TVG-1 and PolyTrend projects for any specific configurations or requirements.

<!-- remember to include an example use case -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [**Joshua Mukisa**](https://github.com/josh-the-developapa) - Developer of **TVG-1**
- [**Emmanuel Asiimwe**](https://github.com/asiimwemmanuel/) - Developer of **PolyTrend**
