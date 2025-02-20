# Credit Default Analysis Project

This project is designed to analyze and predict credit card default using machine learning models. It includes data loading, preprocessing, model training, evaluation, and visualization components. The project is structured to be modular and extensible, allowing for easy integration of new models and datasets.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Classes and Functionality](#classes-and-functionality)
6. [Results and Visualizations](#results-and-visualizations)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview

The goal of this project is to predict whether a credit card holder will default on their payment next month. The project uses a dataset containing various features such as credit limit, age, payment history, and bill amounts. The project implements two machine learning models: **Support Vector Machine (SVM)** and **Random Forest**, and provides a comprehensive pipeline for data loading, preprocessing, model training, evaluation, and visualization.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/azevedo1x/Py-Credit-Card-Default-Analysis.git
   cd Py-Credit-Card-Default-Analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   # Or install it manually, if you encountered some errors like i did :)


4. **Download the dataset:**
   - Place the dataset (`default_of_credit_card_clients.xls`) in the `dataset_source` directory.

## Usage

To run the analysis, execute the following command:

```bash
python src/main.py
```
Or run it through your preferred IDE/Text editor

This will:
- Load and preprocess the data.
- Train and evaluate the models.
- Generate visualizations and save the results in the `outputs` directory.

## Project Structure

The project is organized as follows:

```
credit-default-analysis/
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── random_forest_model.py
│   │   └── svm_model.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── utils/
│   │   ├── helpers.py
│   │   └── metrics.py
│   ├── visualization/
│   │   └── visualizer.py
│   └── main.py
├── dataset_source/
│   └── default_of_credit_card_clients.xls
├── outputs/
│   └── (generated files)
├── requirements.txt
```

## Classes and Functionality

### Data Loading
- **`DataLoader`**: Handles loading data from various file formats (CSV, Excel) and validates the dataset to ensure it contains the required columns.

### Preprocessing
- **`DataPreprocessor`**: Preprocesses the data by handling missing values, normalizing numeric features, and encoding categorical features. It also supports feature engineering.

### Models
- **`BaseModel`**: An abstract base class that defines the interface for model training, tuning, and evaluation.
- **`RandomForestModel`**: Implements a Random Forest classifier with hyperparameter tuning.
- **`SVMModel`**: Implements an SVM classifier with hyperparameter tuning.

### Metrics
- **`MetricsCalculator`**: Calculates various performance metrics such as accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.

### Visualization
- **`Visualizer`**: Generates visualizations such as ROC curves, confusion matrices, feature importance plots, and learning curves.

### Main Pipeline
- **`CreditDefaultAnalysis`**: The main class that orchestrates the entire pipeline, from data loading to model evaluation and visualization.

## Results and Visualizations

The project generates several visualizations and metrics, including:
- **Model Performance Comparison**: Bar charts comparing accuracy, precision, recall, and F1 score across models.
- **Confusion Matrices**: Heatmaps showing the confusion matrices for each model.
- **Feature Importance**: Plots showing the importance of features in the Random Forest model.
- **ROC Curves**: ROC curves for each model, showing the trade-off between true positive rate and false positive rate.
- **Learning Curves**: Learning curves showing the model's performance as a function of training set size.

All visualizations and metrics are saved in the `outputs` directory.

## Contributing

Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/yourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/yourFeature`).
5. Open a pull request.

For any questions or issues, please open an issue on the GitHub repository.