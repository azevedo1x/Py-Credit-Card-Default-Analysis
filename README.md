# Credit Default Analysis Project

## Project Description
A credit risk analysis project using machine learning techniques to predict credit card default probability.

## Project Structure
```
credit_default_analysis/
├── dataset_source/
│   └── default_of_credit_card_clients.xls
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── preprocessing/
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── svm_model.py
│   │   └── random_forest_model.py
│   ├── visualization/
│   │   └── visualizer.py
│   └── utils/
│       └── metrics.py
├── main.py
└── requirements.txt
```

## Requirements
- Python 3.8+
- Libraries:
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - openpyxl

## Installation

1. Clone the repository:
```bash
git clone https://github.com/azevedo1x/Py-Credit-Card-Default-Analysis.git
cd Py-Credit-Card-Default-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Features
- Excel file data loading
- Data preprocessing
- Model training:
  - Support Vector Machine (SVM)
  - Random Forest
- Performance metrics generation
- Visualizations:
  - Metrics comparison
  - Confusion matrices
  - Feature importance analysis

## Methodology
1. Data loading
2. Preprocessing
3. Train-test data split
4. Model training
5. Performance evaluation
6. Result visualization

## Contributing
1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Contact
@azevedo1x - gabriel.azeve04@gmail.com

## References
- Dataset: UCI Machine Learning Repository
- Scikit-learn Documentation
- Pandas Documentation
