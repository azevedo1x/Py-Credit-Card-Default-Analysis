import os
from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.svm_model import SVMModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.visualizer import Visualizer
from src.utils.metrics import MetricsCalculator
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dataset_source', 'default_of_credit_card_clients.xls')

class CreditDefaultAnalysis:
    def __init__(self):

        self.data_loader = DataLoader(DATA_PATH)
        self.preprocessor = DataPreprocessor()
        self.svm_model = SVMModel()
        self.rf_model = RandomForestModel()
        self.visualizer = Visualizer()
        self.metrics_calculator = MetricsCalculator()
        
    def run(self):

        print("1. Carregando dados...")
        data = self.data_loader.load_data()
        X, y = self.data_loader.split_features_target()
        
        print("2. Pré-processando dados...")
        X = self.preprocessor.preprocess(X)
        
        print("3. Dividindo dados em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print("4. Treinando modelos...")
        models = {
            'SVM': self.svm_model,
            'Random Forest': self.rf_model
        }
        
        results = {}
        for name, model in models.items():
            print(f"Treinando {name}...")
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = self.metrics_calculator.calculate_metrics(y_test, y_pred)
            
        print("5. Gerando visualizações...")
        self.visualizer.plot_metrics_comparison(results)
        self.visualizer.plot_confusion_matrices(results)
        
        feature_importance = self.rf_model.get_feature_importance(X.columns)
        self.visualizer.plot_feature_importance(feature_importance)
        
        return results, models

if __name__ == "__main__":
    analysis = CreditDefaultAnalysis()
    results, models = analysis.run()