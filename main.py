import os
import logging
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.svm_model import SVMModel
from src.models.random_forest_model import RandomForestModel
from src.visualization.visualizer import Visualizer
from src.utils.metrics import MetricsCalculator

class CreditDefaultAnalysis:
    
    def __init__(self, data_path=None, random_state=42):

        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.data_path = data_path or self.base_dir / 'dataset_source' / 'default_of_credit_card_clients.xls'
        self.output_dir = self.base_dir / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = DataLoader(self.data_path)
        self.preprocessor = DataPreprocessor(scaler_type='robust')
        self.models = {
            'SVM': SVMModel(search_method='random', n_iter=20),
            'Random Forest': RandomForestModel(search_method='grid')
        }
        self.visualizer = Visualizer()
        self.metrics_calculator = MetricsCalculator()
        self.random_state = random_state
        
    def setup_logging(self):

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'credit_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
    def run(self):

        try:
            self.data = self.load_and_preprocess_data()
            X_train, X_test, y_train, y_test = self.split_data(self.data)
            results, detailed_results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            self.generate_visualizations(results)
            self.save_results(results, detailed_results)
            
            return {
                'results': results,
                'detailed_results': detailed_results,
                'models': self.models,
                'data_summary': self.data_loader.get_data_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}", exc_info=True)
            raise
            
    def load_and_preprocess_data(self):

        self.logger.info("Loading data...")
        self.data = self.data_loader.load_data()

        self.logger.info(f"Data summary: {self.data_loader.get_data_summary()}")
        
        self.logger.info("Preprocessing data...")
        X, y = self.data_loader.split_features_target()
        X = self.preprocessor.preprocess(X, feature_engineering=True)
        
        return X, y
    
    def split_data(self, data):

        self.logger.info("Splitting data into train and test sets...")
        X, y = data

        return train_test_split(X, y, test_size=0.3, random_state=self.random_state)
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):

        results = {}
        detailed_results = {}
        
        for name, model in self.models.items():

            self.logger.info(f"Training {name}...")
            model.tune_and_train(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_prob(X_test) if name == 'Random Forest' else None
            
            results[name] = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_prob)
            detailed_results[name] = model.get_cv_results()
            
            self.logger.info(f"{name} best parameters: {detailed_results[name]['best_params']}")
            self.logger.info(f"{name} best score: {detailed_results[name]['best_score']:.4f}")
        
        return results, detailed_results
    
    def generate_visualizations(self, results):

        self.logger.info("Generating visualizations...")

        self.visualizer = Visualizer(output_dir=Path('outputs'))

        self.visualizer.plot_metrics_comparison(results)

        self.visualizer.plot_confusion_matrices(results)

        self.visualizer.plot_feature_importance(self.feature_importance_dict)

        self.visualizer.plot_feature_importance_with_std(self.mean_importance, self.std_importance)

        self.visualizer.plot_roc_curves(results)

        self.visualizer.plot_learning_curves(self.train_sizes, self.train_scores, self.test_scores,
                                             "Random Forest")
    
    def save_results(self, results, detailed_results):
        with open(self.output_dir / 'metrics_results.json', 'w') as f:
            json.dump(self.make_json_serializable(results), f, indent=4)
            
        with open(self.output_dir / 'cv_results.json', 'w') as f:
            json.dump(self.make_json_serializable(detailed_results), f, indent=4)
            
    @staticmethod
    def make_json_serializable(obj):

        if isinstance(obj, dict):
            return {k: CreditDefaultAnalysis.make_json_serializable(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [CreditDefaultAnalysis.make_json_serializable(v) for v in obj]
        
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        
        return obj

if __name__ == "__main__":

    analysis = CreditDefaultAnalysis()
    results = analysis.run()
    print("\nModel Performance Summary:")

    for model_name, metrics in results['results'].items():
        print(f"\n{model_name}:")

        for metric, value in metrics.items():

            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")