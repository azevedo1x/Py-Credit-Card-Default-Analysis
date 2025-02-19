from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from typing import Dict, Any
import numpy as np

class RandomForestModel(BaseModel):
    
    def build(self) -> None:
        self.model = RandomForestClassifier(random_state=42)
        
    def get_params_grid(self) -> Dict[str, Any]:

        if self.search_method == 'random':
            return {
                'n_estimators': np.arange(100, 1000),
                'max_depth': [None] + list(np.arange(5, 50)),
                'min_samples_split': np.arange(2, 20),
                'min_samples_leaf': np.arange(1, 10),
                'max_features': ['auto', 'sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'criterion': ['gini', 'entropy']
            }
        else:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt'],
                'bootstrap': [True, False],
                'class_weight': ['balanced', None],
                'criterion': ['gini', 'entropy']
            }
            
    def get_feature_importance(self, feature_names=None):

        if self.search is None:
            raise ValueError("Model not trained")
            
        importance = self.search.best_estimator_.feature_importances_
            
        return dict(zip(feature_names, importance))
        
    def get_trees_feature_importance(self, feature_names=None):

        if self.search is None:
            raise ValueError("Model not trained")
            
        trees = self.search.best_estimator_.estimators_
        all_importances = np.array([tree.feature_importances_ for tree in trees])
        
        mean_importance = all_importances.mean(axis=0)
        std_importance = all_importances.std(axis=0)
              
        return {
            'mean_importance': dict(zip(feature_names, mean_importance)),
            'std_importance': dict(zip(feature_names, std_importance))
        }