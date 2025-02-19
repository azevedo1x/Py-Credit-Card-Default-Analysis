from sklearn.svm import SVC
from .base_model import BaseModel
from typing import Dict, Any
import numpy as np

class SVMModel(BaseModel):
    
    def build(self) -> None:
        self.model = SVC(probability=True, random_state=42)
        
    def get_params_grid(self) -> Dict[str, Any]:

        if self.search_method == 'random':
            return {
                'C': np.logspace(-3, 3, 100),
                'gamma': np.logspace(-3, 3, 100),
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': [2, 3, 4],  # for poly kernel
                'class_weight': ['balanced', None],
                'tol': np.logspace(-4, -1, 100)
            }
        else:
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear'],
                'class_weight': ['balanced', None],
                'tol': [1e-3, 1e-4]
            }
            
    def get_feature_importance(self, X, feature_names=None):

        if self.search is None:
            raise ValueError("Model not trained")
            
        best_svm = self.search.best_estimator_
        
        if best_svm.kernel == 'linear':
            importance = np.abs(best_svm.coef_[0])
        else:
            importance = np.abs(best_svm.decision_function(X)).mean(axis=0)
            
        return dict(zip(feature_names, importance))