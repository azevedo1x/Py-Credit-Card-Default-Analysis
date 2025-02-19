from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Dict, Any
import logging

class BaseModel(ABC):
    
    def __init__(self, search_method: str = 'grid', n_iter: int = 10):
        self.model = None
        self.search = None
        self.search_method = search_method
        self.n_iter = n_iter
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def build(self) -> None:
        pass
        
    @abstractmethod
    def get_params_grid(self) -> Dict[str, Any]:
        pass
        
    def tune_and_train(self, X_train, y_train, custom_scorer: Any = None) -> None:

        try:
            self.build()
            hyperparams_grid = self.get_params_grid()
            
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(f1_score)
            }
            
            if custom_scorer:
                scoring['custom'] = make_scorer(custom_scorer)
                
            if self.search_method == 'random':
                self.search = RandomizedSearchCV(
                    self.model,
                    hyperparams_grid,
                    n_iter=self.n_iter,
                    cv=5,
                    n_jobs=-1,
                    scoring=scoring,
                    refit='f1'
                )
            else:
                self.search = GridSearchCV(
                    self.model,
                    hyperparams_grid,
                    cv=5,
                    n_jobs=-1,
                    scoring=scoring,
                    refit='f1'
                )
                
            self.search.fit(X_train, y_train)
            self.logger.info(f"Best parameters: {self.search.best_params_}")
            
        except Exception as e:
            self.logger.error(f"Error in tuning or training: {str(e)}")
            raise
            
    def predict(self, X):

        if self.search is None:
            raise ValueError("Model not trained")
        
        return self.search.predict(X)
        
    def predict_prob(self, X):

        if self.search is None:
            raise ValueError("Model not trained")
        
        if hasattr(self.search.best_estimator_, 'predict_proba'):
            return self.search.best_estimator_.predict_proba(X)
        else:
            raise NotImplementedError("Probability prediction not supported")
            
    def get_cv_results(self) -> Dict[str, Any]:

        if self.search is None:
            raise ValueError("Model not trained")
            
        return {
            'best_score': self.search.best_score_,
            'best_params': self.search.best_params_,
            'cv_results': self.search.cv_results_
        }