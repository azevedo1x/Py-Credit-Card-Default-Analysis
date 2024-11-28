from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def build(self):
        self.model = RandomForestClassifier()
        
    def get_params_grid(self):
        return {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    
    def get_feature_importance(self, feature_names):
        
        if self.grid_search is None:
            raise ValueError("Modelo não treinado")
            
        return dict(zip(feature_names, 
                       self.grid_search.best_estimator_.feature_importances_))
