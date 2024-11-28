from sklearn.svm import SVC
from .base_model import BaseModel

class SVMModel(BaseModel):
    def build(self):
        self.model = SVC()
        
    def get_params_grid(self):
        return {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
