from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.grid_search = None
        
    @abstractmethod
    def build(self):
        
        pass
        
    @abstractmethod
    def get_params_grid(self):
        
        pass
        
    def train(self, X_train, y_train):
        
        self.build()
        params_grid = self.get_params_grid()
        self.grid_search = GridSearchCV(self.model, params_grid, cv=3, n_jobs=-1)
        self.grid_search.fit(X_train, y_train)
        
    def predict(self, X):
        
        if self.grid_search is None:
            raise ValueError("Modelo não treinado")
        return self.grid_search.predict(X)