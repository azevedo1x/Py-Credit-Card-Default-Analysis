import pandas as pd

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        
    def load_data(self, data=None):
        
        if data is not None:
            self.data = pd.DataFrame(data)
        elif self.data_path:
            if self.data_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(self.data_path)
            else:
                self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Nenhuma fonte de dados fornecida")
        
        return self.data
    
    def split_features_target(self):
        
        if self.data is None:
            raise ValueError("Dados não carregados")
            
        X = self.data.drop(['ID', 'default payment next month'], axis=1)
        y = self.data['default payment next month']
        
        return X, y