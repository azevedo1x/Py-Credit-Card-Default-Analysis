from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess(self, X):
        
        X = self._normalize_numeric_features(X)
        X = self._encode_categorical_features(X)
        return X
        
    def _normalize_numeric_features(self, X):
        numeric_features = ['LIMIT_BAL', 'AGE'] + \
                          [f'PAY_{i}' for i in range(1, 6)] + \
                          [f'BILL_AMT{i}' for i in range(1,7)] + \
                          [f'PAY_AMT{i}' for i in range(1,7)]
        
        X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        return X
        
    def _encode_categorical_features(self, X):
        categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
            self.label_encoders[feature] = le
            
        return X