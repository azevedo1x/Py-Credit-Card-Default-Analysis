import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional
import logging

class DataPreprocessor:
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, x: pd.DataFrame, feature_engineering: bool = False) -> pd.DataFrame:

        try:
            x = x.copy()
            x = self._handle_missing_values(x)
            x = self._normalize_numeric_features(x)
            x = self._encode_categorical_features(x)
            
            if feature_engineering:
                x = self._engineer_features(x)
                
            return x
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def _handle_missing_values(self, x: pd.DataFrame) -> pd.DataFrame:

        numeric_features = x.select_dtypes(include=[np.number]).columns
        x[numeric_features] = self.imputer.fit_transform(x[numeric_features])
        
        categorical_features = x.select_dtypes(exclude=[np.number]).columns
        for feature in categorical_features:
            x[feature] = x[feature].fillna(x[feature].mode()[0])
            
        return x
        
    def _normalize_numeric_features(self, x: pd.DataFrame) -> pd.DataFrame:

        numeric_features = ['LIMIT_BAL', 'AGE'] + \
                          [f'PAY_{i}' for i in range(1, 7)] + \
                          [f'BILL_AMT{i}' for i in range(1, 7)] + \
                          [f'PAY_AMT{i}' for i in range(1, 7)]
                          

        for feature in numeric_features:
            if feature in x.columns:
                Q1 = x[feature].quantile(0.25)
                Q3 = x[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                x[feature] = x[feature].clip(lower_bound, upper_bound)
                
        x[numeric_features] = self.scaler.fit_transform(x[numeric_features])
        return x
        
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:

        categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
        
        for feature in categorical_features:
            if feature in X.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])
                self.label_encoders[feature] = le
                
                if len(le.classes_) > 2:
                    one_hot = pd.get_dummies(X[feature], prefix=feature)
                    X = pd.concat([X, one_hot], axis=1)
                    X.drop(feature, axis=1, inplace=True)
                    
        return X
        
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:

        for i in range(1, 7):
            bill_col = f'BILL_AMT{i}'
            pay_col = f'PAY_AMT{i}'
            if bill_col in X.columns and pay_col in X.columns:
                X[f'PAYMENT_RATIO_{i}'] = X[pay_col] / (X[bill_col] + 1e-8)
                
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
        X['AVG_BILL_AMT'] = X[bill_cols].mean(axis=1)
        
        for i in range(1, 6):
            X[f'BILL_TREND_{i}'] = X[f'BILL_AMT{i}'] - X[f'BILL_AMT{i+1}']
            
        X['UTILIZATION_RATE'] = X['BILL_AMT1'] / (X['LIMIT_BAL'] + 1e-8)
        
        return X
