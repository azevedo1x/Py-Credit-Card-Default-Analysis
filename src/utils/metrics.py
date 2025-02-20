from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import numpy as np
from typing import Dict, Any
import logging

class MetricsCalculator:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray = None) -> Dict[str, Any]:
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
            
            if y_prob is not None:

                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                
                precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
                metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
                metrics['pr_curve'] = {'precision': precision, 'recall': recall}
                
            metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(y_true, y_pred)
            metrics['f1_score'] = self._calculate_f1_score(metrics['precision'], metrics['recall'])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return (sensitivity + specificity) / 2