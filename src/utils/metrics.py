from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }