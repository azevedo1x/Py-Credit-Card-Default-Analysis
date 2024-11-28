import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_metrics_comparison(results):
        metrics = ['accuracy', 'precision', 'recall']
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (model_name, metrics_dict) in enumerate(results.items()):
            metrics_values = [metrics_dict[metric] for metric in metrics]
            plt.bar(x + i*width, metrics_values, width, label=model_name)
        
        plt.xlabel('Métricas')
        plt.ylabel('Pontuação')
        plt.title('Comparação de Performance dos Modelos')
        plt.xticks(x + width/2, metrics)
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrices(results):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for i, (model_name, metrics_dict) in enumerate(results.items()):
            sns.heatmap(metrics_dict['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       cmap='Blues',
                       ax=axes[i])
            axes[i].set_title(f'Matriz de Confusão - {model_name}')
            axes[i].set_xlabel('Previsto')
            axes[i].set_ylabel('Real')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance_dict):
        feature_importance = pd.DataFrame({
            'feature': feature_importance_dict.keys(),
            'importance': feature_importance_dict.values()
        })
        
        feature_importance = feature_importance.sort_values('importance', 
                                                          ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Top 10 Features Mais Importantes')
        plt.xlabel('Importância')
        plt.ylabel('Feature')
        plt.show()