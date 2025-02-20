import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path

class Visualizer:
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.style_config = {
            'figsize': (12, 8),
            'palette': 'viridis',
            'font_scale': 1.2
        }
        self._set_style()
        
    def _set_style(self):

        sns.set_theme(style="whitegrid")
        sns.set_context("notebook", font_scale=self.style_config['font_scale'])
        plt.rcParams['figure.figsize'] = self.style_config['figsize']
        
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, float]], save: bool = True):

        try:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(metrics))
            width = 0.35
            n_models = len(results)
            
            for i, (model_name, metrics_dict) in enumerate(results.items()):
                metrics_values = [metrics_dict.get(metric, 0) for metric in metrics]
                offset = width * (i - (n_models-1)/2)
                plt.bar(x + offset, metrics_values, width, label=model_name,
                       alpha=0.8)

                for j, value in enumerate(metrics_values):
                    plt.text(x[j] + offset, value, f'{value:.3f}',
                            ha='center', va='bottom')
            
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, metrics)
            plt.legend(loc='upper right')
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / 'metrics_comparison.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics comparison: {str(e)}")
            raise
            
    def plot_confusion_matrices(self, results: Dict[str, Dict[str, np.ndarray]], save: bool = True):

        try:
            n_models = len(results)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, metrics_dict) in enumerate(results.items()):
                cm = metrics_dict['confusion_matrix']
                
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
                sns.heatmap(cm_percent, annot=True, fmt='.1f%', cmap='Blues', 
                           alpha=0.2, cbar=True, ax=axes[i])
                
                axes[i].set_title(f'Confusion Matrix - {model_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
                
            plt.tight_layout()
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / 'confusion_matrices.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrices: {str(e)}")
            raise
            
    def plot_feature_importance(self, feature_importance_dict: Dict[str, float], 
                              save: bool = True):

        try:
            feature_importance = pd.DataFrame({
                'feature': feature_importance_dict.keys(),
                'importance': feature_importance_dict.values()
            })
            
            feature_importance = feature_importance.sort_values('importance', 
                                                             ascending=True).tail(15)
            
            plt.figure(figsize=(10, 8))
            bars = plt.barh(feature_importance['feature'], 
                          feature_importance['importance'])
            
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center')
            
            plt.title('Top 15 Most Important Features')
            plt.xlabel('Importance Score')
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / 'feature_importance.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise
            
    def plot_feature_importance_with_std(self, mean_importance: Dict[str, float],
                                       std_importance: Dict[str, float],
                                       save: bool = True):
        
        try:

            df = pd.DataFrame({
                'feature': mean_importance.keys(),
                'importance': mean_importance.values(),
                'std': std_importance.values()
            })
            
            df = df.sort_values('importance', ascending=True).tail(15)
            
            plt.figure(figsize=(10, 8))

            plt.barh(df['feature'], df['importance'], 
                    xerr=df['std'], capsize=5, alpha=0.6)
            
            plt.title('Feature Importance with Standard Deviation')
            plt.xlabel('Importance Score')
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / 'feature_importance_with_std.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance with std: {str(e)}")
            raise
            
    def plot_roc_curves(self, results: Dict[str, Dict[str, Any]], save: bool = True):

        try:
            plt.figure(figsize=(8, 8))
            
            for model_name, metrics_dict in results.items():
                if 'roc_curve' in metrics_dict:
                    fpr = metrics_dict['roc_curve']['fpr']
                    tpr = metrics_dict['roc_curve']['tpr']
                    auc = metrics_dict['roc_auc']
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
                    
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend(loc="lower right")
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / 'roc_curves.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curves: {str(e)}")
            raise
            
    def plot_learning_curves(self, train_sizes: np.ndarray, 
                           train_scores: np.ndarray,
                           test_scores: np.ndarray,
                           model_name: str,
                           save: bool = True):

        try:
            plt.figure(figsize=(10, 6))
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, label='Training score')
            plt.plot(train_sizes, test_mean, label='Cross-validation score')
            
            plt.fill_between(train_sizes, train_mean - train_std,
                           train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std,
                           test_mean + test_std, alpha=0.1)
            
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.title(f'Learning Curves for {model_name}')
            plt.legend(loc='best')
            plt.grid(True)
            
            if save and self.output_dir:
                plt.savefig(self.output_dir / f'learning_curve_{model_name}.png')
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting learning curves: {str(e)}")
            raise