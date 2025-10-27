from sklearn import metrics
import numpy as np


def auc_roc(y_true, y_score):
    """
    Calculates the area under the Receiver Operating Characteristic (ROC) curve.

    Args:
    
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.

    Returns:
    
        float: 
            The score of the area under the ROC curve.
    """

    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    """
    Calculates the area under the Precision-Recall (PR) curve.

    Args:
    
        y_true (np.array, required): 
            True binary labels. 0 indicates a normal timestamp, and 1 indicates an anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores. A higher score indicates a higher likelihood of being an anomaly.

    Returns:
    
        float: 
            The score of the area under the PR curve.
    """

    return metrics.average_precision_score(y_true, y_score)


def tabular_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for tabular anomaly detection.

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:
        
        - auc_roc (float):
            The score of area under the ROC curve.
            
        - auc_pr (float):
            The score of area under the precision-recall curve.
            
        - f1 (float): 
            The score of F1-score.
    """

    # F1@k, using real percentage to calculate F1-score
    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), f1





