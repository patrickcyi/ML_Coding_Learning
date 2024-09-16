import numpy as np
from sklearn.metrics import roc_auc_score

class ClassificationMetrics:
    def __init__(self, y_true, y_pred, y_prob=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
    
    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)
    
    def confusion_matrix(self):
        unique_labels = np.unique(np.concatenate((self.y_true, self.y_pred)))
        cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for true, pred in zip(self.y_true, self.y_pred):
            cm[true, pred] += 1
        return cm
    
    def precision(self):
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fp = np.sum((self.y_pred == 1) & (self.y_true == 0))
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    
    def recall(self):
        tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        fn = np.sum((self.y_pred == 0) & (self.y_true == 1))
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    
    def f1_score(self):
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)
    
    def au_roc(self):
        if self.y_prob is None:
            raise ValueError("y_prob must be provided to calculate AU-ROC.")
        return roc_auc_score(self.y_true, self.y_prob)



# Example usage:
# Generating some synthetic data
np.random.seed(0)
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
y_prob = np.random.rand(100)

# Initialize ClassificationMetrics with the true and predicted labels
metrics = ClassificationMetrics(y_true, y_pred, y_prob)

# Compute metrics
print("Accuracy:", metrics.accuracy())
print("Confusion Matrix:\n", metrics.confusion_matrix())
print("Precision:", metrics.precision())
print("Recall:", metrics.recall())
print("F1 Score:", metrics.f1_score())
print("AU-ROC:", metrics.au_roc())


