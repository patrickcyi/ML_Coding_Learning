import numpy as np

def calculate_auc_probability(y_true, y_scores):
    # Separate positive and negative samples
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    # Initialize counter
    count = 0
    total_pairs = len(pos_scores) * len(neg_scores)
    
    # Compare each positive score with each negative score
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                count += 1  # Positive example is ranked higher than negative
            elif pos == neg:
                count += 0.5  # Positive and negative are ranked equally

    # AUC is the fraction of correctly ranked pairs
    auc = count / total_pairs if total_pairs > 0 else 0
    return auc

# Example usage:
y_true = np.array([0


ROC : TPRate/FPrate 
AUC: the probability that the model ranks a random positive example more highly than a random negative example. 

percision = tp/tp+fp not affect by tn, so not affect by imbalance 

PR: too get better recall, percision can varyt. so not 单调




import numpy as np
def compute_roc(y_true, y_scores):
    # Sort the scores and corresponding true labels
    sorted_indices = np.argsort(y_scores)[::-1]  # Sort in descending order
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]

    # Total positive and negative samples
    P = np.sum(y_true_sorted == 1)
    N = np.sum(y_true_sorted == 0)

    # Compute TPR (recall) and FPR at each threshold
    TPR = np.cumsum(y_true_sorted == 1) / P  # True Positive Rate
    FPR = np.cumsum(y_true_sorted == 0) / N  # False Positive Rate

    # Add (0, 0) point
    TPR = np.concatenate(([0], TPR))
    FPR = np.concatenate(([0], FPR))

    return FPR, TPR

def compute_auc(fpr, tpr):
    # Use the trapezoidal rule to compute the area under the curve
    return np.trapz(tpr, fpr)

# Example usage
y_true = np.array([0, 0, 1, 1])  # True labels
y_scores = np.array([0.1, 0.4, 0.35, 0.8])  # Predicted probabilities

# Compute ROC curve
fpr, tpr = compute_roc(y_true, y_scores)

# Compute AUC
auc = compute_auc(fpr, tpr)
print(f"FPR: {fpr}")
print(f"TPR: {tpr}")
print(f"AUC: {auc}")
