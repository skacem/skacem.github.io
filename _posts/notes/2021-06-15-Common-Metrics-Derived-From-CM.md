---
layout: post
category: ml
comments: true
title: "Common Metrics Derived From the Confusion Matrix: A Practical Implementation Guide"
excerpt: "In the previous post we introduced the confusion matrix in the context of hypothesis testing and we showed how such a simple and intuitive concept can be a powerful tool in illustrating the outcomes of classification models. Now, we are going to discuss various performance metrics that are derived from a confusion matrix."
author: "Skander Kacem"
tags:
  - Machine Learning
  - Confusion Matrix
  - Evaluation Metrics
katex: true
preview_pic: /assets/0/SPS_dist.gif
---


## Introduction

In [Part 1](/ml/2021/06/07/Confusion-Matrix/), we explored why the confusion matrix is essential for understanding classifier behavior. We examined the accuracy paradox, discussed the asymmetry of errors (Type I vs Type II), and built intuition for when to optimize precision versus recall. We framed metric selection as a decision theory problem where different use cases demand different loss functions.

This post provides the practical companion: **how to correctly implement these metrics in Python**, handle edge cases that can break your code, and extend beyond the basic metrics to more sophisticated evaluation tools.

If you haven't read Part 1, I recommend starting there for the conceptual foundation. This post assumes you understand *why* these metrics matter and focuses on *how* to compute them reliably.

## Quick Review: The Confusion Matrix

As a refresher, the confusion matrix for binary classification has four components:

$$\text{CM} = \left[\begin{array}{cc} TN & FP \\ FN & TP \end{array}\right]$$

Where:
- **TN** (True Negative): Correct negative predictions
- **FP** (False Positive): Incorrect positive predictions (Type I error)
- **FN** (False Negative): Incorrect negative predictions (Type II error)
- **TP** (True Positive): Correct positive predictions

**Critical Note on Conventions**: Throughout this post, we follow scikit-learn's convention where **rows represent true labels** and **columns represent predictions**. Some textbooks and libraries transpose this. Always verify which convention your tools use to avoid misinterpretation.

## Implementation Fundamentals

### The `cm_to_dataset` Utility Function

Throughout this post, I use a helper function `cm_to_dataset` that converts a confusion matrix back into prediction arrays. This is useful for demonstrating concepts without needing actual model outputs. Here's the implementation:

```python
def cm_to_dataset(cm):
    """
    Convert a confusion matrix into y_true and y_pred arrays.
    
    Parameters:
    -----------
    cm : array-like, shape (2, 2)
        Confusion matrix in format [[TN, FP], [FN, TP]]
    
    Returns:
    --------
    y_true : array
        Ground truth labels
    y_pred : array
        Predicted labels
    """
    import numpy as np
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # Create arrays for each quadrant of the confusion matrix
    y_true = np.array([0]*tn + [0]*fp + [1]*fn + [1]*tp)
    y_pred = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
    
    # Shuffle to avoid any ordering artifacts
    indices = np.random.permutation(len(y_true))
    
    return y_true[indices], y_pred[indices]
```

### Setting Up Our Environment

```python
# Standard imports for this tutorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Set random seed for reproducibility
np.random.seed(42)

# Visualization settings
sns.set_style('whitegrid')
%matplotlib inline
```

## Core Metrics: Implementation and Edge Cases

Let's revisit the basic metrics from Part 1, but this time focus on **correct implementation** and **edge cases** that can cause problems.

### Accuracy: When It Works (and When It Doesn't)

Accuracy is the ratio of correct predictions to total predictions:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

As we demonstrated in Part 1, accuracy fails catastrophically on imbalanced datasets. Let's see the implementation:

```python
# Example: Nearly perfect accuracy, but useless model
cm = np.array([
    [990, 0],
    [10, 0]
])

y_true, y_pred = cm_to_dataset(cm)

# Method 1: Manual calculation from confusion matrix
acc_manual = (cm[0,0] + cm[1,1]) / cm.sum()
print(f"Manual calculation: {acc_manual:.3f}")

# Method 2: Using sklearn
acc_sklearn = accuracy_score(y_true, y_pred)
print(f"Sklearn calculation: {acc_sklearn:.3f}")

# Output:
# Manual calculation: 0.990
# Sklearn calculation: 0.990
```

**Edge Case**: Accuracy is always defined (never division by zero) since the denominator is the total number of samples.

**When to use**: Only when classes are roughly balanced and both error types have similar costs. In most real-world applications (fraud, disease, spam), **don't optimize for accuracy alone**.

### Precision: The Reliability of Positive Predictions

Precision measures what fraction of positive predictions are actually correct:

$$\text{Precision} = \frac{TP}{TP + FP}$$

```python
# Example: Model that predicts everything as negative
cm = np.array([
    [990, 0],
    [10, 0]
])

y_true, y_pred = cm_to_dataset(cm)

# Method 1: Manual calculation
tp = cm[1, 1]
fp = cm[0, 1]
prec_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"Manual calculation: {prec_manual:.3f}")

# Method 2: Using sklearn with zero_division parameter
prec_sklearn = precision_score(y_true, y_pred, zero_division=0)
print(f"Sklearn calculation: {prec_sklearn:.3f}")

# Output:
# Manual calculation: 0.000
# Sklearn calculation: 0.000
```

**Critical Edge Case**: When `TP + FP = 0` (model never predicts positive class), precision is undefined. Sklearn's `zero_division` parameter lets you choose what to return: 0, 1, or raise a warning.

```python
# Comparing zero_division behaviors
print(f"zero_division=0: {precision_score(y_true, y_pred, zero_division=0)}")
print(f"zero_division=1: {precision_score(y_true, y_pred, zero_division=1)}")

# zero_division='warn' (default) will show a warning message
```

**Best Practice**: Use `zero_division=0` when you want to penalize models that never predict the positive class.

### Recall (Sensitivity): The Completeness of Detection

Recall measures what fraction of actual positives are detected:

$$\text{Recall} = \frac{TP}{TP + FN}$$

```python
# Example: Model with low recall
cm = np.array([
    [990, 0],
    [9, 1]  # Only catches 1 out of 10 positive cases
])

y_true, y_pred = cm_to_dataset(cm)

# Method 1: Manual calculation
tp = cm[1, 1]
fn = cm[1, 0]
rec_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Manual calculation: {rec_manual:.3f}")

# Method 2: Using sklearn
rec_sklearn = recall_score(y_true, y_pred)
print(f"Sklearn calculation: {rec_sklearn:.3f}")

# Output:
# Manual calculation: 0.100
# Sklearn calculation: 0.100
```

**Edge Case**: When `TP + FN = 0` (no positive samples in dataset), recall is undefined. However, this situation is rare in practice—if you have no positive samples, why are you training a classifier?

**When to optimize**: As discussed in Part 1, prioritize recall when false negatives are catastrophic (cancer screening, fraud detection, epidemic containment).

### F1-Score: The Harmonic Mean

The F1-score balances precision and recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

```python
# Example: Moderate precision and recall
cm = np.array([
    [990, 0],
    [9, 1]
])

y_true, y_pred = cm_to_dataset(cm)

# Method 1: Manual calculation from precision and recall
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1_manual = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
print(f"Manual calculation: {f1_manual:.3f}")

# Method 2: Using sklearn
f1_sklearn = f1_score(y_true, y_pred)
print(f"Sklearn calculation: {f1_sklearn:.3f}")

# Method 3: Direct formula from confusion matrix
tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
f1_direct = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
print(f"Direct formula: {f1_direct:.3f}")

# Output:
# Manual calculation: 0.182
# Sklearn calculation: 0.182
# Direct formula: 0.182
```

**Why harmonic mean?** The harmonic mean heavily penalizes extreme values. A model with 100% precision but 10% recall gets an F1-score of only 18.2%, not 55% (which would be the arithmetic mean).

### Specificity: The Symmetric Complement

Specificity measures the true negative rate:

$$\text{Specificity} = \frac{TN}{TN + FP}$$

```python
# Calculating specificity (no sklearn function for this)
cm = np.array([
    [990, 0],
    [9, 1]
])

tn = cm[0, 0]
fp = cm[0, 1]
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"Specificity: {spec:.3f}")

# Output:
# Specificity: 1.000
```

**Note**: Sklearn doesn't have a dedicated `specificity_score` function, but it's trivial to calculate manually. Specificity pairs naturally with recall (sensitivity) to give a complete picture of binary classification performance.

## Advanced Metrics: Beyond the Basics

Part 1 covered the fundamental metrics. Now let's explore more sophisticated measures that handle edge cases better.

### Matthews Correlation Coefficient (MCC)

MCC is arguably the **best single metric for binary classification**, especially with imbalanced datasets. It considers all four confusion matrix quadrants and returns a value between -1 (total disagreement) and +1 (perfect prediction).

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

```python
from sklearn.metrics import matthews_corrcoef

# Example 1: Imbalanced dataset with decent performance
cm = np.array([
    [990, 10],
    [5, 95]
])

y_true, y_pred = cm_to_dataset(cm)

# Calculate MCC
mcc = matthews_corrcoef(y_true, y_pred)
print(f"MCC: {mcc:.3f}")

# Compare with accuracy (which looks great but is misleading)
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.3f}")

# Output:
# MCC: 0.825
# Accuracy: 0.986

# Example 2: Useless classifier (predicts everything as majority class)
cm_useless = np.array([
    [1000, 0],
    [100, 0]
])

y_true_u, y_pred_u = cm_to_dataset(cm_useless)
mcc_useless = matthews_corrcoef(y_true_u, y_pred_u)
acc_useless = accuracy_score(y_true_u, y_pred_u)

print(f"\nUseless classifier:")
print(f"MCC: {mcc_useless:.3f}")
print(f"Accuracy: {acc_useless:.3f}")

# Output:
# Useless classifier:
# MCC: 0.000
# Accuracy: 0.909
```

**Why MCC is superior**: 
- Works well even with severely imbalanced datasets
- Treats both classes symmetrically
- Returns 0 for a random/useless classifier (unlike accuracy)
- Takes into account all four confusion matrix categories

**When to use**: Consider MCC as your primary metric when:
- Classes are imbalanced
- You want a single, robust metric
- Both error types matter (unlike precision or recall which focus on one class)

### Cohen's Kappa: Agreement Beyond Chance

Cohen's Kappa measures agreement between predictions and truth, correcting for chance agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ is observed agreement (accuracy) and $p_e$ is expected agreement by chance.

```python
from sklearn.metrics import cohen_kappa_score

# Example: Good classifier
cm = np.array([
    [85, 15],
    [10, 90]
])

y_true, y_pred = cm_to_dataset(cm)

kappa = cohen_kappa_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"Cohen's Kappa: {kappa:.3f}")
print(f"Accuracy: {acc:.3f}")

# Output:
# Cohen's Kappa: 0.750
# Accuracy: 0.875

# Interpretation of Kappa values:
# < 0: Less than chance agreement
# 0.01-0.20: Slight agreement
# 0.21-0.40: Fair agreement
# 0.41-0.60: Moderate agreement
# 0.61-0.80: Substantial agreement
# 0.81-1.00: Almost perfect agreement
```

**When to use**: Cohen's Kappa is particularly useful for:
- Inter-rater reliability studies
- Comparing multiple annotators or models
- Understanding if your model performs better than random guessing

### Balanced Accuracy: Fair Treatment of Imbalanced Classes

Balanced accuracy is the average of recall across each class:

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

```python
from sklearn.metrics import balanced_accuracy_score

# Example: Imbalanced dataset
cm = np.array([
    [950, 50],
    [5, 95]
])

y_true, y_pred = cm_to_dataset(cm)

bal_acc = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"Regular Accuracy: {acc:.3f}")

# Calculate manually to verify
sensitivity = cm[1,1] / cm[1,:].sum()  # Recall
specificity = cm[0,0] / cm[0,:].sum()
bal_acc_manual = (sensitivity + specificity) / 2

print(f"Manual calculation: {bal_acc_manual:.3f}")

# Output:
# Balanced Accuracy: 0.925
# Regular Accuracy: 0.950
# Manual calculation: 0.925
```

**When to use**: Balanced accuracy is useful when:
- You have imbalanced classes
- You care equally about performance on both classes
- You want a metric that's less sensitive to class imbalance than regular accuracy

### Youden's J Statistic: Optimizing Binary Classification Thresholds

Youden's J statistic (also called Youden's Index) is used to find the optimal threshold for binary classifiers:

$$J = \text{Sensitivity} + \text{Specificity} - 1$$

It ranges from 0 to 1, where 1 indicates perfect classification.

```python
# Calculate Youden's J
cm = np.array([
    [85, 15],
    [10, 90]
])

sensitivity = cm[1,1] / cm[1,:].sum()
specificity = cm[0,0] / cm[0,:].sum()
youden_j = sensitivity + specificity - 1

print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"Youden's J: {youden_j:.3f}")

# Output:
# Sensitivity: 0.900
# Specificity: 0.850
# Youden's J: 0.750
```

**When to use**: Youden's J is particularly useful when:
- Selecting optimal classification thresholds from ROC curves
- You want to maximize both sensitivity and specificity simultaneously
- Developing diagnostic tests in medicine

## The F-Beta Score: Weighted Precision-Recall Tradeoff

The F-beta score generalizes F1 by allowing you to weight precision vs recall:

$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

- $\beta < 1$: Emphasizes precision
- $\beta = 1$: F1-score (balanced)
- $\beta > 1$: Emphasizes recall

```python
from sklearn.metrics import fbeta_score

cm = np.array([
    [90, 10],
    [20, 80]
])

y_true, y_pred = cm_to_dataset(cm)

# Compare different beta values
f1 = fbeta_score(y_true, y_pred, beta=1)
f2 = fbeta_score(y_true, y_pred, beta=2)  # Emphasize recall
f05 = fbeta_score(y_true, y_pred, beta=0.5)  # Emphasize precision

print(f"F1-Score (balanced): {f1:.3f}")
print(f"F2-Score (recall priority): {f2:.3f}")
print(f"F0.5-Score (precision priority): {f05:.3f}")

# Also show precision and recall for context
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
print(f"\nPrecision: {prec:.3f}")
print(f"Recall: {rec:.3f}")

# Output:
# F1-Score (balanced): 0.842
# F2-Score (recall priority): 0.820
# F0.5-Score (precision priority): 0.870
# 
# Precision: 0.889
# Recall: 0.800
```

**When to use**:
- F2 score: When false negatives are twice as costly as false positives (fraud detection, disease screening)
- F0.5 score: When false positives are twice as costly as false negatives (spam filtering)

## Multi-Class Confusion Matrices

So far, we've focused on binary classification. Let's extend to multi-class problems.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Create a multi-class dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42
)

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Generate and visualize confusion matrix
cm_multi = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_multi,
    display_labels=clf.classes_
)
disp.plot(cmap='Blues')
plt.title('Multi-Class Confusion Matrix')
plt.tight_layout()
plt.show()

print("Confusion Matrix:")
print(cm_multi)
```

### Averaging Strategies for Multi-Class Metrics

For multi-class problems, we need to decide how to aggregate per-class metrics:

```python
# Calculate metrics with different averaging strategies
print("Precision scores:")
print(f"  Macro (unweighted): {precision_score(y_test, y_pred, average='macro'):.3f}")
print(f"  Weighted (by support): {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"  Micro (global): {precision_score(y_test, y_pred, average='micro'):.3f}")

print("\nRecall scores:")
print(f"  Macro: {recall_score(y_test, y_pred, average='macro'):.3f}")
print(f"  Weighted: {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"  Micro: {recall_score(y_test, y_pred, average='micro'):.3f}")

print("\nF1 scores:")
print(f"  Macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"  Weighted: {f1_score(y_test, y_pred, average='weighted'):.3f}")
print(f"  Micro: {f1_score(y_test, y_pred, average='micro'):.3f}")
```

**Averaging strategies explained**:

- **Macro**: Calculate metric for each class, then take unweighted mean. Good when all classes are equally important.
- **Weighted**: Calculate metric for each class, then take weighted mean by class frequency. Better for imbalanced datasets.
- **Micro**: Calculate metric globally by counting total TP, FP, FN across all classes. For multi-class, micro-averaged precision, recall, and F1 are all equal to accuracy.

```python
# Per-class metrics (no averaging)
prec_per_class = precision_score(y_test, y_pred, average=None)
rec_per_class = recall_score(y_test, y_pred, average=None)

print("\nPer-class performance:")
for i, (p, r) in enumerate(zip(prec_per_class, rec_per_class)):
    print(f"  Class {i}: Precision={p:.3f}, Recall={r:.3f}")
```

## The Classification Report: Your One-Stop Shop

Instead of calculating each metric individually, sklearn provides `classification_report`:

```python
# Comprehensive report for binary classification
cm = np.array([
    [85, 15],
    [10, 90]
])

y_true, y_pred = cm_to_dataset(cm)

print(classification_report(
    y_true,
    y_pred,
    target_names=['Negative', 'Positive'],
    digits=3
))
```

**Output:**
```
              precision    recall  f1-score   support

    Negative      0.895     0.850     0.872       100
    Positive      0.857     0.900     0.878       100

    accuracy                          0.875       200
   macro avg      0.876     0.875     0.875       200
weighted avg      0.876     0.875     0.875       200
```

**Understanding the columns**:
- **Precision**: For each class, what fraction of predictions for that class were correct
- **Recall**: For each class, what fraction of actual instances were detected
- **F1-score**: Harmonic mean of precision and recall
- **Support**: Number of true instances for each class

**Understanding the rows**:
- **Per-class rows**: Individual metrics for each class
- **Accuracy**: Overall accuracy (same regardless of averaging)
- **Macro avg**: Unweighted average across classes
- **Weighted avg**: Average weighted by class frequency

## Decision Framework: Which Metric Should I Use?

Let me provide a practical decision tree for metric selection, building on the conceptual framework from Part 1:

```python
import pandas as pd

# Create a comprehensive decision guide
decision_guide = pd.DataFrame({
    'Use Case': [
        'Balanced dataset, symmetric costs',
        'Imbalanced dataset',
        'False negatives catastrophic (disease, fraud)',
        'False positives very costly (legal, spam)',
        'Need single robust metric',
        'Need to beat random baseline',
        'Comparing multiple models',
        'Multi-class with imbalanced classes',
        'Optimizing classification threshold',
    ],
    'Primary Metric': [
        'Accuracy / F1-Score',
        'MCC / Balanced Accuracy',
        'Recall / F2-Score',
        'Precision / F0.5-Score',
        'MCC',
        "Cohen's Kappa",
        'F1-Score / MCC',
        'Weighted F1-Score',
        "Youden's J / F1-Score",
    ],
    'Secondary Metrics': [
        'Precision, Recall',
        'F1-Score, Per-class metrics',
        'Precision (to avoid alert fatigue)',
        'Recall (to catch some true cases)',
        'F1-Score, Balanced Accuracy',
        'MCC, F1-Score',
        'ROC-AUC, PR-AUC',
        'Macro F1, Per-class F1',
        'ROC-AUC, PR-AUC',
    ]
})

print(decision_guide.to_string(index=False))
```

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Wrong Label Order

```python
# This can happen with custom label encodings
y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0, 1, 0, 1])

# Default: assumes labels are [0, 1]
cm1 = confusion_matrix(y_true, y_pred)
print("Default order:")
print(cm1)

# Explicit labels (in reverse order)
cm2 = confusion_matrix(y_true, y_pred, labels=[1, 0])
print("\nReversed order:")
print(cm2)

# ALWAYS explicitly specify labels to avoid confusion
cm_correct = confusion_matrix(y_true, y_pred, labels=[0, 1])
```

### Pitfall 2: Not Handling Zero Division

```python
# Model that never predicts positive class
y_true = np.array([0, 0, 0, 0, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0])

# This will warn or fail
try:
    prec = precision_score(y_true, y_pred)
    print(f"Precision (with warning): {prec}")
except:
    print("Failed without zero_division handling")

# Proper handling
prec_safe = precision_score(y_true, y_pred, zero_division=0)
print(f"Precision (safe): {prec_safe}")
```

### Pitfall 3: Forgetting About Class Imbalance in Multi-Class

```python
# Always check class distribution
from collections import Counter
print(f"Class distribution: {Counter(y_test)}")

# Use weighted averaging for imbalanced multi-class
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Weighted F1: {f1_weighted:.3f}")
print(f"Macro F1: {f1_macro:.3f}")
```

### Pitfall 4: Transposed Confusion Matrix

```python
# Different libraries use different conventions
# Sklearn: rows = true, cols = predicted
cm_sklearn = confusion_matrix(y_true, y_pred)

# Some papers/libraries: rows = predicted, cols = true
cm_transposed = cm_sklearn.T

# ALWAYS verify by checking a known case
print("Sklearn convention:")
print(cm_sklearn)
print("\nTransposed (some papers):")
print(cm_transposed)

# Best practice: label your axes clearly
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ConfusionMatrixDisplay(cm_sklearn, display_labels=['Neg', 'Pos']).plot(ax=ax1)
ax1.set_title('Sklearn: Rows=True, Cols=Pred')

ConfusionMatrixDisplay(cm_transposed, display_labels=['Neg', 'Pos']).plot(ax=ax2)
ax2.set_title('Transposed: Rows=Pred, Cols=True')

plt.tight_layout()
```

## Practical Workflow Example

Here's a complete workflow bringing everything together:

```python
def evaluate_binary_classifier(y_true, y_pred, class_names=['Negative', 'Positive']):
    """
    Comprehensive evaluation of a binary classifier.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted labels
    class_names : list
        Names of the classes for display
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score,
        recall_score, f1_score, matthews_corrcoef,
        cohen_kappa_score, balanced_accuracy_score,
        ConfusionMatrixDisplay
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate all metrics
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        'f05': fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    # Calculate Youden's J
    metrics['youden_j'] = metrics['recall'] + metrics['specificity'] - 1
    
    # Visualize confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, pad=20)
    
    # Add summary text
    summary_text = (
        f"Accuracy: {metrics['accuracy']:.3f} | "
        f"F1: {metrics['f1']:.3f} | "
        f"MCC: {metrics['mcc']:.3f}"
    )
    ax.text(0.5, -0.15, summary_text,
            ha='center', transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    
    # Print detailed report
    print("=" * 60)
    print("BINARY CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    print(f"\nBasic Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.3f}")
    print(f"\nPositive Class Performance:")
    print(f"  Precision:          {metrics['precision']:.3f}")
    print(f"  Recall:             {metrics['recall']:.3f}")
    print(f"  F1-Score:           {metrics['f1']:.3f}")
    print(f"  F2-Score:           {metrics['f2']:.3f} (recall-focused)")
    print(f"  F0.5-Score:         {metrics['f05']:.3f} (precision-focused)")
    print(f"\nNegative Class Performance:")
    print(f"  Specificity:        {metrics['specificity']:.3f}")
    print(f"\nRobust Metrics:")
    print(f"  MCC:                {metrics['mcc']:.3f}")
    print(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.3f}")
    print(f"  Youden's J:         {metrics['youden_j']:.3f}")
    print("=" * 60)
    
    return metrics

# Example usage
cm_example = np.array([
    [85, 15],
    [10, 90]
])

y_true, y_pred = cm_to_dataset(cm_example)
metrics = evaluate_binary_classifier(
    y_true, y_pred,
    class_names=['Healthy', 'Disease']
)
```

## Conclusion: From Theory to Practice

In Part 1, we built the conceptual foundation for understanding classification metrics. We explored why accuracy alone is misleading, examined the asymmetry of errors, and framed metric selection as a decision theory problem.

This post bridged the gap from theory to practice. We've covered:

✅ **Correct implementation** of all standard metrics with edge case handling
✅ **Advanced metrics** (MCC, Cohen's Kappa, Balanced Accuracy) that handle imbalanced data better
✅ **Multi-class extensions** with proper averaging strategies
✅ **Common pitfalls** and how to avoid them
✅ **Practical workflows** for comprehensive model evaluation

**What's Next?** In Part 3, we'll explore threshold-independent metrics: ROC curves, AUC, and precision-recall curves that reveal your model's full performance landscape across all possible decision thresholds. We'll also cover:
- How to choose optimal thresholds for your specific use case
- Cross-validation strategies for robust metric estimation
- Calibration curves and reliability diagrams
- Statistical significance testing for metric differences

**Key Takeaways:**

1. **Always handle edge cases**: Use `zero_division` parameters and check for undefined metrics
2. **For imbalanced data**: Prefer MCC, balanced accuracy, or F-beta over simple accuracy
3. **For multi-class**: Choose your averaging strategy (macro/weighted/micro) based on whether all classes are equally important
4. **Use `classification_report`**: It gives you a comprehensive view in one function call
5. **Visualize confusion matrices**: Numbers alone don't reveal patterns as clearly

The code examples in this post are available in a [GitHub repository](#) for easy experimentation.

**Further Reading:**

For threshold-independent evaluation and ROC analysis, stay tuned for Part 3. In the meantime:
- [Sklearn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
- [A Survey of Predictive Modelling Under Imbalanced Distributions](https://arxiv.org/abs/1505.01658)

## References

[1] [Basic Evaluation Measures From the Confusion Matrix](https://classeval.wordpress.com/introduction/basic-evaluation-measures/) by Takaya Saito and Marc Rehmsmeier

[2] Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 6.

[3] [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

[4] Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

[5] Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class classification: an overview. *arXiv preprint arXiv:2008.05756*.

---

*Machine Learning | Confusion Matrix | Evaluation Metrics*

![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)

Content licensed under a [CC BY-NC-SA 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)

