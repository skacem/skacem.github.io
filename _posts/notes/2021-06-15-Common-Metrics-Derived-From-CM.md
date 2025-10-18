---
layout: post
category: ml
comments: true
title: "Common Metrics Derived From the Confusion Matrix"
excerpt: "In the previous post we introduced the confusion matrix in the context of hypothesis testing and we showed how such a simple and intuitive concept can be a powerful tool in illustrating the outcomes of classification models. Now, we are going to discuss various performance metrics that are derived from a confusion matrix."
author: "Skander Kacem"
tags:
  - Machine Learning
  - Confusion Matrix
  - Evaluation Metrics
katex: true
---


## Introduction

In [Part 1](/ml/2021/06/07/Confusion-Matrix/), we explored why accuracy alone can be dangerously misleading. We looked at the accuracy paradox through examples like fraud detection and medical diagnostics, examined why Type I and Type II errors rarely carry equivalent costs, and discussed how to think about metric selection as a decision theory problem.

Now for the practical side. You understand *why* these metrics matter—this post is about getting them right in your code. We'll dig into the implementation details that textbooks skip over: edge cases that silently break your calculations, zero division errors, label ordering gotchas, and when sklearn's defaults might not do what you expect.

We'll also go beyond the basic metrics from Part 1. Matthews Correlation Coefficient, Cohen's Kappa, balanced accuracy—these aren't just academic extras. They handle real-world messiness (like class imbalance) far better than the standard precision-recall-F1 trio.

## Quick Review: The Confusion Matrix

Here's a quick refresher. For binary classification, the confusion matrix breaks down into four outcomes:

$$\text{CM} = \left[\begin{array}{cc} TN & FP \\ FN & TP \end{array}\right]$$

Where:
- **TN** (True Negative): Correct negative predictions
- **FP** (False Positive): Incorrect positive predictions (Type I error)
- **FN** (False Negative): Incorrect negative predictions (Type II error)  
- **TP** (True Positive): Correct positive predictions

One thing to watch out for—sklearn puts **rows as true labels** and **columns as predictions**. Some textbooks and other libraries flip this around. I've seen people spend hours debugging their model only to realize they were reading the confusion matrix backwards. Always double-check which convention you're using:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0, 1, 0, 1])

# Sklearn: rows = true, columns = predicted
cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[1 1]    <- actual 0s: 1 correct, 1 wrong
#  [1 1]]   <- actual 1s: 1 wrong, 1 correct

# To avoid confusion, explicitly specify labels
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
```

## Implementation Fundamentals

### The `cm_to_dataset` Utility

Throughout this post, I use a helper function that converts a confusion matrix back into arrays of predictions. It's useful for demonstrating concepts without training actual models. Here's how it works:

```python
def cm_to_dataset(cm):
    """
    Convert a confusion matrix into y_true and y_pred arrays.
    
    Useful for testing and demonstrations when you want to work
    backwards from known confusion matrix values.
    """
    import numpy as np
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # Build arrays for each quadrant
    y_true = np.array([0]*tn + [0]*fp + [1]*fn + [1]*tp)
    y_pred = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
    
    # Shuffle to avoid ordering artifacts
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

Let's walk through the basic metrics from Part 1, but this time focusing on **getting the code right** and handling the weird edge cases that can bite you.

### Accuracy: Still Not Great, But At Least It Won't Break

Accuracy is straightforward—it's just the fraction of correct predictions:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

We already know from Part 1 that accuracy is misleading on imbalanced datasets. But at least it never throws errors, since the denominator is always the total number of samples.

```python
# The classic example: 99% accuracy, completely useless model
cm = np.array([
    [990, 0],
    [10, 0]
])

y_true, y_pred = cm_to_dataset(cm)

# Two ways to calculate it
acc_manual = (cm[0,0] + cm[1,1]) / cm.sum()
acc_sklearn = accuracy_score(y_true, y_pred)

print(f"Manual: {acc_manual:.3f}")
print(f"Sklearn: {acc_sklearn:.3f}")
# Both give 0.990
```

The model just predicts "negative" for everything and still gets 99% accuracy. This is why we need better metrics.

### Precision: The Part That Actually Breaks

As a quick reminder from Part 1: precision asks "Of all the things we predicted as positive, how many actually were?" It's the reliability of your positive predictions—high precision means when you say "positive," you're usually right.

$$\text{Precision} = \frac{TP}{TP + FP}$$

Here's where things get interesting. What happens when your model never predicts the positive class?

```python
# A model that's so conservative it never predicts positive
cm = np.array([
    [990, 0],
    [10, 0]
])

y_true, y_pred = cm_to_dataset(cm)

# Naive calculation - what could go wrong?
tp = cm[1, 1]
fp = cm[0, 1]
prec_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"Manual (with check): {prec_manual:.3f}")

# Sklearn has a parameter for this exact situation
prec_sklearn = precision_score(y_true, y_pred, zero_division=0)
print(f"Sklearn: {prec_sklearn:.3f}")
# Both give 0.000
```

That `zero_division` parameter? It's there because dividing by zero when `TP + FP = 0` is a real problem. You can set it to 0, 1, or let it warn you. I usually go with 0 because a model that never predicts positive deserves a zero score, not a free pass.

```python
# See the difference
print(f"zero_division=0: {precision_score(y_true, y_pred, zero_division=0)}")
print(f"zero_division=1: {precision_score(y_true, y_pred, zero_division=1)}")
```

Setting it to 1 would give you 100% precision for a model that makes no predictions. That seems... generous.

### Recall: Catching What Matters

Recall (or sensitivity) measures completeness—what fraction of actual positives did you find? From Part 1, remember this is critical when missing a positive case is catastrophic (cancer, fraud, epidemic containment).

$$\text{Recall} = \frac{TP}{TP + FN}$$

```python
# A model that found only 1 out of 10 positive cases
cm = np.array([
    [990, 0],
    [9, 1]
])

y_true, y_pred = cm_to_dataset(cm)

tp = cm[1, 1]
fn = cm[1, 0]
rec_manual = tp / (tp + fn)
rec_sklearn = recall_score(y_true, y_pred)

print(f"Manual: {rec_manual:.3f}")
print(f"Sklearn: {rec_sklearn:.3f}")
# Both give 0.100 - only caught 10% of positives
```

Recall is less prone to division by zero issues than precision. You'd need literally zero positive samples in your dataset for it to break, which would mean you're trying to train a classifier on data that has nothing to classify. If that's happening, you have bigger problems.

As we discussed in Part 1: optimize for recall when missing a positive case is catastrophic (cancer, fraud, anything where false negatives can get people hurt or cost millions).

### F1-Score: The Compromise

The F1-score tries to balance precision and recall with a harmonic mean. From Part 1, remember we use the harmonic mean (not arithmetic) because it punishes models that excel at one metric while tanking the other.

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Why harmonic mean? Because it doesn't let you cheat.

```python
cm = np.array([
    [990, 0],
    [9, 1]
])

y_true, y_pred = cm_to_dataset(cm)

# Calculate precision and recall first
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)

# Then F1
f1_manual = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
f1_sklearn = f1_score(y_true, y_pred)

print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1_sklearn:.3f}")
# F1 = 0.182 - much lower than the arithmetic mean would be
```

A model with 100% precision but 10% recall gets an F1 of only 18%, not 55%. That's the point—the harmonic mean doesn't let you cheat.

### Specificity: The Other Side of the Coin

Specificity is recall's mirror image—it measures how good you are at identifying negatives:

$$\text{Specificity} = \frac{TN}{TN + FP}$$

Sklearn doesn't have a `specificity_score` function, probably because people don't ask for it as often. But it's easy enough to calculate:

```python
cm = np.array([
    [990, 0],
    [9, 1]
])

tn = cm[0, 0]
fp = cm[0, 1]
spec = tn / (tn + fp)
print(f"Specificity: {spec:.3f}")
# Output: 1.000
```

In medical testing, you'll often see sensitivity (recall) and specificity reported together. They give you a complete picture: sensitivity tells you how good the test is at catching disease, specificity tells you how good it is at not falsely alarming healthy people.

## Advanced Metrics: Better Tools for Messy Data

The metrics above are standard, but they all have issues with imbalanced data. Here's where things get more interesting.

### Matthews Correlation Coefficient: The Underrated Champion

If I could only pick one metric for binary classification, it'd probably be MCC. It's a correlation coefficient between predictions and truth, ranging from -1 (total disagreement) to +1 (perfect prediction), with 0 meaning you're basically guessing randomly.

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

The magic of MCC is that it uses all four confusion matrix values and treats both classes symmetrically. Let me show you why it's better than accuracy:

```python
from sklearn.metrics import matthews_corrcoef

# Scenario 1: A decent classifier on imbalanced data
cm = np.array([
    [990, 10],
    [5, 95]
])

y_true, y_pred = cm_to_dataset(cm)

mcc = matthews_corrcoef(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"MCC: {mcc:.3f}")
print(f"Accuracy: {acc:.3f}")
# MCC: 0.825
# Accuracy: 0.986

# Scenario 2: The useless classifier that just picks the majority class
cm_useless = np.array([
    [1000, 0],
    [100, 0]
])

y_true_u, y_pred_u = cm_to_dataset(cm_useless)
print(f"\nUseless classifier:")
print(f"MCC: {matthews_corrcoef(y_true_u, y_pred_u):.3f}")
print(f"Accuracy: {accuracy_score(y_true_u, y_pred_u):.3f}")
# MCC: 0.000
# Accuracy: 0.909
```

See that? The useless classifier gets 91% accuracy (because the classes are imbalanced) but gets an MCC of exactly 0—which is what a random classifier would get. Meanwhile, the decent classifier's MCC of 0.825 properly reflects that it's doing real work.

This is why MCC should be your go-to when dealing with imbalanced datasets. It won't lie to you the way accuracy does.

### Cohen's Kappa: Are You Better Than a Coin Flip?

Cohen's Kappa asks a simple question: is your model actually learning something, or is it just getting lucky? It measures agreement between predictions and truth, but corrects for the agreement you'd expect by pure chance.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ is observed agreement (just accuracy) and $p_e$ is what you'd expect from random guessing given the class frequencies.

```python
from sklearn.metrics import cohen_kappa_score

cm = np.array([
    [85, 15],
    [10, 90]
])

y_true, y_pred = cm_to_dataset(cm)

kappa = cohen_kappa_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"Cohen's Kappa: {kappa:.3f}")
print(f"Accuracy: {acc:.3f}")
# Kappa: 0.750
# Accuracy: 0.875
```

The interpretation scale (these are rough guidelines):
- Below 0: Worse than random
- 0.01-0.20: Barely better than guessing
- 0.21-0.40: Fair
- 0.41-0.60: Moderate  
- 0.61-0.80: Substantial
- 0.81-1.00: Near perfect

So a kappa of 0.75 means substantial agreement—your model is genuinely learning something useful.

### Balanced Accuracy: Equal Treatment

Balanced accuracy is just the average of recall on each class. Simple, but effective for imbalanced data:

$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$

```python
from sklearn.metrics import balanced_accuracy_score

cm = np.array([
    [950, 50],
    [5, 95]
])

y_true, y_pred = cm_to_dataset(cm)

bal_acc = balanced_accuracy_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"Balanced Accuracy: {bal_acc:.3f}")
print(f"Regular Accuracy: {acc:.3f}")
# Balanced: 0.925
# Regular: 0.950
```

Regular accuracy is inflated by the majority class. Balanced accuracy treats both classes equally, which is often what you actually want.

### Youden's J: Finding the Sweet Spot

Youden's J statistic is simple but useful, especially when you're trying to pick a classification threshold:

$$J = \text{Sensitivity} + \text{Specificity} - 1$$

It ranges from 0 to 1, and basically asks "how much better than random are you on both classes combined?"

```python
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
# Youden's J: 0.750
```

This is particularly useful when you're looking at ROC curves and trying to find the optimal threshold—just pick the point that maximizes J. We'll talk more about that in Part 3.

### F-Beta Score: When You Need to Pick Sides

F1-score tries to balance precision and recall equally. But what if you don't want balance? What if you care more about one than the other?

That's where F-beta comes in. It's just F1 with a dial you can turn:

$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

- $\beta < 1$: Precision matters more
- $\beta = 1$: F1-score (balanced)
- $\beta > 1$: Recall matters more

```python
from sklearn.metrics import fbeta_score

cm = np.array([
    [90, 10],
    [20, 80]
])

y_true, y_pred = cm_to_dataset(cm)

# Try different beta values
f1 = fbeta_score(y_true, y_pred, beta=1)
f2 = fbeta_score(y_true, y_pred, beta=2)
f05 = fbeta_score(y_true, y_pred, beta=0.5)

prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)

print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"\nF0.5 (precision priority): {f05:.3f}")
print(f"F1 (balanced): {f1:.3f}")
print(f"F2 (recall priority): {f2:.3f}")
```

Use F2 when false negatives are about twice as bad as false positives. Use F0.5 when false positives are about twice as bad. The beta value is basically saying "this error type is β times more important."

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

The choice matters a lot with imbalanced classes. Macro average treats all classes equally (so rare classes have equal weight to common ones). Weighted average accounts for how common each class is. Pick based on your priorities—do you care more about getting rare classes right, or overall performance?

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

## Which Metric Should You Actually Use?

This is the question that matters. You've got a model, you need a number to optimize—which one do you pick?

Here's my take, built on top of the conceptual framework from Part 1:

**If your classes are balanced and errors cost about the same:** Just use accuracy or F1. They'll both tell you basically the same story. Don't overthink it.

**If your classes are imbalanced:** This is where most people go wrong. Skip accuracy entirely. Use MCC or balanced accuracy instead. They won't lie to you the way accuracy does when you've got 99 negatives for every positive.

**If missing positives is catastrophic** (cancer, fraud, terrorism): Optimize for recall. You'd rather deal with false alarms than miss the one case that matters. Consider F2-score if you want to balance things slightly (weights recall more than precision).

**If false positives are very expensive** (spam filtering, legal accusations): Optimize for precision. Sending an important email to spam or accusing someone falsely has real costs. F0.5-score works if you want some balance while still prioritizing precision.

**If you need one robust number for comparing models:** MCC. It handles imbalanced data well, treats both classes fairly, and actually returns zero for a random classifier (unlike accuracy). This is my default for any serious evaluation.

**If you're tuning a classification threshold:** You want Youden's J statistic or just look at ROC/PR curves directly (that's Part 3 territory).

**For multi-class problems with imbalanced classes:** Use weighted F1-score. The "weighted" part accounts for class imbalance. Or calculate MCC if you're doing binary classification for each class.

## Putting It All Together

Instead of calculating metrics one by one, sklearn's `classification_report` gives you everything at once:

```python
from sklearn.metrics import classification_report

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

Output:
```
              precision    recall  f1-score   support

    Negative      0.895     0.850     0.872       100
    Positive      0.857     0.900     0.878       100

    accuracy                          0.875       200
   macro avg      0.876     0.875     0.875       200
weighted avg      0.876     0.875     0.875       200
```

This gives you precision, recall, and F1 for each class, plus the overall accuracy and both macro and weighted averages. For most use cases, this is your starting point. Look at the numbers, see where your model struggles, then dig deeper with the specific metrics that matter for your problem.

For the advanced stuff—MCC, Cohen's Kappa, balanced accuracy—you'll need to calculate those separately. But honestly, for day-to-day model evaluation, `classification_report` plus maybe MCC is usually enough.

## Wrapping Up

Part 1 gave you the conceptual tools to think about classification metrics—why accuracy fails, when to optimize for precision versus recall, how to think about the asymmetry of errors. This post was about getting the implementation right and going beyond the basics. I've included brief reminders of the key definitions from Part 1 because, well, repetition helps with understanding.

The most important things to take away:

- **Handle edge cases.** Use `zero_division` parameters when needed. Always explicitly specify your label ordering. 
- **For imbalanced data, skip accuracy.** Use MCC, balanced accuracy, or F-beta instead. They won't mislead you.
- **Know your averaging strategy.** With multi-class classification, understand when to use macro vs weighted averaging.
- **Use `classification_report`.** It gives you everything at once and formats it nicely.

The advanced metrics we covered—especially MCC and balanced accuracy—handle the messy reality of real-world data much better than the basic precision-recall-F1 combo. They deserve to be used more widely.

**What's next:** Part 3 will dive into threshold-independent metrics: ROC curves, AUC, precision-recall curves. Basically, how to see your model's full performance landscape instead of just evaluating it at a single decision threshold.

All the code examples are in a [GitHub repository](#) if you want to play around with them.

For more on evaluation:
- [Sklearn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)
- [A Survey of Predictive Modelling Under Imbalanced Distributions](https://arxiv.org/abs/1505.01658)

## References

[1] [Basic Evaluation Measures From the Confusion Matrix](https://classeval.wordpress.com/introduction/basic-evaluation-measures/) by Takaya Saito and Marc Rehmsmeier

[2] Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 6.

[3] [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

[4] Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

[5] Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class classification: an overview. *arXiv preprint arXiv:2008.05756*.


