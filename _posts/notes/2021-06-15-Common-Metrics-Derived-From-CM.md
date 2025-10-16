---
layout: post
category: ml
comments: true
title: Common Metrics Derived From the Confusion Matrix
excerpt: "In the previous post we introduced the confusion matrix in the context of hypothesis testing and we showed how such a simple and intuitive concept can be a powerful tool in illustrating the outcomes of classification models. Now, we are going to discuss various performance metrics that are derived from a confusion matrix."
author: "Skander Kacem"
tags: 
  - Machine Learning 
  - Confusion Matrix 
  - Evaluation Metrics
katex: true
preview_pic: /assets/0/SPS_dist.gif
---

In the previous post we introduced the confusion matrix in the context of hypothesis testing and we showed how such a simple and intuitive concept can be a powerful tool in illustrating the outcomes of classification models. Now, we are going to discuss various performance metrics that are computed from a confusion matrix.

## Introduction

Confusion matrix is a basic instrument in machine learning used to depict the outcomes of  classification models. It provides insight into the nature of the classification errors by illustrating the match and mismatch between the predicted values and the corresponding true values. In binary classification tasks, there are four possible outcomes:

1. True negative ($$TN$$): number of correct negative prediction
2. False positive ($$FP$$): number of incorrect positive prediction
3. False negative ($$FN$$): number of incorrect negative prediction
4. True positive ($$TP$$): number of correct positive prediction

which together comprise the 2 x 2 Confusion Matrix (CM):

$$\text{CM} = 
\left[\begin{array}{cc} TN & FP \\ FN & TP \end{array}\right],$$

where:

* The predicted outcomes are the columns and the true outcomes are the rows<sup id='n1'>[1](#note1)</sup>,  
* The diagonal elements of the matrix show  the number of correct predictions, and  
* The off-diagonal elements show the number of incorrect predictions.

CM provides all the information necessary to derive the metrics that are used to evaluate the performance of a classifier.   In the next sections we are going to discuss some of these metrics and how to generate them using Python and `sklearn` module.

## Accuracy

A simple way of measuring the performance of a classifier is to consider how often it was right, that is accuracy.  
Accuracy (ACC) is the most widely-used metric in classification models. It is a ratio of correct predictions to the total sample size. It indicates how often is the classifier correct and it is defined as:

$$
ACC = \frac{TP + TN}{TP + FN + TN + FP}  
$$

Accuracy works best if the class labels are uniformly distributed.  

To understand the above statement, let's consider the following confusion matrix:  

$$\text{CM} = 
\left[\begin{array}{cc} TN & FP \\ FN & TP \end{array}\right] =
\left[\begin{array}{cc}
 990 & 0 \\
   10 & 0
 \end{array}\right]
$$
  
According to the accuracy, the classifier's performance would be nearly perfect ($$=99\%$$). Which is clearly misleading, since the model doesn't detect a single  positive instance.  Hence, accuracy is not an adequate performance measure in this case or any other imbalanced dataset.

Imbalanced datasets are common in real life, and it is usually the rare class that we aim to detect e.g. diseases in medicine, fraudulent claims in insurance  or potential buyers in e-commerce. In these cases, it is more appropriate to compute the accuracy for positives and negatives separately. That is the topic of the next two sections.  

For now, though, let's see how all this is implemented in Python:

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset
from sklearn.metrics import accuracy_score

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [10, 0]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
acc = sum(cm.diagonal())/cm.sum()
print(f"1. Method: ACC = {acc}")

# 2. Method use sklearn.metrics module
acc = accuracy_score(y_true, y_pred)
print(f"2. Method: ACC = {acc}")
```
    1. Method: ACC = 0.99
    2. Method: ACC = 0.99

## Precision

Precision (PREC) is the ratio of correct positive predictions out of all positive predictions made, or the accuracy of predicted positives. It is defined as follows:

$$PREC = \frac{TP}{TP + FP}$$

As we can see from the equation above, a model that produces no false positives has a precision of $$100\%$$, no matter how many true positives it classifies. It could be one or all of them. In fact, precision doesn't care  how many positive instances there are in the dataset ($$TP + FN$$). It only cares about the ones it predicts ($$TP + FP$$). This metric is important when the cost of $$FP$$ is particularly high and/or $$TP$$s are rare.

Let's take the example from the previous section:

$$\text{CM} = \left[\begin{array}{cc}
 990 & 0 \\
   10 & 0
 \end{array}\right]$$  

 Putting these values into the precision formulae we obtain: $$PREC = 0$$. Well, now we can see that our model which labeled all samples as negative wasn't very helpful. Although it has a very high accuracy score, it has 0 precision.  

 Let's now modify our CM slightly, and correctly identify a single instance as positive:

 $$\text{CM} =
 \left[\begin{array}{cc} 990 & 0 \\ 9 & 1 \end{array}\right]
 $$
 
 Putting these new values into the precision formulae we obtain: $$PREC = 1.0$$. Only one correctly classified positive instance is enough to have a perfect score. However, precision reflects only one part of the performance. To fully evaluate the effectiveness of a model, we must examine both precision and recall.
And now the corresponding python code:

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset
from sklearn.metrics import precision_score

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [10, 0]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
# tp = 0 => prec = 0 
prec = cm[1,1] / 1.0
print(f"1. Method: PREC = {prec}")

# 2. Method use sklearn.metrics module
prec = precision_score(y_true, y_pred, zero_division=0)
print(f"2. Method: PREC = {prec}")
```

    1. Method: PREC = 0.0
    2. Method: PREC = 0.0

Python code of the example with the slightly modified confusion matrix:

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset
from sklearn.metrics import precision_score

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [9, 1]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
# tp = 0 => prec = 0 
prec = cm[1,1] / cm[:,1].sum()
print(f"1. Method: PREC = {prec}")

# 2. Method use sklearn.metrics module
prec = precision_score(y_true, y_pred, zero_division=0)
print(f"2. Method: PREC = {prec}")
```

    1. Method: PREC = 1.0
    2. Method: PREC = 1.0

## Recall (Sensitivity)

Recall (also known as sensitivity) is defined as follows:

$$
REC = \frac{TP}{TP + FN}
$$

 It is the ratio of correct predicted positives to the total number of real positives. It is about completeness, classifying all instances as positive yields 100% recall, but a very low precision. Recall tells you how often your predictions actually capture the positive class!

Let's compute the recall of the previous example, where

 $$\text{CM} =
 \left[\begin{array}{cc} 1 & 9 \\ 0 & 990 \end{array}\right]
 $$

$$REC = \frac{1}{1 + 9} = 0.1$$. Well, that was a low score. Which was in line with our expectation.  This is because precision and recall are often in tension. That is, an improvement in precision typically worsens recall and vice versa.  

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset
from sklearn.metrics import recall_score

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [9, 1]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
rec = cm[1,1] / cm[1,:].sum()
print(f"1. Method: REC = {rec}")

# 2. Method use sklearn.metrics module
rec = recall_score(y_true, y_pred)
print(f"2. Method: REC = {rec}")
```

    1. Method: REC = 0.1
    2. Method: REC = 0.1

## F1-Score

Precision and recall metrics are usually reported together, rather than individually. Since, it is easy to vary the sensitivity of a model to improve precision at the expense of recall, or vice versa. Unless, we want explicitly to maximize either recall or precision at the expense of the other metric.  

To make a summary out of them, we usually use the F1-Score, also known as a harmonic mean, which is computed by doubling the product over the sum:

$$
F_1 = 2\times\frac{PREC.REC}{PREC + REC}
$$

We use the harmonic mean instead of a simple average because it punishes extreme values.  There are other metrics for combining precision and recall, such as the Geometric Mean of precision and recall, but the F1 score is the most commonly used.

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset
from sklearn.metrics import f1_score

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [9, 1]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
# prec and rec from the last sections
f1 = 2 * (prec * rec)/ (prec + rec)
print(f"1. Method: F1-Score = {f1}")

# 2. Method use sklearn.metrics module
f1 = f1_score(y_true, y_pred)
print(f"2. Method: F1-Score = {f1}")
```
    1. Method: F1-Score = 0.18
    2. Method: F1-Score = 0.18

## Specificity

Specificity is the ability of a model to correctly predict negative instances. It is defined as follows:

$$ SPEC = \frac{TN}{TN + FP} $$

Specificity is usually combined with sensitivity. Their combination uses all four numbers in the confusion matrix, as opposed to precision and recall which only use three.  
To make a summary out of them, we usually use the geometric mean, which is defined as the square root of the product:

$$ G = \sqrt{sensitivity \times specificity}$$

This formula has the beneficial property of averaging out both scores while penalizing unbalanced pairs.

```python
# Imports
import numpy as np
from cmutils import cm_to_dataset

# Set the confusion matrix
cm = np.array([ 
              [990, 0], 
              [9, 1]])
# generate y_true and y_pred based on the confusion matrix
y_true, y_pred = cm_to_dataset(cm)

# 1. Method: use confusion matrix and formulae
spec = cm[0,0] / cm[0,:].sum()
print(f"1. Method: SPEC = {spec}")

# 2. there is no skleaern method 
```
    1. Method: SPEC = 1.0

## Summary

In this article, we presented some of the common scores derived from the confusion matrix. We also discussed:

* their advantages and disadvantages and
* When to use them and when not to.

At the end of each section, we showed how to compute each score in Python, using either the formulas and the confusion matrix or sklearn module. In reality, we don't need to calculate each metric individually, we just call the `classification_report` function of `sklearn` and we are done. 

In advanced classification problems, we tend to use other, more sophisticated metrics because they contain more information. In fact, one of these metrics is usually good enough to evaluate the performance of the model. This will be the topic of my next post.


## References

[1] [Evaluation Metrics, ROC-Curves and imbalanced datasets](http://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/) by David S. Batista  
[2] [Modelling Rare Events](https://medium.com/eliiza-ai/modelling-rare-events-c169cb081d8b) by Eike Germann  
[3] Bruce, P., A. Bruce, and P. Gedeck. 2020. Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python. O’Reilly Media.  
[4] [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) by Google
[5] [Beyond Accuracy: Precision and Recall](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) by Will Koehrsen
[6] [What are Classification Metrics?](https://towardsdatascience.com/the-illustrated-guide-to-classification-metrics-the-basics-cf3c2e9b89b2) by Ygor R. Serpa

---
<a name="note1">1</a>: You sometimes find this matrix transposed, with the columns representing the true values and the rows the predicted ones, as in my precedent post about confusion matrix. In this article, I wanted to be consistent with the python `sklearn` module, hence this choice. So be careful when interpreting confusion matrices! [↩](#n1)

[comment]: <> (Once you have your confusion matrix ready, you can use it to calculate more nuanced performance metrics used in evaluating classification models.)
[comment]: <> (In this post we are going to discuss various performance metrics that are computed from a confusion matrix.   We start with the first two basic measures: Error rate and Accuracy)



