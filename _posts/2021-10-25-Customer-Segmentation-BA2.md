---
layout: post
comments: true
title: "Customer Segmentation using RFM Analysis"
excerpt: "It is impossible to develop a precision marketing strategy without identifying who you want to target. Marketing campaigns tailored to specific customers rely on the ability to accurately categorize those based on their demographic characteristics and buying behavior. Segments should be clearly differentiated from each other according to the maxim: \"Customers are unique, but sometimes similar\".
In this tutorial, we will learn how to segment your customers database using RFM analysis along with hierarchical cluster analysis (HCA) we introduced in the previous tutorial."
author: "Skander Kacem"
tags: 
    - Business Analytics
    - Tutorial
    - Hierarchical Cluster Analysis
    - Customer Segmentation
katex: true
preview_pic: /assets/6/customers.png
---

## Introduction

The field of marketing analytics is very broad and can include fascinating topics such as:

* Text mining,
* Social network analysis,
* sentiment analysis,
* realtime bidding and so on.

However, at the heart of marketing lie few basic questions, that often remain unanswered:

1. Who are my customers?
2. Which customers should I target, and spend most of my marketing budget on?
3. What is the future value of my customers?

In this tutorial, we will explore the first two questions using customer segmentation techniques involving RFM analysis along with HCA.

## Customer Segmentation

You can't treat your customers the same way, offer them the same product, charge the same price, or communicate the same benefits. Otherwise you will miss a lot of value. So you need to understand the distinctions in your customers' needs, wants, preferences and behaviors. Only then can you customize your offering, personalize your customer approach, and optimize your marketing campaigns.  

Now, in today's digital world, almost every online retailer holds a massive customer database. So, how do we deal with big customer databases?  
Treating each and every customer individually is very costly. 

A great segmentation is all about finding a good balance between simplifying enough, so it remains usable and not simplifying too much, so it's still statistically and managerially relevant.

## RFM Analysis

RFM (Recency, Frequency, Monetary) analysis is a segmentation approach based on customer behavior. It groups customers according to their purchasing behavior. How recently, how frequently, and how much has a customer spent on purchases.  It is one of the best predictor of future purchase and customer lifetime value in terms of their activity and engagement.

Recency indicates when a customer made his or her last purchase. Usually the smaller the recency, that is the more recent the last purchase, the more likely the next purchase will happen soon. On the other hand, if a customer has lapsed and has not made any purchase for a long period of time, he may have lost interest or switched to competition.  
Frequency is about the number of customers purchases in a given period. Customers who bought frequently were more likely to buy again, as opposed to customers who made only one or two purchases.  
Monetary is about how much a customer has spent during a certain period of time. Customers who spend more should be treated differently than customers who spend less. Furthermore, they are more likely to buy again.  

## Practical Example in Python

We start as usual, by loading the required libraries and the dataset, and set up our environment. However, make sure you have `pandassql` and `squarify` installed on your Python environment.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram , linkage, cut_tree
import squarify
```

```python
# Notebooks set up
pysqldf = lambda q: sqldf(q, globals())
%precision %.2f
pd.options.display.float_format = '{:,.2f}'.format
plt.rcParams["figure.figsize"] = (12,8)

# read the dataset from the csv file
headers = ['customer_id', 'purchase_amount', 'date_of_purchase']
df = pd.read_csv('../Datasets/purchases.txt', header=None, 
                 names=headers, sep='\t')
```
### Data Wrangling

The dataframe consists of 51,243 observations across 3 variables:

1. Customer ID,
2. purchase amount in USD and
3. date of purchase

The data is clean and covers the period January 2nd 2005 to December 31st 2015.

```python
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 51243 entries, 0 to 51242
Data columns (total 3 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customer_id       51243 non-null  int64  
 1   purchase_amount   51243 non-null  float64
 2   date_of_purchase  51243 non-null  object 
dtypes: float64(1), int64(1), object(1)
memory usage: 1.2+ MB
```

We proceed by transforming the column `data_of_purchase` from object a date object.

```python
# Transform the last column to a date object
df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], format='%Y-%m-%d')
```

To compute `recency` we will create a new variable `day_since`, which contains the number of days since the last purchase. For that, we set January 1st 2016 as the pining date and count backward the number of days from the latest purchase for each customer based on customer id.  

```python
# Extract year of purchase and save it as a column
df['year_of_purchase'] = df['date_of_purchase'].dt.year
# Add a day_since column showing the difference between last purchase and a basedate
basedate = pd.Timestamp('2016-01-01')
df['days_since'] = (basedate - df['date_of_purchase']).dt.days
```

### RFM Computation

To implement the RFM analysis, further data processing need to be done.  
Next we are going to compute the customers recency, frequency and average purchase amount. This part is a big tricky specially when it's done with pandas. The trick here is that the customer ID will only appear once for every customer. So even though we have 51,243 purchases we'll only have as many unique customer IDs as there are in the database.  
Now, for each customer, we need to compute the minimum number of days between all of his or her purchases and January 1st, 2016. Of course, if we take the minimum number of days, then we are going to have the day of the last purchase, which is the very definition of recency.  
Then for each customer we need to compute the frequency, which is basically how many purchases that customer has made.  
This is  done with the python sql module:
```python
# Compute recency, frequency, and average purchase amount
q = """
        SELECT customer_id,
        MIN(days_since) AS 'recency',
        COUNT(*) AS 'frequency',
        AVG(purchase_amount) AS 'amount'
        FROM df GROUP BY 1"""
customers = sqldf(q)
```

The asterisk here actually refers to anything in the data that is related to this customer; and then for the amount, we averaged the purchase amount for that specific customer ID and named that aggregate calculation as `amount`.  
Now, the trick is we want to make sure that each row only appears once per customer. So we're going to calculate that from the data and group by one, which means that everything here is going to be calculated and grouped by customer ID.  

Of course, it would have been possible to perform all the calculations using only the pandas. However, it would have required much more work. When working with databases, it is very useful to have some knowledge of sql. As we can see here, only one sql command is required to create an RFM table.  
The results are then stored under the variable `customers` and the first five entries are as follows:

```python
customers.head()
```

```text
+----+---------------+-----------+-------------+----------+
|    |   customer_id |   recency |   frequency |   amount |
|----+---------------+-----------+-------------+----------|
|  0 |            10 |      3829 |           1 |       30 |
|  1 |            80 |       343 |           7 |  71.4286 |
|  2 |            90 |       758 |          10 |    115.8 |
|  3 |           120 |      1401 |           1 |       20 |
|  4 |           130 |      2970 |           2 |       50 |
+----+---------------+-----------+-------------+----------+
```

### Exploratory Data Analysis

We start by generating some descriptive statistics, including those that summarize the central tendency, dispersion and shape of each of our RFM variables.

```python
customers[['recency', 'frequency', 'amount']].describe()
```

```text
+-------+-----------+-------------+----------+
|       |   recency |   frequency |   amount |
|-------+-----------+-------------+----------|
| count |     18417 |       18417 |    18417 |
| mean  |   1253.04 |     2.78237 |   57.793 |
| std   |   1081.44 |     2.93689 |   154.36 |
| min   |         1 |           1 |        5 |
| 25%   |       244 |           1 |  21.6667 |
| 50%   |      1070 |           2 |       30 |
| 75%   |      2130 |           3 |       50 |
| max   |      4014 |          45 |     4500 |
+-------+-----------+-------------+----------+
```











## References

1. Karl Melo. [Customer Analytics I - Customer Segmentation](https://rstudio-pubs-static.s3.amazonaws.com/226524_10f550ea696f4db8a033c6583a8fc526.html). 2016
2. Navlani, A. [Introduction to Customer Segmentation in Python](https://www.datacamp.com/community/tutorials/introduction-customer-segmentation-python). datacamp, 2018.
3. Lilien, Gary L, Arvind Rangaswamy, and Arnaud De Bruyn. 2017. Principles of Marketing Engineering and Analytics. State College, PA: Decisionpro.
4. Jim Novo. [Turning Customer Data into Profits](https://www.jimnovo.com/RFM-tour.htm)
5. Kohavi, Ron, Llew Mason, Rajesh Parekh, and Zijian Zheng. 2004. "Lessons and Challenges from Mining Retail E-Commerce Data." Machine Learning 57 (1/2): 83–113.
6. Preview Picture by Vecteezy. [Customer Segmentation Vectors](https://www.vecteezy.com/free-vector/customer-segmentation) 
7. 