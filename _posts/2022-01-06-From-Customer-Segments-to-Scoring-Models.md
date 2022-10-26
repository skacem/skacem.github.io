---
layout: post
comments: true
title: "From RFM Analysis to Predictive Scoring Models"
excerpt: "From a managerial perspective, it is extremely useful to determine not only the contribution of customer segments to today's sales, but also the expected contribution of each segment to future revenues. After all, how can we develop good plans and make informed strategic decisions if we can't  forecast sales and revenues?
In this tutorial, we'll learn how to predict next year's customer activity and dollar value based on RFM analysis."
author: "Skander Kacem"
tags: 
    - Business Analytics
    - Tutorial
    - RFM Analysis
    - Customer Segmentation
katex: true
preview_pic: /assets/8/salesup.png
---

## Introduction

Business forecasting is not about accurately predicting the future, nor is it an end in itself. It is a means to help companies manage uncertainty and develop strategies to minimize potential risks and threats or maximize financial returns.  It relies on identifying and estimating past patterns that are then generalized to forecast the future. As such, forecasting has a wide range of business applications and can be useful throughout the value chain of any organization.  
For instance, forecasting customer purchases allows companies not only to calculate the expected revenue of each segment so they can focus on profitable ones, but also to properly plan inventory and manpower for the next period.  Other business forecasting applications include customer lifetime value, credit card attrition or future revenues. In other words, every business needs forecasting.

In this tutorial we will develop a typical marketing application. We will predict the likelihood that a customer will make a purchase in a near future. And if so, we also want to predict how much s/he will spend. For this, we will first segment our customer database using RFM analysis --I hope you are by now familiar with this segmentation method-- and then we will build two distinct models:

1. A first model to calculate the probability of purchase
2. A second model to predict the amount spent  

For simplicity sake, we will use linear regression in both models. As it is one of the simplest supervised learning algorithms.  
Finally, we will combine the two models into one large scoring model.

## RFM Analysis and Customer Segmentation

In this section we will use the same code as in the previous tutorial. The only difference is that we will save the resulting customer segmentation as `customers_2015` rather than just `customers`.  
As usual, we start by importing the necessary libraries, setting up the notebook environment and reading the data set as a dataframe.

```python
# import needed packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import squarify
from pandasql import sqldf
from tabulate import tabulate

# Set up the notebook environment
pysqldf = lambda q: sqldf(q, globals())
%precision %.2f
pd.options.display.float_format = "{:,.2f}".format
plt.rcParams["figure.figsize"] = 10, 8

# Load text file into a local variable
columns = ["customer_id", "purchase_amount", "date_of_purchase"]
df = pd.read_csv("purchases.txt", header=None, sep="\t", names=columns)

# interpret the last column as datetime
df["date_of_purchase"] = pd.to_datetime(df["date_of_purchase"], format="%Y-%m-%d")
```

Then we add two variables: `year_of_purchase` and `days_since`.

```python
# Extract year of purchase and save it as a column
df["year_of_purchase"] = df["date_of_purchase"].dt.year

# Add a day_since column showing the difference between last purchase and a basedate
basedate = pd.Timestamp("2016-01-01")
df["days_since"] = (basedate - df["date_of_purchase"]).dt.days
```

We now use SQL for the RFM analysis and then segment our customer database accordingly.

```python
q = """
        SELECT customer_id,
        MIN(days_since) AS 'recency',
        MAX(days_since) AS 'first_purchase',
        COUNT(*) AS 'frequency',
        AVG(purchase_amount) AS 'amount'
        FROM df GROUP BY 1"""
customers_2015 = sqldf(q)
```

```python
# Managerial Segmentation based on RFM analysis
customers_2015.loc[customers_2015["recency"] > 365 * 3, "segment"] = "inactive"
customers_2015["segment"] = customers_2015["segment"].fillna("NA")
customers_2015.loc[
    (customers_2015["recency"] <= 365 * 3) & (customers_2015["recency"] > 356 * 2),
    "segment",
] = "cold"
customers_2015.loc[
    (customers_2015["recency"] <= 365 * 2) & (customers_2015["recency"] > 365 * 1),
    "segment",
] = "warm"
customers_2015.loc[customers_2015["recency"] <= 365, "segment"] = "active"
customers_2015.loc[
    (customers_2015["segment"] == "warm")
    & (customers_2015["first_purchase"] <= 365 * 2),
    "segment",
] = "new warm"
customers_2015.loc[
    (customers_2015["segment"] == "warm") & (customers_2015["amount"] < 100), "segment"
] = "warm low value"
customers_2015.loc[
    (customers_2015["segment"] == "warm") & (customers_2015["amount"] >= 100), "segment"
] = "warm high value"
customers_2015.loc[
    (customers_2015["segment"] == "active") & (customers_2015["first_purchase"] <= 365),
    "segment",
] = "new active"
customers_2015.loc[
    (customers_2015["segment"] == "active") & (customers_2015["amount"] < 100),
    "segment",
] = "active low value"
customers_2015.loc[
    (customers_2015["segment"] == "active") & (customers_2015["amount"] >= 100),
    "segment",
] = "active high value"
```

Once we have the segments populated with corresponding customers we transform the type of the `segment` variable to categorical and reorder the segments, so that it makes more sense.

```python
# Transform segment column datatype from object to category
customers_2015["segment"] = customers_2015["segment"].astype("category")
# Re-order segments in a better readable way
sorter = [
    "inactive",
    "cold",
    "warm high value",
    "warm low value",
    "new warm",
    "active high value",
    "active low value",
    "new active",
]
customers_2015.segment.cat.set_categories(sorter, inplace=True)
customers_2015.sort_values(["segment"], inplace=True)
```

So let's plot the results as a treemap using `squarify`.

```python
plt.rcParams["figure.figsize"] = 12, 9
c2015 = pd.DataFrame(customers_2015["segment"].value_counts())
norm = mpl.colors.Normalize(vmin=min(c2015["segment"]), vmax=max(c2015["segment"]))
colors = [mpl.cm.RdYlBu_r(norm(value)) for value in c2015["segment"]]
squarify.plot(sizes=c2015["segment"], label=c2015.index, 
              value=c2015.values, color=colors, alpha=0.6)
plt.axis("off");
```

{% include image.html url="/assets/8/c2015.png" description="Market Segments as a Treemap" zoom="100%" %}

We will also need to compute the revenue of each customer in 2015 to be able to predict the amount of the next purchase. 

```python
# Compute revenues generated by customers in 2015 using SQL
q = """
        SELECT customer_id, 
        SUM(purchase_amount) AS 'revenue_2015'
        FROM df
        WHERE year_of_purchase = 2015
        GROUP BY 1
    """
revenue_2015 = sqldf(q)
```

Note that not all customers have made a purchase in 2015.

```python
print('Number of customers with at least one completed payment:', revenue_2015.shape[0])
print('Total number of customers as in 2015:', customers_2015.shape[0])
```

```tex
Number of customers with at least one completed payment: 5398
Total number of customers as in 2015: 18417
```

`revenue_2015` contains only 5,398 observations, while we have 18,417 customers to date. This is because a large portion of our customers were actually inactive in 2015. Nevertheless, we need to include them in our statistics. We do this as we merge the variable `revenue_2015` with the dataframe `customers_2015`. In `SQL` syntax, this corresponds to a left join with `customers_2015` as left table.  Note that this only works if both dataframes have one common index: `Customer_ID`.

{% include image.html url="/assets/8/merge_.png" description="Different Types of SQL JOINs (Source: https://www.w3schools.com)" zoom="60%" %}


```python
# Merge 2015 customers and 2015 revenue, while making sure that customers
# without purchase in 2015 also appear in the new df
actual = customers_2015.merge(revenue_2015, how="left", on="customer_id")
```

And of course we don't want to have `NaN`'s in our dataframe.

```python
# Replace NaNs in revenue_2015 column with 0s
actual["revenue_2015"].fillna(0, inplace=True)
```

Now we want to compute the average revenue generated by each customer segment. As a manager you are interested in knowing to what extent each segment today contributes to today's revenues and this for many obvious reasons. Often, you will find that small segments, in terms of number of customers, generate a significant amount of revenue. Precisely this group of customers is what you want to target in your next marketing campaign.

```python
actual[["revenue_2015", "segment"]].groupby("segment").mean()
```

```tex
+-------------------+----------------+
| segment           |   revenue_2015 |
|-------------------+----------------|
| inactive          |         0      |
| cold              |         0      |
| warm high value   |         0      |
| warm low value    |         0      |
| new warm          |         0      |
| active high value |       323.569  |
| active low value  |        52.306  |
| new active        |        79.1661 |
+-------------------+----------------+
```

As you can see from above, `active high value` customers generated the highest revenue in 2015. Although they represent only 3% of all customers, they have generated more than 71% of total revenues.  
As far as resource allocation and how much money you want to invest to maintain good relationships with certain customers, based on what segment they belong to, such information is crucial.  
And in case you're wondering why there are so many zeros. It's because, by definition, only customers who have purchased at least once in the last 365 days are considered `active`.

## Segmenting a Database Retrospectively

From a business strategy perspective, it is also extremely useful to determine not only the extent to which each segment is contributing to today's revenues, but also to what extent each segment today would likely contribute to tomorrow's revenues. Only then can we achieve a forward-looking analysis of revenue development; namely: from which of today's customers will tomorrow's revenue come?  
Unfortunately, we cannot answer this question with certainty, as tomorrow has not yet happened. Only future can tell. Nevertheless, if we examine the recent past, we can have a pretty good idea of what tomorrow will be like; just ask the weather experts about it. After all, customers in a segment today are likely to behave pretty much the same as customers in that same segment did a year ago. So analyzing the past will inform us about the most likely future.
Now let's find out how this can be implemented.  

We want to do an RFM analysis and a customer segmentation as if we were a year in the past, namely 2014. Thus, every transaction that has happened in the last 365 days should be ignored, as if it never happened; or hasn't happened yet. The SQL query here is as follows:

```python
q = """
        SELECT customer_id,
        MIN(days_since) - 365 AS 'recency',
        MAX(days_since) - 365 AS 'first_purchase',
        COUNT(*) AS 'frequency',
        AVG(purchase_amount) AS 'amount'
        FROM df
        WHERE days_since > 365
        GROUP BY 1"""

customers_2014 = sqldf(q)
```

The rest, besides the new dataframe name, is the same as in the previous section.  

```python
customers_2014.loc[customers_2014["recency"] > 365 * 3, "segment"] = "inactive"
customers_2014["segment"] = customers_2014["segment"].fillna("NA")
customers_2014.loc[
    (customers_2014["recency"] <= 365 * 3) & (customers_2014["recency"] > 356 * 2),
    "segment",
] = "cold"
customers_2014.loc[
    (customers_2014["recency"] <= 365 * 2) & (customers_2014["recency"] > 365 * 1),
    "segment",
] = "warm"
customers_2014.loc[customers_2014["recency"] <= 365, "segment"] = "active"
customers_2014.loc[
    (customers_2014["segment"] == "warm")
    & (customers_2014["first_purchase"] <= 365 * 2),
    "segment",
] = "new warm"
customers_2014.loc[
    (customers_2014["segment"] == "warm") & (customers_2014["amount"] < 100), "segment"
] = "warm low value"

customers_2014.loc[
    (customers_2014["segment"] == "warm") & (customers_2014["amount"] >= 100), "segment"
] = "warm high value"
customers_2014.loc[
    (customers_2014["segment"] == "active") & (customers_2014["first_purchase"] <= 365),
    "segment",
] = "new active"

customers_2014.loc[
    (customers_2014["segment"] == "active") & (customers_2014["amount"] < 100),
    "segment",
] = "active low value"

customers_2014.loc[
    (customers_2014["segment"] == "active") & (customers_2014["amount"] >= 100),
    "segment",
] = "active high value"

# Transform segment column datatype from object to category
customers_2014["segment"] = customers_2014["segment"].astype("category")

# Re-order segments in a better readable way
sorter = [
    "inactive",
    "cold",
    "warm high value",
    "warm low value",
    "new warm",
    "active high value",
    "active low value",
    "new active",
]
customers_2014.segment.cat.set_categories(sorter, inplace=True)
customers_2014.sort_values(["segment"], inplace=True)
```

Now let's plot the results as a Treemap.

```python
c2014 = pd.DataFrame(customers_2014["segment"].value_counts())

plt.rcParams["figure.figsize"] = 12, 9
norm = mpl.colors.Normalize(vmin=min(c2014["segment"]), vmax=max(c2014["segment"]))
colors = [mpl.cm.Spectral(norm(value)) for value in c2014["segment"]]
squarify.plot(sizes=c2014["segment"], label=c2014.index, 
              value=c2014.values, color=colors, alpha=0.6)
plt.axis("off");
```

{% include image.html url="/assets/8/c2014.png" description="Market Segments from 2014" zoom="100%" %}

As we can see the figures have changed over the last year. Our new customers segment increased by more than 20% and inactive customers have drastically increased through the years so that half of our customer database is now inactive.

```python
# The number of "new active high" customers has increased between 2014 and 2015.
# What is the rate of that increase?

ne = (customers_2015["segment"] == "active high value").sum()
ol = (customers_2014["segment"] == "active high value").sum()

print("New customers growth rate: %.0f%%" % round((ne - ol) / ol * 100))
```

```python
New customers growth rate: 21%
```

Conducting a horizontal analysis such as the one addressed above is extremely valuable when it comes to gaining a more accurate and well-founded understanding of the evolution and challenges the company has been facing over the years. It enables managers to identify trends and growth patterns, determine the product lifecycle stage the product is currently in, as well as predict what the future may look like.  

## Building a Predictive Model

Forecasting involves building a statistical model by analyzing past and present data trends to predict future behaviors or to suggest actions to take for optimal outcomes. In this example, we decided to use a simple linear regression approach, using the customers' RFM from 2014 as predictors and the revenue generated by those same customers in 2015 as target variables.

We first start by merging both dataframes: `customers_2014` with `revenue_2015` while making sure that new customers from 2015 are not included. We are going to call this new dataframe `in_sample`; meaning predictors and targets are already known.

```python
# Merge 2014 customers and 2015 revenue
in_sample = customers_2014.merge(revenue_2015, how="left", on="customer_id")
# Transform NaN's to 0
in_sample["revenue_2015"].fillna(0, inplace=True)
```

Then we create a new variable `active_2015`, which indicates if a client made a purchase in 2015 or not and we set its type as `int`.

```python
in_sample.loc[in_sample["revenue_2015"] > 0, "active_2015"] = 1
in_sample["active_2015"] = in_sample["active_2015"].astype("int")
in_sample.info()
```

```tex
<class 'pandas.core.frame.DataFrame'>
Int64Index: 16905 entries, 0 to 16904
Data columns (total 8 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   customer_id     16905 non-null  int64  
 1   recency         16905 non-null  int64  
 2   first_purchase  16905 non-null  int64  
 3   frequency       16905 non-null  int64  
 4   avg_amount      16905 non-null  float64
 5   max_amount      16905 non-null  float64
 6   revenue_2015    16905 non-null  float64
 7   active_2015     16905 non-null  int64  
dtypes: float64(3), int64(5)
memory usage: 1.2 MB
```

### The Likelihood a Customer will be Active in 2015

A powerful and relatively simple technique for calculating the probability of an event is logistic regression. In this section we are going to see how to  use logistic regression to predict the likelihood a customer is going to be active in 2015. For that we are going to use `statsmodels` package, so make sure that you have it installed.






