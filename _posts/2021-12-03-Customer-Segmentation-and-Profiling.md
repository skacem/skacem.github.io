---
layout: post
comments: true
title: "Customer Segmentation and Profiling: A Managerial Approach"
excerpt: "In the last tutorial, we presented how to segment customers databases using hierarchical cluster analysis. The approach is quite simple and does not require any parameters other than the number of clusters we want to obtain. However, in practice, this method is not very efficient, as we have no control over how the clusters are formed.  And what is the point of segmentation, if we do not understand how the segments differ or if we cannot treat each market segment appropriately. In this tutorial, we will develop a non-statistical segmentation also known as managerial segmentation."
author: "Skander Kacem"
tags: 
    - Business Analytics
    - Tutorial
    - Managerial Segmentation
    - Customer Segmentation
katex: true
preview_pic: /assets/6/customers.png
---

## Introduction

One of the most used and familiar concepts in marketing is market segmentation, that is, the division of markets, based on common characteristics such as geographic, demographic, psychographic and behavioristic, into homogeneous groups of customers, in order to develop segment-specific marketing strategies or to look for new product opportunities to effectively capture the target segments.  

In reality, there is no single best method for segmenting a market and, more importantly, the way we segment a market should reflect what we intend to achieve. Useful segments must, however, possess the following four qualities (Kottler 1980):

1. Measurability; in terms of the size and purchasing power of the segment.
2. Actionability, that is, the extent to which it is possible to design an effective marketing plan for the target segment  
3. Accessibility which means that your company's marketing activities and distribution system are capable of reaching and adequately serving the target segment. 
4. Substantiality; implying that the target segments are large enough to be profitable.

Broadly speaking, any market segmentation falls into one of these two approaches: *a priori* (or prescriptive) and *post hoc* (or exploratory) segmentation. Segmentation based on hierarchical cluster analysis, which we introduced in the last tutorial, is a typical example of exploratory post-hoc segmentation. In this tutorial, we will introduce and implement an a priori approach to segmentation, namely managerial market segmentation.


## Managerial Segmentation

Managerial segmentation is an *a priori* segmentation, and is performed when the manager proactively chooses the basis of the market analysis and the variables to be included in the segmentation. Hence, the basis for segmentation varies depending on the specific business decisions that management is facing. For instance, if the management is concerned with the likely impact of a price increase on its customers, the appropriate basis might be the current customers' price sensitivity. If, however the management is concerned with the loss of customers, the basis for segmentation could be recency.

In this tutorial, we want to predict which customers are most likely to make further purchases in the near future. A basic criterion for this would be whether the customer has made a purchase recently or not. After all, someone who bought at our store a few weeks ago is much more likely to buy again in the future than someone whose last purchase was several years ago.

For example we could divide our customer database into five groups or segments based on recency; as depicted below.

We refer to customers as **Active** if they have purchased something within the last 12 months. As **Warm** someone whose last purchase happened a year before that is between 13 and 24 months and for those who haven’t purchased anything for more than 3 years, we label them as **Inactive**
  
Now, given the scope and diversity of marketing decisions, attempting to use a single basis for segmentation to develop a marketing strategy is likely to result in adopting the wrong solutions and wasting resources. So let's add another variable to our segmentation model, namely average spending. By doing so, we want to differentiate between valuable and less valuable customers in each subgroup within **Warm** and **Active** customers. Furthermore, we want to differentiate between those who have only made one purchase so far, regardless of how much money they spend, and refer to them as **New Customers** (see image). 

IIMMAAGGEE

As a result, we end up with a total of eight segments. At this point, we can decide which groups to allocate our marketing budget to and how to target our marketing campaigns towards them.

## Customer Profiling

