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

One of the most used and familiar concepts in marketing is market segmentation, that is, the division of markets, based on common characteristics, into homogeneous groups of customers, in order to develop segment-specific marketing strategies to effectively capture the target segments. \\ 
The customer characteristics that can be applied to segmentation are widespread in the literature and generally seem to represent minor variations on one of these familiar themes: location, demographics, psychographics and customer buying behaviors. In reality, there is no single best method for segmenting a market and, more importantly, the way we segment a market should reflect what we intend to achieve. For simplicity sake, we will be using in what follows the RFM (recency, frequency, monetary) analysis introduced in the previous tutorial. 


## Managerial Requirements

Managerial segmentation is an *a priori* segmentation, and is performed when the manager proactively chooses the basis of the market analysis and the variables to be included in the segmentation. Hence, the basis for segmentation varies depending on the specific business decisions that management is facing. For instance, if the management is concerned with the likely impact of a price increase on its customers, the appropriate basis might be the current customers' price sensitivity. If, however the management is concerned with the loss of customers, the basis for segmentation could be recency.

For this tutorial, we want to predict which customers are most likely to make further purchases in the near future. A basic criterion would then be whether they have purchased recently or not; that is recency. After all, someone who bought at our store a few weeks ago is much more likely to buy again in the future than someone whose last purchase was several years ago.

## Segmentation Model

The following segmentation model is suitable for the stated managerial requirements:

IIMMAAGGE

As we can see from the above depicted model, our customer database is divided into 5 groups or segments based on recency:

    * We refer to customers as **active** if they have purchased something within the last 12 months
    * As **warm** someone whose last purchase happened a year before that is between 13 and 24 months
    * For those who haven’t purchased anything for more than 3 years, we label them as **inactive**
  
Given the scope and diversity of marketing decisions, attempting to use a single basis for segmentation to develop a marketing strategy is likely to result in adopting the wrong solutions and wasting resources. So let's add another variable to our segmentation model, namely average spending. By doing so, we want to differentiate between valuable and less valuable customers in each subgroup within "warm" and "active" customers. Furthermore, we want to differentiate between those who have only made one purchase so far, regardless of how much money they spend, and refer to them as **new customers** (see image). That is nothing else but an RFM-analysis.


IIMMAAGGEE


As a result, we end up with a total of eight segments. Now we can decide which groups to allocate our marketing budget to and how to target our marketing campaigns around them, so it can bring benefits our business.

## Customer Profiling

