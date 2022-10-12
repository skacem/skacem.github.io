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

Business forecasting is not about accurately predicting the future, nor is it an end in itself. It is a means to help companies manage uncertainty and develop strategies to minimize potential risks and threats or maximize financial returns.  It relies on identifying and estimating past patterns and/or relationships that are then generalized to forecast the future. As such, forecasting has a wide range of business applications and can be useful throughout the value chain of any organization.  
For instance, forecasting customer purchases allows companies not only to calculate the expected revenue of each segment so they can focus on profitable ones, but also to properly plan inventory and manpower for the next period.  Other business forecasting applications include customer lifetime value, credit card attrition or future revenues. In other words, every business needs forecasting.

In this tutorial we will develop a typical marketing application. We will predict the likelihood that a customer will make a purchase in a near future. And if so, we also want to predict how much s/he will spend. For this, we will first segment our customer database using RFM analysis --I hope you are now familiar with this segmentation method-- and then we will build two distinct models:

1. A first model to calculate the probability of purchase
2. A second model to predict the amount spent  

Finally, we will combine the two models into one large scoring model.