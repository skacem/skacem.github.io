---
layout: post
comments: true
title: "Customer Lifetime Value Prediction Using RFM-Analysis"
excerpt: " The transactional approach to business and marketing leads most managers to focus on the wrong thing - the next transaction. So they become fixated with marketing mix and core products. Performance is then measured by metrics such as conversion rate, cost per acquisition, sales growth, or market share, and they forget what matters most: the customer. 
In fact, there is nothing more important, nothing more fundamental to our business than a long-term relationship with our high-value customers. So marketing becomes an interaction aimed at building, maintaining and improving those relationships. And at the heart of customer relationship lies Customer Lifetime Value (CLV). 
In this tutorial, we'll learn how  to forecast CLV in a non-contractual setting based on RFM-Analysis and first-order Markov chain."
author: "Skander Kacem"
tags: 
    - Business Analytics
    - Tutorial
    - RFM Analysis
    - CLV
katex: true
preview_pic: /assets/9/clv.png
---

## Introduction

If there's one thing you should be doing in your business right now, and I'm willing to bet you're not, it's Customer lifetime value (CLV) analysis.  

CLV is the present value of the future cash flows or payments a company will realize from a customer over the entire relationship. It helps businesses understand how to best use their marketing budget and strategies to grow their customer base. For instance, by understanding CLV, a business can determine how much it can afford to spend to acquire a new customer, identify and target future top customers or minimize spending for unprofitable customers.
In fact, whenever a business makes a decision about a customer relationship, it wants to make sure that the decision is going to be profitable in the long run.  
In this article, we will demonstrate how to use Python to implement a CLV forecasting model. It's important to note that we assume a non-contractual business environment. In a contract-based business, customers are typically locked into long-term agreements, where they pay the same fee on a recurring basis, making the CLV relatively easy to estimate. However, in a non-contractual environment, where customers can come and go as they please, estimating variable future revenue streams in terms of time and amount can be more challenging.
Fortunately, we've already addressed similar issues in previous articles and know that nothing beats the holy RFM trinity when it comes to predicting customer behavior. If you are unfamiliar with this model, take a moment to read my previous articles on RFM-analysis (recency, frequency, monetary value).  

But before I get into the code, I'd like to briefly introduce some keywords I've been using throughout the article. Those who are simply interested in programming can skip the next parts and go straight to xx.

 

## References

[1] Preview pic designed by pch.vector / [Freepik][http://www.freepik.com]