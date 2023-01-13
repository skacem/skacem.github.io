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
In this article, we will demonstrate how to use Python to implement a CLV forecasting model. It's important to note that we assume a non-contractual business setting. In a contract-based business, customers are typically locked into long-term agreements, where they pay the same fee on a recurring basis,making it relatively easy to estimate CLV based on retention rate.  
However, in a non-contractual environment, where customers can come and go as they please, estimating variable future revenue streams in terms of time and amount can be more challenging.
Fortunately, we've already addressed similar issues in previous articles and know that nothing beats the RFM trinity when it comes to predicting customer behavior. So, if you are unfamiliar with RFM-analysis (recency, frequency, monetary value), take a moment to read my previous article.  

Now, before I get into the code, I'd like to briefly discuss the importance of customer-centric marketing in today's business environment. Those who are simply interested in programming can skip the next parts and go straight to xx.


## From Product-Centricity to Customer-Centric Marketing

Until recently, retailers have approached marketing from a product-oriented perspective, focusing mainly on the features, benefits, and prices of their products and services while ignoring the needs and wants of their customers. The focus of this approach is on quick transactions, using  marketing mix and advertising campaigns to grab consumers' attention and persuade them to make an impulse purchase. Sales are then treated as faceless transactions with the ultimate goal to increase conversion and short-term profits.  

However, in today's competitive business landscape and mature marketplaces, retailers are looking for new ways to differentiate themselves from their competitors. Those that adopt customer-centric marketing can provide tailored experiences that are engaging for their target customers. This requires understanding customers on an individual level and forecasting their CLV to determine how profitable they will be in a long-term relationship. After all, you don't want to waste your time and marketing budget on unprofitable customers.

Now besides the competitive 

    *  Today's businesses face overcrowded marketplaces and are looking for new ways to differentiate themselves from their competitors.  With a customer-centric marketing, companies can create a strong value proposition that differentiates them from their peers.
    *  With the rise of digital marketing, companies now have access to vast amounts of data on their customers, which they can use to gain insights. This has enabled companies to create more personalized marketing campaigns and to build stronger relationships with their customers.



## Glossary

* Transactional Marketing: is a one-time point of sale transaction based on the exchange of goods or services for money. It focuses on quick transactions, often using promotional campaigns to quickly capture consumers’ attention and get them to make an immediate purchase. Its primary goal is to increase short-term sales and profit.
* Relationship marketing
* Customer Relationship Management CRM
* 
## References

[1] Preview pic designed by pch.vector / [Freepik][http://www.freepik.com]