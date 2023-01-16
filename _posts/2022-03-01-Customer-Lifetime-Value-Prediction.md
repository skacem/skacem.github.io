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

## From Product-Centric to Customer-Centric Marketing

Until recently, retailers have approached marketing from a product-oriented perspective, focusing mainly on the features, benefits, and prices of their products and services while ignoring the needs and wants of their customer base. The focus of this approach is on quick transactions, using  marketing mix and advertising campaigns to grab new consumers' attention and persuade them to make an impulse purchase. Sales are then treated as faceless transactions with the ultimate goal to increase conversion and short-term profits.  

However, in today's competitive business landscape, retailers are looking for new ways to differentiate themselves from their competitors. Those that adopt customer-centric marketing can provide tailored experiences that are engaging for their customer base. This requires understanding customers on an individual level and forecasting their CLV to determine how profitable they will be in a long-term relationship. After all, you don't want to waste your time and marketing budget on unprofitable customers.

Indeed, the transition to a more customer-centric marketing strategy among retailers comes from a combination of factors such as competitive pressures, changing consumer behavior, digital transformation and data availability.
With the advent of social media and e-commerce, customers have gained greater access to a variety of retailers offering the same or similar products and have become more knowledgeable in their purchasing decisions. They now expect organizations to understand and cater to their individual needs and wants, and have higher expectations for personalized and tailored value-added services and buying experiences.  
In addition, digital transformation and the availability of data have made it easier for retailers to gain insights into customers' behavior, preferences and concerns. This data can be used to create more targeted and personalized marketing campaigns and improve products and services. Customer relationship management (CRM) tools also enabled retailers to manage customer interactions more efficiently and effectively while improving overall customer satisfaction and retention.  

In summary, it is essential for organizations to adopt both product and customer-centric marketing strategies to remain competitive in saturated marketplaces.  
A customer-centric approach is important for building strong relationships with the customer base and fostering trust and loyalty, which can attract repeat purchases, increase word of mouth and prevent customers from turning to competitors. On the other hand, a product-centric approach is also important when it comes to attracting new customers and generating initial interest in a product or service.  
By using both strategies, retailers can effectively balance the needs of new customer acquisition and customer retention, ultimately leading to long-term success in the marketplace.

## CLV: Motivation and Use Cases

Imagine your company launches a big campaign to attract new customers. You invest in PR, advertising and email campaigns, buy a lot of online ads, create a special promotion and offer a big discount to first-time buyers. However, after all these efforts, the campaign results in only a small number of new customers and sales that don't even cover the cost of the campaign. Well, you could argue that these new customers will stay with you and continue to buy, making the campaign worthwhile in the long run. But wouldn't it be nice to have a way to measure whether or not that's true? That's where customer lifetime value (CLV) models come in. These models look at past and current customer behavior to predict how much revenue a customer will bring in over time. Only then can you determine if the cost of customer acquisition was worth it.  

CLV models offer a wide range of practical applications. For example, you might compare one acquisition campaign to another. Suppose you are comparing two campaigns, one with a substantial discount and one with a modest discount. The advertising campaign with the greater discount will likely attract more customers, but those customers will be less profitable in the short term and much less loyal in the long term once the discount is removed. After all, the only reason they came in the first place was to take advantage of the discount. Offering a small discount and bringing in fewer customers might be a better strategy, assuming of course, they are more loyal and bring in more money in the long run. But how can you be sure about that, if you can't quantify the long-term value of your customers.  
CLV is also very useful for existing customers. In fact, one of the best uses of Customer Lifetime Value is to determine which customers or customer segments are more valuable to the future growth of your business. Let's say you have two segments, with customers in segment one contributing an average of $100 per quarter to revenue. Customers in segment two generate an average of $150. At first glance, you might want to put a little more effort into satisfying and retaining the customers in segment two. Perhaps additional resources should be allocated. Make sure they have access to the best customer service possible. Keep an eye on loyalty, and so on. However, an analysis of their lifetime value shows that customers in the first segment are more loyal, stay longer, and regularly spend more money with the company. Customers in the second segment are simply seasonal and not loyal at all. They switch to the competition as soon as they discover a better offer or special promotion, and may only be one-time customers. Comparing lifetime values, customers in the first segment spend an average $5,000 during all they relationship with your company, while customers in segment two spend an average of only $600. Wouldn't it be wonderful if you knew how to calculate these numbers in the first place so you could focus your marketing resources on the most valuable customers over time? What do you think would happen if you didn't?

## CLV Computation

CLV can be tricky to compute. Some methods are too simplistic to be useful, while others are too complex. In this tutorial, we'll take a middle-of-the-road approach and use segmentation - a concept we've already covered in previous articles - to compute CLV.

You may recall that we talked about managerial segmentation, where you can assign customers to different segments based on their behaviors like recency, frequency, monetary value (RFM). We also discussed how you can use this segmentation not just on current data, but also retrospectively on past data. This provides  significant information into how your business is performing, which segments are increasing and on the account of what other segments, and whether you're attracting more new consumers than before.

Each segmentation is like a snapshot of your customer database. By taking multiple snapshots over time, you can see how customers move from one segment to another. For example, some high-value customers may stay in the same segment while others move to lower-value segments. By analyzing these transitions, you can predict how your customers will behave in the future.  
To make this easier, we will use a tool called a transition matrix also known as migration model. This is a mathematical representation of the transition probabilities of a Markov chain. It is a square matrix and in this case shows the probability of customers moving from one segment to another. Each row represents the current segment, while each column represents the segment customers were in a year ago.  The sum of each row should equal 1, which means that the probabilities of switching from one state to another are equal to 1. It's important to note that by definition, some transitions may have zero probability, such as a customer moving from active to inactive in just one period. This is normal and expected. 
Based on the analysis of this matrix, you can predict how customers will behave in the future.





## Glossary

* Transactional Marketing: is a one-time point of sale transaction based on the exchange of goods or services for money. It focuses on quick transactions, often using promotional campaigns to quickly capture consumers’ attention and get them to make an immediate purchase. Its primary goal is to increase short-term sales and profit.
* Relationship marketing
* Customer Relationship Management CRM
* 
## References

[1] Preview pic designed by pch.vector / [Freepik][http://www.freepik.com]