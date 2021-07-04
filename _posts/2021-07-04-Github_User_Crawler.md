---
layout: post
comments: true
title: "Github Crawler"
excerpt: "In the previous post, we learned the basics of web crawling and developed our first one-page crawler. In this post, we implement something more fun and challenging. Something that every Github user could use: a Github Users Crawler! How does it sound? 
Disclaimer: This project is intended for Educational Purposes ONLY."
author: "Skander Kacem"
tags: 
  - Python
  - Web Crawler
  - BeautifulSoup
katex: true
preview_pic: /assets/0/scraping-gh.png

---

This project is organized in two sections:  

1. Importing followers or "followings" of a given user.
2. Extracting some data from each imported user.

In the first section, we will crawl my own Github page to import the users we intend to parse. Because I personally have just three followers on Github, I'm using my following list as a reference, which is about 70 users. In the second part, we extract the necessary data from each user on that list.  
So here we go.

## 1. Import a List of Users

  The first thing we want to do, is importing the required modules for our code. The usual four: `pandas`, `numpy`, `requests` and `bs4`:

```python
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import numpy as np
```  

Then we want to define our global constants. Those are usually written in capital letters.  

```python
# In order to simulate a browser's user agent
HEADER = {
  'user-agent': 
  'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36',
  'Cache-Control':'no-cache'
}
# and The info we would like to extract from each user
# We are going to save them as a pandas DataFrame object.
COLUMNS  =  [ 'Name', 'Nickname', 'City', 'Work', 
              'Followers', 'Following', 'Likes', 
              'Repos', 'Contributions' ]

```  

Now we can start drafting a function that returns a list of followers from a specified Github page. We need to keep in mind that a maximum of 50 users will be displayed per page. This means that in some cases we will have to iterate through more than one page, depending on the number of the corresponding followers:

```python

def get_followers(user, what='followers'):
    
    url = 'https://github.com/{}?page={}&tab={}'
    results = []
    i = 0
    while True:
        # page number
        i+=1
        # link to the html of a given page number
        link = url.format(user, i, what)
        html = requests.get(link, headers=HEADER, timeout=10).text
        
        # Since there are 50 users per page
        # we need to make sure that we got them all
        if('You’ve reached the end of' in html):
            # we reached an empty page
            break
            
        soup = bs(html, 'lxml')
        nicknames = soup.find_all('span', class_='Link--secondary')
        for nickname in nicknames:
            results.append(nickname.text)
        
    return results
```  

Since I only have three followers, I will be importing the list of my following in here. The good thing about the HTML source code of Github is that it is a pretty clean piece of code. Which remains an exception, because the web in general is a messy place.  
To do this, we simply replace `followers` in the URL with `following`:

```python
def get_followings(user):
    return get_followers(user, what='following')
```
Now let's call the above function and check if everything is working.

```python
users = get_followings('skacem')
print(len(users))
```
    67

Yes, the number of users I follow is correct.  All right!
With that, the first part of the project is done. Now let's treat ourselves with a coffee break and then resume with second part of our cool project. :)

## 2. Extract Users Information

The first part was quite straightforward. We only had to check the URL and the CSS class that render the users.  
The second part of the project is a bit more challenging. We need to determine the CSS class of each chunk of data we plan to extract. This is usually done using the browser's Devtools. It takes trial and error to find the DOM path to the specific element in question. Usually there are different paths to target the very same element, the main rules here are "keep it simple" and "practicality beats purity".

Note, a missing information might disrupt the normal flow of our program and cause it to terminate abruptly. To avoid this, we catch such an exception and assign `np.NaN`<sup id='n1'>[1](#note1)</sup> to the corresponding variable.  
That being said, we can start writing a method to extract the needed information of a given user:

```python
def extract_info(user):
    # create bs object
    user_gh = url.format(user)
    html = requests.get(user_gh, headers=HEADER, timeout=10)
    soup = bs(html.text, 'lxml')
    
    # extract info
    try :
        full_name = soup.find('span', attrs={'itemprop':'name'}).text.strip()
    except:
        full_name = np.NAN
    
    try:
        city = soup.find('span', class_='p-label').text.strip()
    except:
        city = np.NAN
    try:
        work = soup.find('span', class_='p-org').text.strip()
    except:
        work = np.NAN
    try:
        repos = soup.find('span', class_='Counter').text.strip()
    except:
        repos = np.NAN
    try:
        contributions = soup.find('h2', class_='f4 text-normal mb-2').text.split()[0]
    except:
        contributions = np.NAN
        
    numbers = soup.find_all('span', class_='text-bold color-text-primary')
    try:
        followers = numbers[0].text
    except:
        followers = np.NAN
    
    try:
        following = numbers[1].text
    except:
        following = np.NAN
    
    try:
        likes = numbers[2].text
    except:
        likes = np.NAN
        
        
    return [full_name, user, city, work, followers, following, likes, repos, contributions]
```

As output we get a list with a total of nine features, some of which could be 'NaN' - well, certainly not the users alias.

Finally, we call the `extract_info()` function for all users in a `for-loop` and then save the output as a `csv` file.

```python
# crawl all users and scarape the required information
for user in users:
  result = extract_info(user)
  results.append(result)
df = pd.DataFrame(results, columns=COLUMNS)
df.to_csv('github_users_info.csv')
```
Let's check the result:

```python
# display 5 random users
df.sample(5)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th> Name </th>
      <th> Nickname </th>
      <th> City </th>
      <th> Work </th>
      <th> Followers </th>
      <th> Following </th>
      <th> Likes </th>
      <th> Repos </th>
      <th> Contributions [2021] </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>Alexandre Sanlim</td>
      <td>alexandresanlim</td>
      <td>Curitiba - PR, Brazil</td>
      <td>@Avanade</td>
      <td>281</td>
      <td>77</td>
      <td>254</td>
      <td>31</td>
      <td>3,020</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Johannes Gontrum</td>
      <td>jgontrum</td>
      <td>Uppsala, Sweden</td>
      <td>NaN</td>
      <td>50</td>
      <td>34</td>
      <td>275</td>
      <td>72</td>
      <td>113</td>
    </tr>
    <tr>
      <th>25</th>
      <td></td>
      <td>vxunderground</td>
      <td>International</td>
      <td>NaN</td>
      <td>1.6k</td>
      <td>0</td>
      <td>10</td>
      <td>4</td>
      <td>1,007</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rahul Dave</td>
      <td>rahuldave</td>
      <td>Somerville, MA</td>
      <td>Harvard University/univ.ai</td>
      <td>334</td>
      <td>1</td>
      <td>31</td>
      <td>121</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pierian Data</td>
      <td>Pierian-Data</td>
      <td>Las Vegas, Nevada</td>
      <td>Pierian Data Inc.</td>
      <td>6k</td>
      <td>0</td>
      <td>1</td>
      <td>18</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>

We made it.  It wasn't hard, was it? I hope you enjoyed it!  
In case you are interested in experimenting more with the program, a more usable Python program that summarizes everything we've done here can be found at this [repo](https://github.com/skacem/TIL/blob/3e4eaf87fb3dc34a45164cbb17d638a9e2ec31d2/Python/github_crawler.py).

---
<a name="note1">1</a>: NaN is used as a placeholder for missing data consistently in pandas. No matter if it is a float, integer or string. And we want to save our results as a `pandas.DataFrame`.[↩](#n1)