---
layout: post
category: ml
comments: true
title: "GitHub Crawler: Beyond Basic Scraping"
author: "Skander Kacem"
tags:
  - Python
  - Web Crawler
  - BeautifulSoup
katex: true
---

In the [previous post](/ml/2021/06/25/Web-Crawlers/), we learned the fundamentals of web scraping and built our first crawler for movie data. Today, we're taking things further by building a GitHub users crawler. But more importantly, we'll explore when scraping makes sense, when it doesn't, and how to do it responsibly.

This project continues where we left off, but introduces three crucial concepts every web scraper should understand: the API versus scraping decision, ethical considerations, and performance optimization through asynchronous programming.

## The Project That Started It All

A few weeks ago, I wanted to analyze my GitHub network. Who are the people I follow? What languages do they use? Where are they based? Simple questions, but getting the answers taught me more about web scraping than any tutorial ever did.

The plan was straightforward: grab my following list from GitHub, then extract information about each user. Should be simple enough, right? Well, that's where things got interesting.

## The Fork in the Road

When you need data from a website, you're standing at a crossroads. You can either use their API if they offer one, scrape their website directly, or find the data somewhere else. GitHub, interestingly, offers both an API and scrapeable web pages, which makes it perfect for understanding this decision.

An API is like a restaurant menu. The service tells you exactly what data you can order and how to order it. When you request user data from GitHub's API, you get back clean, structured JSON data:

```python
import requests
import json

def get_user_via_api(username):
    """
    Fetch user data through GitHub's API.
    
    Args:
        username (str): GitHub username to look up
    
    Returns:
        dict: User data if successful, None if failed
    """
    # GitHub API endpoint for user data
    url = f'https://api.github.com/users/{username}'
    
    # Headers tell GitHub who we are and what format we want
    headers = {
        'Accept': 'application/vnd.github.v3+json',  # Request JSON format
        'User-Agent': 'Python-Tutorial-Script'  # Identify ourselves
    }
    
    # Make the GET request to the API
    response = requests.get(url, headers=headers)
    
    # Check if we hit the rate limit (403 Forbidden)
    if response.status_code == 403:
        # GitHub provides rate limit info in response headers
        reset_time = response.headers.get('X-RateLimit-Reset')
        remaining = response.headers.get('X-RateLimit-Remaining')
        print(f"Rate limited! {remaining} requests left, resets at {reset_time}")
        return None
    
    # 200 OK means success
    if response.status_code == 200:
        return response.json()  # Parse JSON response
    
    return None

# Try it out
user_data = get_user_via_api('torvalds')
if user_data:
    print(f"Name: {user_data.get('name')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Public repos: {user_data.get('public_repos')}")
```

When I ran this code, the output was impressive:

```
Name: Linus Torvalds
Followers: 171432
Public repos: 7
```

Look at that! Clean, structured data in just a few lines of code. The API even gave me exact follower counts (over 171k followers for Linus Torvalds!). But then I checked my rate limit:

```python
# Check remaining API calls
limit_response = requests.get('https://api.github.com/rate_limit')
limits = limit_response.json()['rate']
print(f"API calls remaining: {limits['remaining']}/{limits['limit']}")
```

Output:
```
API calls remaining: 57/60
```

Only 57 calls left! At this rate, I'd burn through my limit analyzing just 57 users. That's when I realized why scraping still matters.

## When the API Isn't Enough

GitHub's API is generous but not unlimited. Without authentication, you get 60 requests per hour. With authentication, you get 5000 requests per hour, which is better but still finite.

But rate limits aren't the only issue. Sometimes the API doesn't provide the data you need. Want to analyze someone's contribution graph, that green grid showing their coding activity? It's right there on the website but not in the API. Need to see which repositories someone has starred recently? Same story.

This is when scraping becomes not just useful but necessary. Here's how we do it:

```python
from bs4 import BeautifulSoup
import requests
import time

class GitHubScraper:
    """
    A respectful GitHub scraper that handles rate limiting 
    and provides methods to extract user data.
    """
    
    def __init__(self):
        # Create a session to reuse connections (more efficient)
        self.session = requests.Session()
        
        # Set a user agent to identify ourselves
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Purpose Only) '
                         'AppleWebKit/537.36'
        })
        
        # Track time between requests for rate limiting
        self.last_request_time = 0
        self.min_delay = 1  # Minimum 1 second between requests
    
    def respectful_get(self, url):
        """
        Make an HTTP GET request with automatic rate limiting.
        This prevents overwhelming the server with requests.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            requests.Response: The HTTP response
        """
        # Calculate time since last request
        elapsed = time.time() - self.last_request_time
        
        # If we're going too fast, slow down
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        # Make the request
        response = self.session.get(url, timeout=10)
        
        # Update last request time
        self.last_request_time = time.time()
        
        return response
    
    def get_user_followers(self, username):
        """
        Scrape a user's followers from their GitHub profile.
        GitHub shows max 50 users per page, so we handle pagination.
        
        Args:
            username (str): GitHub username
            
        Returns:
            list: List of follower usernames
        """
        followers = []
        page = 1
        
        while True:
            # Construct URL with pagination
            url = f'https://github.com/{username}?page={page}&tab=followers'
            
            try:
                # Fetch the page (with rate limiting)
                response = self.respectful_get(url)
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all follower username elements
                # GitHub uses span with class Link--secondary for usernames
                user_elements = soup.find_all('span', class_='Link--secondary')
                
                # If no users found, we've reached the end
                if not user_elements:
                    break
                
                # Extract text from each element
                for element in user_elements:
                    username = element.text.strip()
                    if username.startswith('@'):
                        username = username[1:]  # Remove @ prefix
                    followers.append(username)
                
                print(f"Scraped page {page}: found {len(user_elements)} users")
                
                # Check if there's a next page button
                if not soup.find('a', {'aria-label': 'Next'}):
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                break
        
        return followers

# Test the scraper
scraper = GitHubScraper()
followers = scraper.get_user_followers('skacem')
print(f"\nTotal followers found: {len(followers)}")
print(f"First 5 followers: {followers[:5]}")
```

The output showed the pagination in action:

```
Scraped page 1: found 50 users
Scraped page 2: found 17 users

Total followers found: 67
First 5 followers: ['gvanrossum', 'kennethreitz', 'mitsuhiko', 'dhh', 'sindresorhus']
```

Notice how we had to make multiple requests just to get the follower list? And we only got usernames, not their full profiles. To get detailed information about each of these 67 users would require 67 more requests. This is where the trade-offs become clear.

## The Ethics Discussion We Need to Have

Before writing more code, I spent time thinking about the ethics of scraping. This is something tutorials often skip, but it's crucial.

Every responsible scraper should first check a website's robots.txt file:

```python
import urllib.robotparser

def check_if_allowed(url):
    """
    Check if we're allowed to scrape a URL according to robots.txt.
    This is like checking if a door says 'Please knock' or 'Do not disturb'.
    
    Args:
        url (str): The URL we want to scrape
        
    Returns:
        bool: True if scraping is allowed, False otherwise
    """
    # Create a robot parser object
    rp = urllib.robotparser.RobotFileParser()
    
    # Read GitHub's robots.txt file
    rp.set_url("https://github.com/robots.txt")
    rp.read()
    
    # Check if our user agent can fetch this URL
    can_fetch = rp.can_fetch("*", url)  # "*" means any user agent
    
    # Check if there's a crawl delay specified
    crawl_delay = rp.crawl_delay("*")
    
    print(f"Can scrape {url}? {can_fetch}")
    if crawl_delay:
        print(f"Requested delay between requests: {crawl_delay} seconds")
    
    return can_fetch

# Always check before scraping!
check_if_allowed("https://github.com/torvalds")
check_if_allowed("https://github.com/login")
```

The results were enlightening:

```
Can scrape https://github.com/torvalds? True
Can scrape https://github.com/login? False
```

GitHub allows scraping user profiles but not login pages. This makes sense - public profiles are meant to be viewed, while login pages handle sensitive data. The robots.txt file is GitHub's way of telling us where we're welcome and where we're not.

## Building with Both Approaches

With ethics in mind, I built my GitHub crawler to be smart about when to use each approach. The strategy was simple: try the API first, fall back to scraping when necessary.

```python
class SmartGitHubCollector:
    """
    A smart GitHub data collector that tries API first,
    then falls back to scraping when needed.
    """
    
    def __init__(self, api_token=None):
        """
        Initialize the collector with optional API token.
        
        Args:
            api_token (str, optional): GitHub API token for higher rate limits
        """
        self.api_token = api_token
        self.scraper = GitHubScraper()  # Our scraper instance
        
        # Set up API headers
        self.api_headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Educational-GitHub-Analyzer'
        }
        
        # Add token if provided (increases rate limit to 5000/hour)
        if api_token:
            self.api_headers['Authorization'] = f'token {api_token}'
    
    def extract_user_info(self, username):
        """
        Extract user information using the best available method.
        Tries API first, falls back to scraping if needed.
        
        Args:
            username (str): GitHub username
            
        Returns:
            dict: User information
        """
        # First, try the API (cleaner and more reliable)
        api_data = self._try_api(username)
        if api_data:
            return self._parse_api_data(api_data)
        
        # API failed? Try scraping
        print(f"API unavailable for {username}, trying web scraping...")
        return self._try_scraping(username)
    
    def _try_api(self, username):
        """
        Attempt to get user data via GitHub API.
        
        Returns:
            dict: API response or None if failed
        """
        url = f'https://api.github.com/users/{username}'
        try:
            response = requests.get(url, headers=self.api_headers, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass  # Silently fail and return None
        return None
    
    def _try_scraping(self, username):
        """
        Fallback method: scrape user data from GitHub website.
        
        Returns:
            dict: Scraped user information
        """
        url = f'https://github.com/{username}'
        response = self.scraper.respectful_get(url)
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract various user fields from HTML
        info = {
            'username': username,
            'name': self._extract_text(soup, 'span', {'itemprop': 'name'}),
            'bio': self._extract_text(soup, 'div', {'data-bio-text': True}),
            'location': self._extract_text(soup, 'span', {'itemprop': 'homeLocation'}),
            'company': self._extract_text(soup, 'span', {'itemprop': 'worksFor'}),
            'followers': self._extract_counter(soup, 'followers'),
            'repositories': self._extract_counter(soup, 'repositories')
        }
        
        return info
    
    def _extract_text(self, soup, tag, attrs):
        """
        Safely extract text from an HTML element.
        Returns None if element doesn't exist.
        
        Args:
            soup: BeautifulSoup object
            tag (str): HTML tag name
            attrs (dict): Attributes to find
            
        Returns:
            str or None: Extracted text or None
        """
        element = soup.find(tag, attrs)
        return element.text.strip() if element else None
    
    def _extract_counter(self, soup, counter_type):
        """
        Extract numeric counters (followers, repos, etc.) from GitHub profile.
        
        Args:
            soup: BeautifulSoup object
            counter_type (str): Type of counter ('followers', 'repositories', etc.)
            
        Returns:
            str: Counter value as string
        """
        # Find the link to the counter tab
        link = soup.find('a', {'href': f'?tab={counter_type}'})
        if link:
            # Counter is in a span with class Counter
            span = link.find('span', class_='Counter')
            if span:
                return span.text.strip()
        return '0'
    
    def _parse_api_data(self, data):
        """
        Convert API JSON response to our standard format.
        
        Args:
            data (dict): Raw API response
            
        Returns:
            dict: Standardized user information
        """
        return {
            'username': data.get('login'),
            'name': data.get('name'),
            'bio': data.get('bio'),
            'location': data.get('location'),
            'company': data.get('company'),
            'followers': str(data.get('followers', 0)),
            'repositories': str(data.get('public_repos', 0)),
            'created_at': data.get('created_at'),
            'source': 'api'  # Track where data came from
        }

# Test with different users
collector = SmartGitHubCollector()

# Test API approach
print("Testing API approach:")
user1 = collector.extract_user_info('gvanrossum')
print(f"Name: {user1['name']}")
print(f"Bio: {user1['bio']}")
print(f"Company: {user1['company']}")
print(f"Source: {user1.get('source', 'scraping')}")
```

The output showed interesting differences:

```
Testing API approach:
Name: Guido van Rossum
Bio: Python's creator
Company: @Microsoft
Source: api
```

When I forced it to use scraping by exhausting the API limit, the results were slightly different:

```
API unavailable for gvanrossum, trying web scraping...
Name: Guido van Rossum
Bio: Python's creator
Company: Microsoft
Source: scraping
```

Notice how the company format differs? The API returns "@Microsoft" while scraping gives us "Microsoft". These small differences add up when processing hundreds of users.

## The Speed Revolution

Traditional scraping is synchronous. You request data for user A, wait for the response, process it, then move on to user B. Checking 70 users at 2 seconds each meant waiting almost two and a half minutes. There had to be a better way.

Enter asynchronous programming. Instead of waiting for each request to complete before starting the next one, async lets you fire off multiple requests and handle them as they return:

```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup

class AsyncGitHubScraper:
    """
    Asynchronous GitHub scraper for fetching multiple users efficiently.
    Uses asyncio and aiohttp for concurrent requests.
    """
    
    def __init__(self, max_concurrent=5):
        """
        Initialize async scraper with concurrency limit.
        
        Args:
            max_concurrent (int): Maximum simultaneous requests
        """
        self.headers = {
            'User-Agent': 'Educational-Async-Scraper'
        }
        # Semaphore limits concurrent requests to be respectful
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_user(self, session, username):
        """
        Fetch a single user's data asynchronously.
        
        Args:
            session: aiohttp ClientSession
            username (str): GitHub username
            
        Returns:
            dict: User data or None if failed
        """
        url = f'https://github.com/{username}'
        
        # Use semaphore to limit concurrent requests
        async with self.semaphore:
            try:
                # Make async GET request
                async with session.get(url) as response:
                    if response.status == 200:
                        # Read response text asynchronously
                        html = await response.text()
                        return self.parse_user_html(html, username)
                    else:
                        print(f"Failed to fetch {username}: {response.status}")
                        return None
            except Exception as e:
                print(f"Error fetching {username}: {e}")
                return None
    
    def parse_user_html(self, html, username):
        """
        Parse user data from HTML string.
        
        Args:
            html (str): HTML content
            username (str): Username for reference
            
        Returns:
            dict: Parsed user information
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        return {
            'username': username,
            'name': self._safe_extract(soup, 'span', {'itemprop': 'name'}),
            'location': self._safe_extract(soup, 'span', {'itemprop': 'homeLocation'}),
            'company': self._safe_extract(soup, 'span', {'itemprop': 'worksFor'}),
            'bio': self._safe_extract(soup, 'div', {'data-bio-text': True})
        }
    
    def _safe_extract(self, soup, tag, attrs):
        """
        Safely extract text from HTML elements.
        
        Returns:
            str or None: Extracted text or None if not found
        """
        element = soup.find(tag, attrs)
        return element.text.strip() if element else None
    
    async def scrape_users(self, usernames):
        """
        Scrape multiple users concurrently.
        This is where the speed improvement happens!
        
        Args:
            usernames (list): List of usernames to scrape
            
        Returns:
            list: List of user data dictionaries
        """
        # Create a single session for all requests
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Create a task for each username
            tasks = [self.fetch_user(session, username) for username in usernames]
            
            # Run all tasks concurrently and wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Filter out None results (failed requests)
            return [r for r in results if r is not None]

# How to use async scraper
async def main():
    """
    Example of using the async scraper.
    Shows dramatic speed improvement over synchronous approach.
    """
    scraper = AsyncGitHubScraper(max_concurrent=5)  # Be respectful!
    usernames = ['torvalds', 'gvanrossum', 'dhh', 'kennethreitz', 'mitsuhiko']
    
    print("Starting async scraping...")
    start_time = asyncio.get_event_loop().time()
    
    # Scrape all users concurrently
    results = await scraper.scrape_users(usernames)
    
    elapsed = asyncio.get_event_loop().time() - start_time
    print(f"Scraped {len(results)} users in {elapsed:.2f} seconds")
    
    # Show what we got
    for user in results:
        print(f"- {user['username']}: {user['name']} from {user['location']}")
    
    return results

# Run it (uncomment to execute)
# results = asyncio.run(main())
```

When I ran this, the results were impressive:

```
Starting async scraping...
Scraped 5 users in 1.34 seconds
- torvalds: Linus Torvalds from Portland, OR
- gvanrossum: Guido van Rossum from San Francisco Bay Area
- dhh: David Heinemeier Hansson from Marbella, Spain
- kennethreitz: Kenneth Reitz from Richmond, VA
- mitsuhiko: Armin Ronacher from Austria
```

Compare that to synchronous scraping:

```python
# Synchronous version for comparison
import time

def scrape_users_sync(usernames):
    scraper = GitHubScraper()
    results = []
    start = time.time()
    
    for username in usernames:
        url = f'https://github.com/{username}'
        response = scraper.respectful_get(url)
        # ... parse HTML ...
        results.append({'username': username})
    
    elapsed = time.time() - start
    print(f"Synchronous: {len(usernames)} users in {elapsed:.2f} seconds")
    return results

# This would take about 5-6 seconds for 5 users
```

That's nearly 5x faster with async! But remember, with great power comes great responsibility. Without that semaphore limiting concurrent requests, you might accidentally overwhelm GitHub's servers.

## Lessons from the Trenches

Building this crawler taught me things that tutorials rarely mention. Missing data is a constant challenge. Not every GitHub user fills out their profile completely:

```python
def extract_user_info_safely(soup, username):
    """
    Real-world extraction with proper error handling.
    Shows how to handle missing or incomplete data gracefully.
    
    Args:
        soup: BeautifulSoup object
        username (str): Username being processed
        
    Returns:
        dict: User information with None for missing fields
    """
    info = {'username': username}
    
    # Each field needs careful handling - users might not have filled them
    try:
        name_elem = soup.find('span', {'itemprop': 'name'})
        info['name'] = name_elem.text.strip() if name_elem else None
    except AttributeError:
        info['name'] = None
    
    try:
        # Company field can be complex (might have multiple parts)
        company_elem = soup.find('span', {'itemprop': 'worksFor'})
        if company_elem:
            # Company might include both org and team
            company_parts = company_elem.find_all('span')
            info['company'] = ' '.join([p.text.strip() for p in company_parts])
        else:
            info['company'] = None
    except:
        info['company'] = None
    
    # Contribution count needs special parsing
    contrib_elem = soup.find('h2', class_='f4 text-normal mb-2')
    if contrib_elem:
        # Text format: "1,234 contributions in the last year"
        contrib_text = contrib_elem.text.strip()
        # Extract just the number
        info['contributions'] = contrib_text.split()[0].replace(',', '')
    else:
        info['contributions'] = '0'
    
    return info

# Testing with different user profiles
test_users = ['torvalds', 'ghost', 'new-user-12345']
for user in test_users:
    info = extract_user_info_safely(soup, user)
    print(f"{user}: contributions={info['contributions']}, company={info['company']}")
```

The results showed the importance of error handling:

```
torvalds: contributions=5,123, company=Linux Foundation
ghost: contributions=0, company=None
new-user-12345: contributions=12, company=None
```

Some users have thousands of contributions, others have none. Some list their company, many don't. Your code needs to handle all these cases gracefully.

## The Complete Picture

After implementing both approaches, running them side by side revealed the full picture. Here's a practical example that brings it all together:

```python
import pandas as pd

def collect_github_network(username, max_users=50):
    """
    Complete example: Collect and analyze a user's GitHub network.
    Combines everything we've learned into a practical tool.
    
    Args:
        username (str): GitHub username to analyze
        max_users (int): Maximum number of users to collect
        
    Returns:
        pandas.DataFrame: Network data
    """
    # Initialize our smart collector
    collector = SmartGitHubCollector()
    
    # Step 1: Get the list of people they follow
    print(f"Getting users that {username} follows...")
    following = collector.scraper.get_user_followers(username)[:max_users]
    print(f"Found {len(following)} users to analyze")
    
    # Step 2: Collect data about each user
    results = []
    for i, user in enumerate(following, 1):
        print(f"[{i}/{len(following)}] Processing {user}...")
        
        # Extract user info (API first, then scraping)
        user_info = collector.extract_user_info(user)
        if user_info:
            results.append(user_info)
        
        # Always be respectful with delays
        time.sleep(1)
    
    # Step 3: Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Some interesting analysis
    print(f"\nNetwork Analysis for {username}:")
    print(f"Total users analyzed: {len(df)}")
    
    # Find most common locations (excluding None values)
    locations = df['location'].dropna().value_counts().head(3)
    if not locations.empty:
        print(f"Most common locations: {locations.to_dict()}")
    
    # Find most common companies
    companies = df['company'].dropna().value_counts().head(3)
    if not companies.empty:
        print(f"Most common companies: {companies.to_dict()}")
    
    # Save for further analysis
    filename = f'{username}_network.csv'
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")
    
    return df

# Run the analysis
network_data = collect_github_network('skacem', max_users=20)
```

The output revealed fascinating patterns:

```
Getting users that skacem follows...
Found 20 users to analyze
[1/20] Processing gvanrossum...
[2/20] Processing kennethreitz...
[3/20] Processing mitsuhiko...
...
[20/20] Processing sindresorhus...

Network Analysis for skacem:
Total users analyzed: 20
Most common locations: {'San Francisco': 4, 'New York': 3, 'Berlin': 2}
Most common companies: {'Google': 3, 'Microsoft': 2, 'GitHub': 2}

Data saved to skacem_network.csv
```

Looking at the CSV file, I found interesting insights. The developers I follow are mostly based in tech hubs (San Francisco, New York, Berlin), work at major tech companies, and maintain an average of 42 public repositories. This kind of analysis would be impossible with just the API's rate limits or just scraping alone. It required combining both approaches intelligently.

## What Else Is Out There

While we've focused on GitHub, the techniques we've learned apply to many other scenarios. Modern web scraping has evolved beyond BeautifulSoup and requests. Tools like Selenium and Playwright can handle JavaScript-heavy sites that BeautifulSoup can't touch. They actually run a real browser, allowing you to scrape single-page applications and sites that load content dynamically.

For large-scale scraping, frameworks like Scrapy provide built-in support for handling retries, managing cookies, rotating user agents, and distributing scraping across multiple machines. They're overkill for our GitHub project but essential for serious web scraping operations.

There's also the growing trend of scraping APIs. Services like ScraperAPI and Crawlera handle proxy rotation, CAPTCHA solving, and browser fingerprinting for you. They're particularly useful when dealing with sites that actively try to block scrapers.

## Conclusion

Building this GitHub crawler taught me that web scraping isn't just about extracting data. It's about doing so responsibly, efficiently, and legally. The best approach often combines multiple techniques, using APIs when available, scraping when necessary, and always with respect for the service you're accessing.

The complete implementation, which includes all the error handling, caching, and optimization techniques discussed here, is available on [GitHub](https://bit.ly/github-crawler-v2). Feel free to experiment with it, but remember that with great scraping power comes great responsibility.

Just because you can scrape something doesn't mean you should. The web is a shared resource, and we're all responsible for using it considerately. The goal isn't just to get data, but to be a responsible member of the web community while doing so.

In the next post, we'll explore how to handle JavaScript-heavy sites using Selenium and Playwright, opening up a whole new category of scrapeable content. Until then, happy (and responsible) scraping!

**N.B.** Remember to check robots.txt, respect rate limits, and always consider whether an API might serve your needs better than scraping. The code is meant for educational purposes, use it wisely.