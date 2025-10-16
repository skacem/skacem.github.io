# Blog Structure Guide

## Overview
This Jekyll blog is organized into three main sections:
1. **Thoughts** - Your original reflections and ideas
2. **Readings** - Notes on books and articles (finance, politics, culture, philosophy)
3. **Technical Notes** - Learning by doing - tutorials and technical explorations

## Directory Structure
```
skacem.github.io/
├── _posts/               # All blog posts go here
│   ├── thoughts/         # Template directory for thoughts (templates only)
│   └── *.md             # Actual blog posts (all categories)
├── thoughts.html        # Thoughts section page
├── readings.html        # Readings section page
├── technical-notes.html # Technical Notes section page
├── index.html           # Home page with all three sections
└── posts.html           # All posts page
```

## Creating New Posts

### For Thoughts
Create a new file in `_posts/` with the format: `YYYY-MM-DD-Title.md`

Front matter template:
```yaml
---
layout: post
category: thoughts
comments: true
title: "Your Thought Title"
excerpt: "Brief description for preview"
author: "Skander Kacem"
tags:
  - Culture
  - Philosophy
  - Your Tags
---
```

### For Readings
Create a new file in `_posts/` with the format: `YYYY-MM-DD-Title.md`

Front matter template:
```yaml
---
layout: post
category: readings
comments: true
title: "Book/Article Title"
excerpt: "Brief description of what you're reading about"
author: "Skander Kacem"
tags:
  - Finance
  - Politics
  - Philosophy
  - Your Tags
---
```

### For Technical Notes
Create a new file in `_posts/` with the format: `YYYY-MM-DD-Title.md`

Front matter template:
```yaml
---
layout: post
category: technical  # or 'ml' for backward compatibility
comments: true
title: "Your Technical Topic"
excerpt: "Brief description for preview"
author: "Skander Kacem"
tags:
  - Machine Learning
  - Python
  - Data Science
  - Your Tags
katex: true  # Enable if using mathematical notation
preview_pic: /assets/technical/image.jpg  # Optional
---
```

## Important Notes
- **Category is crucial**: Use `category: thoughts`, `category: readings`, or `category: technical` (or `ml`)
- All posts go in the main `_posts/` directory (not in subdirectories)
- The `_posts/thoughts/` directory contains only templates, not actual posts
- Posts are automatically sorted by date in Jekyll
- Old posts with `category: ml` will still appear in Technical Notes

## Navigation
The site navigation includes:
- **Thoughts** - Your original reflections
- **Readings** - Notes on books and articles
- **Technical Notes** - Learning tutorials and technical explorations
- **About** - Your about page

## Home Page
The home page displays three section cards in this order:
1. Thoughts - Recent 5 posts
2. Readings - Recent 5 posts
3. Technical Notes - Recent 5 posts

## Philosophy
This is a personal blog for "thinking out loud" - no pretensions, no performance. Write to understand things better, share your learning journey, and engage with ideas authentically.
