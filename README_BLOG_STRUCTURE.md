# Blog Structure Guide

## Overview
This Jekyll blog is organized into two main sections:
1. **ML & Statistical Learning** - Technical articles about machine learning, data science, and business analytics
2. **Thoughts** - Personal reflections, ideas, and musings

## Directory Structure
```
skacem.github.io/
├── _posts/               # All blog posts go here
│   ├── thoughts/         # Template directory for thoughts (templates only)
│   └── *.md             # Actual blog posts (both ML and thoughts)
├── ml.html              # ML section page
├── thoughts.html        # Thoughts section page
├── index.html           # Home page with both sections
└── posts.html           # All posts page
```

## Creating New Posts

### For ML Articles
Create a new file in `_posts/` with the format: `YYYY-MM-DD-Title.md`

Front matter template:
```yaml
---
layout: post
category: ml
comments: true
title: "Your ML Article Title"
excerpt: "Brief description for preview"
author: "Skander Kacem"
tags: 
  - Machine Learning
  - Data Science
  - Your Tags
katex: true  # Enable if using mathematical notation
preview_pic: /assets/ml/image.jpg  # Optional
---
```

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
  - Personal
  - Reflection
  - Your Tags
preview_pic: /assets/thoughts/image.jpg  # Optional
---
```

## Important Notes
- **Category is crucial**: Use `category: ml` or `category: thoughts` to ensure posts appear in the correct section
- All posts go in the main `_posts/` directory (not in subdirectories)
- The `_posts/thoughts/` directory contains only templates, not actual posts
- Posts are automatically sorted by date in Jekyll

## Navigation
The site navigation includes:
- **ML** - Shows all ML & statistical learning articles
- **Thoughts** - Shows all personal thoughts and reflections
- **About** - Your about page

## Home Page
The home page displays:
- Welcome message
- Two section cards showing recent posts from each category
- A combined list of all recent posts

## Assets
Store images in organized folders:
- `/assets/ml/` - Images for ML articles
- `/assets/thoughts/` - Images for thoughts posts
- `/assets/[number]/` - Your existing numbered asset folders