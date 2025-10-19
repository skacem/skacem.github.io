---
layout: post
category: ml
comments: true
title: "Plotting with Seaborn - Part 1: Foundations & Essential Plots"
author: "Skander Kacem"
tags:
    - Visualization
    - Tutorial
    - Python
katex: true
---


## The Matplotlib Problem

Picture this: You've just finished analyzing your data, discovered something interesting, and now you need to visualize it. You open up matplotlib and... three hours later, you're still tweaking axis labels, adjusting colors, and wrestling with legends that refuse to position themselves correctly. Sound familiar?

This is the matplotlib experience. Don't get me wrong: matplotlib is an incredible library. It gives you complete control over every pixel of your visualization. But that control comes at a cost: time. Lots of it.

What if there was a better way? What if you could create beautiful, publication-ready visualizations with just one or two lines of code? Enter Seaborn.

Seaborn sits on top of matplotlib and does all the tedious work for you. It automatically handles statistical aggregations, picks beautiful color schemes, creates informative legends, and makes your plots look professional by default. Think of it as matplotlib with a really good personal assistant, one that anticipates what you need before you ask.

In this tutorial, you'll discover how to create stunning visualizations efficiently. We'll start with the fundamentals, work through practical examples with real datasets, and by the end, you'll have the essential toolkit for everyday data visualization.

## Navigating the Python Visualization Landscape

Before we dive into Seaborn, let's take a step back. The Python visualization ecosystem is... complex. There are dozens of libraries, each with their own philosophy and use cases. If you've felt overwhelmed trying to choose between them, you're not alone.

<img src="/assets/3/landscape.png" alt="The python Visualization Landscape" width="800">

The landscape looks chaotic because it evolved organically over years, with different libraries solving different problems. matplotlib provides low-level control, plotly creates interactive web visualizations, bokeh powers dashboards - the list goes on. For a fascinating deep dive into how we got here, check out [Jake Vanderplas' PyCon 2017 talk](https://www.youtube.com/watch?v=FytuB8nFHPQ).

Here's the good news: for data science and machine learning work, one library stands out above the rest. Seaborn has become the definitive choice for statistical data visualization, and once you understand why, you'll never want to go back.

### What Makes Seaborn Special

Imagine you're cooking dinner. matplotlib is like starting from individual ingredients - you have complete control, but you need to prep everything yourself. Seaborn is like having a meal kit delivered: the hard work is done, but you can still customize to taste.

When you give Seaborn a pandas DataFrame and tell it what kind of plot you want, magic happens under the hood. It automatically converts your data into visual attributes like colors and sizes. It computes statistical transformations like means and confidence intervals. It decorates your plot with clear labels and legends. And it does all of this while making aesthetic choices that would take you hours to configure in matplotlib.

But here's the beautiful part: Seaborn doesn't lock you out of matplotlib's power. When you need fine-grained control for that one tricky customization, matplotlib is right there, accessible and ready. You get the best of both worlds.

That said, understanding a bit about how matplotlib works will help you get the most out of Seaborn. We'll cover the essentials shortly, but first, let me show you why this matters with a simple comparison.

## Seeing is Believing: Matplotlib vs. Seaborn

Let's create a visualization using both libraries and see the difference for ourselves. We'll start by loading the tools we need:

```python
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Notebook settings
plt.rcParams['figure.figsize'] = 9, 4
```

Now we'll generate some random walk data - imagine tracking six different stocks over time:

```python
# Create random walk data
rng = np.random.RandomState(123)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with matplotlib defaults
plt.style.use('classic')
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

<img src="/assets/3/rndwlk_mpl.png" alt="visualization of a simple random-walk with matplotlib pyplot" width="700">

There it is - a perfectly functional matplotlib plot. It shows all the information we need, and if you've used MATLAB before, you might even feel a bit nostalgic. This similarity to MATLAB wasn't accidental; it was a deliberate design choice that helped Python gain traction in scientific computing. The plot works, but aesthetically... well, it looks like something from the 1990s.

Now let's create the exact same plot, but this time we'll add just one line to enable Seaborn's styling:

```python
sns.set()
# Same plotting code as above
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

<img src="/assets/3/rndwlk_sns.png" alt="visualization of a simple random-walk with seaborn classic settings" width="700">

Look at that transformation. Better colors, cleaner grid lines, improved contrast - suddenly our plot looks like it belongs in a modern publication. And we achieved this by adding exactly one line of code.

This is Seaborn's philosophy in action: excellent defaults that require no effort. Before we go further, take a moment to browse the [Seaborn gallery](https://seaborn.pydata.org/examples/index.html). You'll see not just the range of visualizations possible, but the consistent professional quality they all share.

Here's something important to understand about data science: your work isn't done when you finish your analysis. You need to communicate your findings, often to people with very different backgrounds. Your engineering colleagues want technical details and precision. Your executives want clarity and big-picture insights. Creating separate visualizations for each audience wastes time.

With Seaborn, you don't have to choose. The plots you create during exploratory analysis are already polished enough to show stakeholders. This efficiency matters more than you might think - being able to effectively communicate your findings to diverse audiences often determines whether your work creates real impact.

## Understanding Matplotlib's Foundation

Now that you're motivated to learn Seaborn, let's briefly talk about matplotlib's structure. I promise this will be quick, and understanding these concepts will make you much more effective when you need to customize your plots.

Matplotlib has three layers: Backend (handles the rendering), Artist (manages visual elements), and Scripting (the interface you use). The Artist layer is where the magic happens - every single visual element in a plot, from lines to text to legends, is an Artist object that you can access and modify.

<img src="/assets/3/anatomy.png" alt="Anatomy of a figure" width="700">

Here's the thing though: Seaborn's defaults are so well-designed that you'll rarely need to dig into the Artist layer. The library's designers have already made thousands of aesthetic decisions for you, and they made them well. Your time is better spent understanding your data than tweaking plot aesthetics.

Still, when you do need customization, knowing these three concepts will save you hours of frustration: Figure, Axes, and Axis. People confuse these constantly, so let's clarify them once and for all.

### The Three Amigos: Figure, Axes, and Axis

Think of a Figure as the canvas - the entire window or page you're working with. It's the container for everything else. A Figure can hold multiple plots, just like a page in a magazine can have multiple images.

Let's create a Figure with four plots to see this in action:

```python
# create a figure with 4 axes
fig, ax = plt.subplots(nrows=2, ncols=2)

# Annotate the first subplot
ax[0, 0].annotate('This is an Axes',
                  (0.5, 0.5), color='r',
                  weight='bold', ha='center',
                  va='center', size=14)

# Set the axis limits of the second subplot
ax[0, 1].axis([0, 3, 0, 3])

# Title of the figure
plt.suptitle('One Figure with Four Axes', fontsize=18);
```

<img src="/assets/3/1f4a.png" alt="figure object with four axes" width="700">

Now, Axes (plural, with an 's') refers to an individual plot area - the region where your data lives. It's confusing because it sounds like "axis," but they're completely different things. Each Axes is like a picture frame containing one visualization. A Figure can have many Axes, but each Axes belongs to exactly one Figure.

```python
print(fig.axes)
# Output: [<AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>]
```

Finally, Axis (singular) refers to the actual x-axis or y-axis - the number lines with tick marks. Each Axes contains two Axis objects (or three for 3D plots). When you set limits like "show x from 0 to 10," you're modifying an Axis object.

```python
# Print the axis limits of the second subplot
print(ax[0, 1].axis())
# Output: (0.0, 3.0, 0.0, 3.0)
```

Got it? Figure is the canvas, Axes are the individual plots, and Axis refers to the x and y number lines. Keep these straight, and you'll never be confused when reading documentation or Stack Overflow answers.

With that foundation in place, let's explore how Seaborn actually works.

## Two Flavors of Seaborn Functions

Here's something crucial that trips up many Seaborn beginners: not all Seaborn functions work the same way. They come in two distinct flavors, and understanding the difference will save you from frustration and confusion.

### Axes-Level Functions: The Team Players

The first type is called axes-level functions. These are the team players - they work with a single Axes object and play nicely with matplotlib. Think of them as well-behaved guests at a party who can mingle with anyone.

When you call an axes-level function, it creates a plot on a single Axes without affecting anything else in your Figure. This makes them perfect for building complex multi-plot layouts where you want precise control over what goes where.

Let me show you what I mean with a practical example. We'll load the famous penguins dataset - it contains measurements from three penguin species in Antarctica:

```python
# Load our dataset
penguins = sns.load_dataset('penguins')

# Create a figure with two subplots using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(8, 4), 
                        gridspec_kw=dict(width_ratios=[4, 3]))

# Use Seaborn to generate a scatter plot on the first subplot
sns.scatterplot(data=penguins, x="flipper_length_mm", 
                y="bill_length_mm", hue="species", ax=axs[0])

# Use matplotlib to create a bar chart on the second subplot
species_counts = dict(penguins['species'].value_counts())
axs[1].bar(species_counts.keys(), species_counts.values(),
           color=['royalblue', 'darkseagreen', 'darkorange']);
```

<img src="/assets/3/peng1.png" alt="a matplotlib plot with a seaborn plot in the same figure" width="700">

See how seamlessly they work together? The `ax=axs[0]` parameter tells Seaborn exactly where to draw the plot. This is the power of axes-level functions - they integrate perfectly into any matplotlib workflow.

### Figure-Level Functions: The Showrunners

The second type is figure-level functions. These are the showrunners - they take charge of the entire Figure. When you call a figure-level function, it creates a complete figure from scratch and returns a special object called a FacetGrid.

Figure-level functions sacrifice some flexibility for convenience and power. You can't easily combine them with matplotlib plots, but they excel at creating consistent, multi-panel visualizations with minimal code.

Here's the relationship between the two types:

<img src="/assets/3/function_overview2.png" alt="Figure-level functions versus axes-level functions in seaborn" width="800">

Each figure-level function acts as a coordinator for several related axes-level functions. For example, `displot()` is the figure-level function that can create histograms, kernel density plots, and other distribution visualizations.

Let's see it in action:

```python
# Create a kernel density estimate plot
sns.displot(data=penguins, x="flipper_length_mm", hue="species",
            multiple="stack", kind="kde")
```

<img src="/assets/3/kde.png" alt="kernel density estimation plot of penguins dataset generated with seaborn displot" width="700">

Beautiful, right? One line of code gave us a publication-ready visualization with a clear legend, good colors, and proper styling. Recreating this with matplotlib would take dozens of lines.

The real superpower of figure-level functions, though, isn't just their aesthetics - it's their ability to create sophisticated faceted plots. Imagine automatically creating separate subplots for each category in your data, all with consistent styling and proper alignment. That's where figure-level functions shine, and we'll explore this capability in depth in Part 3 of this tutorial series.

For now, we're going to focus on axes-level functions for two important reasons. First, they're easier to integrate into complex figures where you might want to mix Seaborn and matplotlib. Second, understanding axes-level functions gives you a solid foundation - once you master them, figure-level functions become intuitive.

Just remember: anything you can do with an axes-level function, you can also do with the corresponding figure-level function. They're two ways to achieve the same goal, each with different trade-offs.

## Your Essential Plot Types

Let's dive into the practical stuff. I'm going to walk you through the five essential visualizations that handle 80% of everyday data analysis needs. We'll use real datasets to see how these plots work in practice, and I'll share insights about when each type of plot shines and when you should avoid it.

For deeper dives into any of these, the [Seaborn official tutorial](https://seaborn.pydata.org/tutorial.html) is comprehensive and well-written.

---

## Bar Plots: Comparing Categories

Bar plots are everywhere, and there's a good reason for that. When you need to compare average values across categories, nothing beats a well-designed bar chart for immediate visual impact.

Think about presenting quarterly sales figures to your team. Numbers in a table make people's eyes glaze over. A bar plot tells the story instantly - which quarters performed well, which struggled, and by how much.

Let's work with an employee dataset to see this in practice:

```python
# Load data
df = pd.read_csv('data/employee.csv')
```

Seaborn offers four presentation styles: `paper`, `notebook`, `talk`, and `poster`. They're optimized for different contexts - from journal publications to conference presentations. The default is `notebook`, which works great for Jupyter notebooks and typical screens. You switch between them like this:

```python
sns.set_context("notebook", rc={"figure.figsize": (10, 6)})

sns.barplot(x=df['Department'], y=df['Age'])
plt.ylim(0, 45)
plt.title('Average Age by Department');
```

<img src="/assets/3/bar.png" alt="Bar plot with seaborn of employees ages per department" width="750">

The plot automatically shows the mean age for each department, with confidence intervals represented by the error bars. Those error bars are telling you something important: how certain we can be about these averages. Larger error bars mean more uncertainty.

Now, what if you want to add another dimension to your comparison? Maybe you're curious whether business travel frequency varies by department and affects age patterns. The `hue` parameter lets you split each bar into nested groups:

```python
sns.barplot(x=df['Department'], y=df['Age'],
            hue=df['BusinessTravel'],
            palette='magma_r')
plt.title('Average Age by Department and Travel Frequency');
```

<img src="/assets/3/bar2.png" alt="bar plot with nested grouping of variables" width="750">

Notice how we used a different color palette - `magma_r`. Seaborn includes six palette variations: `deep`, `muted`, `pastel`, `bright`, `dark`, and `colorblind`. That last one is important. If your work might be viewed by people with color vision deficiencies (and statistically, some of your audience probably will be), the `colorblind` palette ensures everyone can distinguish your categories.

Bar plots work best with a manageable number of categories. Beyond about ten, they become hard to read. Also, resist the temptation to use bar plots for continuous data - that's what scatter plots and line plots are for. Each visualization type has its purpose.

---

## Count Plots: Showing Distribution

While bar plots show aggregate statistics, count plots answer a simpler question: "How many of each category do I have?" It's like taking inventory.

This matters more than you might think. Before you build a classification model, you need to know whether your classes are balanced. Before you analyze survey responses, you need to understand your sample composition. Count plots give you this insight instantly.

```python
sns.countplot(df['EducationField'])
plt.xticks(rotation=45)
plt.title('Employee Distribution by Education Field');
```

<img src="/assets/3/count.png" alt="count plot with total number of employees and their degrees" width="750">

Now you can immediately see that life sciences and medical fields dominate this workforce, while human resources and technical degrees are underrepresented. This context matters when you're interpreting any analysis based on this data.

As with bar plots, you can add a second categorical variable with `hue`. Let's examine how employee attrition relates to education field:

```python
sns.countplot(x=df['EducationField'],
              hue=df['Attrition'],
              palette='colorblind')
plt.xticks(rotation=30)
plt.title('Attrition Rates by Education Field');
```

<img src="/assets/3/count2.png" alt="count plot with hue" width="750">

Interesting pattern emerging here - some fields show higher attrition rates than others. This single visualization could spark valuable conversations about retention strategies in different departments.

---

## Line Plots: Trends and Relationships

Here's where we need to pause for an important conversation. If you're coming from matplotlib or other libraries, Seaborn's line plots might surprise you. They don't just connect points - they do something more sophisticated.

By default, when Seaborn encounters multiple y-values for the same x-position, it aggregates them. It shows you the mean and wraps a confidence interval around it. This behavior is incredibly useful for time series and grouped data, but if you just want simple point-to-point lines, you might find it confusing.

Let's see what I mean:

```python
sns.lineplot(x=df['Department'], y=df['Age']);
```

<img src="/assets/3/line.png" alt="line plot with estimate of the central tendency and a confidence interval for the estimates" width="750">

That shaded region? It's showing you the 95% confidence interval around the mean age for each department. Seaborn is automatically computing statistics and visualizing uncertainty. Pretty neat, right?

This becomes even more powerful when you're comparing multiple groups:

```python
plt.style.use('ggplot')
sns.lineplot(x=df['Department'], y=df['Age'],
             hue=df['EducationField'])
plt.legend(loc='lower center', title='Education Field')
plt.title('Average Age by Department and Education Field');
```

<img src="/assets/3/line3.png" alt="seaborn line plot with many lines and central tendency estimates" width="750">

Now you're seeing trends across both department and education field simultaneously, with automatic handling of all the complexity underneath.

But here's the thing: computing confidence intervals is computationally expensive. If you're working with large datasets and those intervals aren't adding value to your analysis, turn them off:

```python
sns.lineplot(x=df['Department'], y=df['Age'],
             hue=df['EducationField'],
             ci=None)
plt.legend(loc='lower center', title='Education Field');
```

<img src="/assets/3/line4.png" alt="seaborn lineplot with many lines without central tendency estimates" width="750">

Much faster to render, and for large datasets (say, more than 10,000 points), the confidence intervals often become so narrow that they don't add much information anyway.

One more thing: if you want simple point-to-point line connections without any statistical aggregation, just use matplotlib's `plt.plot()` instead. There's no shame in that - use the right tool for the job.

---

## Scatter Plots: Exploring Relationships

If line plots show trends over sequences, scatter plots reveal relationships between two continuous variables. Each point is an observation, and patterns in the cloud of points tell you about correlations, clusters, and outliers.

Scatter plots are fundamental to exploratory data analysis. They help you discover relationships you didn't know existed and verify assumptions about relationships you suspected.

Let's work with the penguins dataset again. These are real measurements from three Antarctic penguin species, collected by researchers studying how body size relates to ecological niches:

```python
penguins = sns.load_dataset('penguins')

# Customize the appearance
sns.set_style('darkgrid')
sns.set_context("notebook", font_scale=1.3,
                rc={"figure.figsize": (10, 8)})

sns.scatterplot(data=penguins, x="flipper_length_mm",
                y="bill_length_mm", hue="species", s=60)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Length (mm)");
```

<img src="/assets/3/scatter1.png" alt="seaborn scatter plot of the penguins dataset" width="750">

Look at how the species cluster - each occupies a distinct region of the plot. This isn't random; it reflects real biological differences in how these penguins have adapted to their environments. Adelie penguins (orange) are smaller overall, Gentoo (green) have long flippers relative to bill length, and Chinstrap (blue) sit in between.

Now, you can add even more dimensions to a scatter plot. Color (hue), size, and shape can all encode additional variables:

```python
markers = {'Male': 'o', 'Female': 'X'}

sns.set_context("notebook", font_scale=1.2,
                rc={"figure.figsize": (10, 8)})

sns.scatterplot(x="flipper_length_mm", y="bill_length_mm",
                hue="species", style='sex', size="body_mass_g",
                data=penguins, markers=markers,
                sizes=(20, 300), alpha=.5)
plt.legend(ncol=2, loc=4)
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Length (mm)");
```

<img src="/assets/3/scatter2.png" alt="seaborn scatterplot with five dimensions" width="750">

Technically impressive? Yes. Easy to interpret? Debatable. We're now encoding five dimensions: x-position, y-position, color, shape, and size. This can work for presentations where you guide people through the plot, but for written reports or exploratory analysis, simpler is usually better.

Here's a design principle worth remembering: more dimensions don't automatically mean better insights. If your plot becomes a puzzle to decode, you've gone too far. Sometimes the best solution is creating multiple simpler plots rather than one complex visualization.

There's one situation where you might prefer figure-level functions over axes-level: when your legend blocks important data. This happens when data fills the entire plot area and the legend is large. Here's the problem:

```python
plt.rcParams["figure.figsize"] = 8, 5
plt.style.use('fivethirtyeight')

sns.scatterplot(x=df['Age'], y=df['MonthlyRate'],
                hue=df['Department'],
                size=df['YearsAtCompany'],
                sizes=(20, 200), alpha=.3)
plt.xlim(29.5, 45.5)
plt.title('Age vs Monthly Rate');
```

<img src="/assets/3/emp1.png" alt="scatterplot with dots scattered over the entire axes instance" width="750">

See how the legend covers a chunk of data? Frustrating. The figure-level solution puts the legend outside:

```python
sns.relplot(x=df['Age'], y=df['MonthlyRate'],
            hue=df['Department'],
            size=df['YearsAtCompany'],
            sizes=(20, 200), alpha=.3, 
            kind="scatter",
            height=5, aspect=8/5)
plt.xlim(29.5, 45.5)
plt.title('Age vs Monthly Rate');
```

<img src="/assets/3/em2.png" alt="scatterplot using the figure-level function sns.relplot()" width="750">

Problem solved. This is one of those cases where the figure-level function's opinionated layout actually helps.

---

## Heat Maps: Seeing Patterns in Matrices

Some data naturally lives in a matrix - correlation coefficients, confusion matrices, missing data patterns. For this kind of data, heat maps are unbeatable. They encode matrix values as colors, letting you spot patterns at a glance that would be invisible in a table of numbers.

The classic use case is correlation matrices. Let's look at the Titanic dataset:

```python
titanic = sns.load_dataset('titanic')

sns.heatmap(titanic.corr(), annot=True, cmap='GnBu');
```

<img src="/assets/3/heat1.png" alt="heat map of the titanic dataframe correlation matrix" width="750">

The `annot=True` parameter displays the actual correlation values in each cell, while the color intensity provides immediate visual ranking. You can instantly see that fare and passenger class are strongly related (not surprising - first class tickets cost more), while age has weak correlations with most other variables.

But heat maps shine in another, often overlooked application: visualizing missing data. When data is missing, it's rarely random. The pattern of missingness often reveals something important about how your data was collected, and that insight can be valuable.

```python
sns.heatmap(titanic.isnull(), yticklabels=False, 
            cbar=False, cmap='YlOrRd');
```

<img src="/assets/3/heat2.png" alt="using heatmaps to show missing values" width="750">

The red streaks show missing values. Immediately you can see that the `deck` column is mostly empty, and `age` has scattered missingness throughout. This isn't just trivia - it affects how you should handle these variables in analysis. Maybe passengers in certain classes had deck information recorded while others didn't. Maybe age was left blank when passengers didn't provide it at booking.

Understanding missingness patterns can save you from making wrong assumptions later. If you want to go deeper into missing data analysis, check out the `missingno` library - it builds on matplotlib and Seaborn to provide specialized visualizations just for this purpose.

---

## Your Essential Toolkit

You've now mastered the five plot types that handle most everyday data visualization needs:

**Bar plots** for comparing means across categories - your go-to for showing "which group is highest/lowest."

**Count plots** for showing frequencies - essential for understanding your data's composition before analysis.

**Line plots** for trends with statistical aggregation - perfect for time series and comparing trajectories across groups.

**Scatter plots** for exploring relationships - the foundation of correlation analysis and pattern discovery.

**Heat maps** for matrix data - unmatched for correlation analysis and missing data patterns.

These five plot types will serve you well in most situations. They're simple enough to interpret quickly, flexible enough to handle various data types, and professional enough for any audience.

## What's Next

In **Part 2: Distributions & Statistical Plots**, we'll go deeper into statistical visualizations. You'll learn to understand distributions through histograms and box plots, handle large datasets with letter-value plots, reveal distribution shapes with violin plots, and model relationships with regression plots. These tools transform you from creating plots to conducting visual statistical analysis.

You'll also learn about complementary plots like strip plots, swarm plots, and rug plots that add detail to your visualizations without overwhelming them.

The journey continues - you have the essentials, now let's add depth.

## Resources for Going Deeper

The [Seaborn official tutorial](https://seaborn.pydata.org/tutorial.html) is comprehensive and well-maintained. The [example gallery](https://seaborn.pydata.org/examples/index.html) shows what's possible and provides code for each visualization. When you need details about specific parameters, the [API reference](https://seaborn.pydata.org/api.html) is authoritative.

All the code from this tutorial, along with the datasets used, is available in [this GitHub repository](https://github.com/skacem/TIL/blob/main/seaborn_intro.ipynb). Feel free to download and experiment.

---

## References

[1] VanderPlas, Jake. 2016. *Python Data Science Handbook*. O'Reilly Media, Inc.

[2] [Matplotlib - Axes Class](https://www.tutorialspoint.com/matplotlib/matplotlib_axes_class.htm)

[3] Desai, Meet. "Matplotlib + Seaborn + Pandas." Medium, Towards Data Science, 30 Oct. 2019

[4] Waskom, M. L., (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021, https://doi.org/10.21105/joss.03021

