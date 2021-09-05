---
layout: post
comments: true
title: "Plotting with Seaborn - Part 2"
excerpt: "Seaborn provides a high-level interface to Matplotlib and is compatible with Pandas’ data structures. Given a pandas DataFrame and a specification of the plot to be created, seaborn automatically converts the data values into visual attributes, internally computes statistical transformations and decorates the plot with informative axis labels and legends. In other words, seaborn saves you all the effort you would normally need to put into creating figures with matplotlib."
author: "Skander Kacem"
tags: [Visualization, tutorial, Seaborn]
katex: true
preview_pic: /assets/0/seabornp2.png
---

## Faceting

One powerful technique in data visualization, particularly when exploring multidimensional data sets, is to create multiple versions of the same plot with several subsets of that data alongside each other, sharing the same scale and axis.  This allows a lot of information to be presented compactly and in a comparable way. It is related to the idea of ["small multiples"](https://en.wikipedia.org/wiki/Small_multiple) (also known as trellis or lattice) first introduced by Eadweard Muybridge and then popularized by Edward Tufte [1].

> At the heart of quantitative reasoning is a single question: *Compared to what?* Small multiple designs, multivariate and data bountiful, answer directly by visually enforcing comparisons of changes, of the differences among objects, of the scope of alternatives. For a wide range of problems in data presentation, small multiples are the best design solution. -- Edward Tufte, 1990

While this may sound intuitive, it is very tedious to produce unless you use the right visualization library.   
One of the most powerful aspects of the seaborn  is how easy it is to create these types of plots. With a single command, you can split a given plot into many related plots using: `FacetGrid()`, `JoinGrid()` or `PairGrid()`.

In the following we'll put these seaborn's classes in action, and see why they are so useful.

## FacetGrid

The most basic utility provided by seaborn for faceting is the FacetGrid. It is an object that specifies how to split the layout of the data to be visualized. It partitions a figure into a matrix of axes defined by row and column faceting variables. It is most useful when you have two discrete variables, and all combinations of the variables exist in the data..  A third dimension can be added by specifying a hue parameter.


For example, suppose that we're interested in comparing male and female penguins of each specie in some way. To do this, we start by importing the required modules, setting the notebook environment and loading the penguins dataset as pandas DataFrame:

```python
# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the notebook environment
%precision %.3f
pd.options.display.float_format = '{:,.3f}'.format
plt.rcParams["figure.figsize"] = 9,5
pd.set_option('display.width', 100)
plt.rcParams.update({'font.size': 22})
plt.style.use('fivethirtyeight')

# Load the penguins dataset
penguins = sns.load_dataset('penguins')
```

Then we create an instance of the FacetGrid class with our data, telling it that we want to break the `sex` and `species` variables, by `row` and `col`. Since we have 3 species, we obtain a 2x3-grid of facets ready for us to plot something on them.  

```python
g = sns.FacetGrid(penguins, row='sex', col='species', aspect=1.75)
g.set_titles(row_template='{row_name}', col_template='{col_name} Penguins');
```

<div class="imgcap">
<img src="/assets/4/facetGrid1.png" style="zoom:90%;" alt="3x2 matrix of grids generated with FacetGrid"/>
<div class="thecap"> </div></div>
      
Each facet is labeled at the top. The overall layout minimizes the duplication of axis labels and other scales.  

We then use the `FacetGrid.map()` method to plot the data on the instantiated grid object.

```python
# plot a histogram 
g.map(sns.histplot, 'body_mass_g', binwidth=200, kde=True)

# set x, y labels
g.set_axis_labels('Body Mass (g)', "Count")
```

<div class="imgcap">
<img src="/assets/4/facetGrid2.png" style="zoom:90%;" alt="3x2 body mass facets generated with FacetGrid"/>
<div class="thecap"> </div></div>

Now, the function we pass to the `map` method doesn't have to be a seaborn plotting function. It accepts any kind of function you'd like as long as that function has a data argument.  
So, let's define our own custom  function, which compute the average of a given variable and displays it on all facets:



```python
def add_mean(data, var, **kws):
    # Author @kimfetti

    # Calculate mean for each group
    m = np.mean(data[var])
    
    # Get current axis
    ax = plt.gca()
    
    # Add line at group mean
    ax.axvline(m, color='maroon', lw=3, ls='--')
    
    # Annotate group mean
    x_pos=0.6
    if m > 5000: x_pos=0.2
    ax.text(x_pos, 0.7, f'mean={m:.0f}', 
            transform=ax.transAxes,  
            color='maroon', fontweight='bold', 
            fontsize=14)

```

Let's go ahead and plot the same figure as above, but without the histograms. On top of that we will be adding our own custom function for computing the mean:

```python
g = sns.FacetGrid(penguins, row='sex', col='species', hue='species', height=6, aspect=1)

g.map(sns.kdeplot, 'body_mass_g', lw=3, shade=True)
g.map_dataframe(add_mean, var='body_mass_g')
g.set_axis_labels('Body Mass (g)', "Count")
g.add_legend()
```

As you might notice, instead of using the `map()` method to call our custom function, we used the `map_dataframe()` method.

<div class="imgcap">
<img src="/assets/4/facetGrid3_1.png" style="zoom:90%;" alt="3x2 body mass facets generated with FacetGrid"/>
<div class="thecap"> </div></div>


As you can see, FacetGrid is simple yet powerful.  In just a few lines and without thinking about the layout and the appearance, you can elegantly convey a great deal of information. In my opinion, this is a vastly under-used technique in visualization and should actually be part of every exploratory data analysis, whether your goal is a report or a model.




serves as the backbone for `relplot()`, `catplot()`, `lmplot()` and `displot()`. 









[ ] Small Multiples
[ ] 





In the first part of this long tutorial, we first introduced seaborn and showed how easy it was to create elegant yet professional plots with little code compared to matplotlib. Then, using different datasets, we explained the main types of plots for exploratory and explanatory data analysis. Each time, providing clarification on which plot works best in the given context.   All plot functions we used were axes-level functions.  So that, each plot can be used as a subplot on a non-Seaborn figure.


## References

[1] Tufte, Edward (1990). Envisioning Information. Graphics Press. p. 67. ISBN 978-0961392116.