---
layout: post
comments: true
title: "Plotting with Seaborn - Part 2"
excerpt: "Seaborn provides a high-level interface to Matplotlib and is compatible with Pandas’ data structures. Given a pandas DataFrame and a specification of the plot to be created, seaborn automatically converts the data values into visual attributes, internally computes statistical transformations and decorates the plot with informative axis labels and legends. In other words, seaborn saves you all the effort you would normally need to put into creating figures with matplotlib."
author: "Skander Kacem"
tags: 
    - Visualization
    - Tutorial
    - Seaborn
    - Python
katex: true
preview_pic: /assets/0/seabornp2.png
---

## Faceting

One powerful technique in data visualization, particularly when exploring multidimensional data sets, is to create multiple versions of the same plot with several subsets of that data alongside each other, sharing the same scale and axis.  This allows a lot of information to be presented compactly and in a comparable way. 

<div class="imgcap">
<img src="/assets/4/Horse_in_Motion.jpg" style="zoom:50%;" alt="The horse in motion by Eadweard Muybridge as one of the first small multiple technique"/>
<div class="thecap"> The Horse in Motion by Eadweard Muybridge</div></div>


It is related to the idea of ["small multiples"](https://en.wikipedia.org/wiki/Small_multiple) (also known as trellis or lattice) first introduced by Eadweard Muybridge and then popularized by Edward Tufte [1].

> At the heart of quantitative reasoning is a single question: *Compared to what?* Small multiple designs, multivariate and data bountiful, answer directly by visually enforcing comparisons of changes, of the differences among objects, of the scope of alternatives. For a wide range of problems in data presentation, small multiples are the best design solution. -- Edward Tufte, 1990

While this may sound intuitive, it is very tedious to produce unless you use the right visualization library.   
One of the most powerful aspects of the seaborn  is how easy it is to create these types of plots. With a single command, you can split a given plot into many related plots using: `FacetGrid()`, `JointGrid()` or `PairGrid()`.

In the following we'll put these seaborn's classes in action, and see why they are so useful.

## Faceting with Seaborn

### FacetGrid

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
plt.style.use('bmh')
# plot a histogram 
g.map(sns.histplot, 'body_mass_g', binwidth=200, kde=True)

# set x, y labels
g.set_axis_labels('Body Mass (g)', "Count")
```

<div class="imgcap">
<img src="/assets/4/facetGrid2.png" style="zoom:120%;" alt="3x2 body mass facets generated with FacetGrid"/>
<div class="thecap"> </div></div>

Now, the function we pass to the `map` method doesn't have to be a seaborn or matplotlib plotting function. It can handle any type of function as long as it meets the following requirements:

1. It must plot on the "currently active" matplotlib `axes`. This applies to functions in the `matplotlib.pyplot` namespace. In fact you can call `matplotlib.pyplot.gca()` to get a reference to the current `axes` if you intend to use their methods.
2. It must accept the data that it plots as arguments.
3. It must have a generic dictionary of `**kwargs` and pass it along to the underlying plotting function.

So let's define our own custom function according to the above requirements. The function is supposed to compute the average of a given variable and add it as a dashed line to the currently active `axes`.

```python
def add_mean(data, var, **kwargs):

    # 1. requirement
    # Calculate mean for each group
    m = np.mean(data[var])
    
    # 2. requirement
    # Get current axis
    ax = plt.gca()
    
    # 3. requirement
    # Add line at group mean
    ax.axvline(m, color='maroon', lw=3, ls='--', **kwargs)
    
    # Annotate group mean
    x_pos=0.6
    if m > 5000: x_pos=0.2
    ax.text(x_pos, 0.7, f'mean={m:.0f}', 
            transform=ax.transAxes,  
            color='maroon', fontweight='bold', 
            fontsize=14)

```

Let's go ahead and plot the same figure as above, but without the histograms and on top of that we will be adding our own custom function.

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

FacetGrid serves as a backbone for the following figure-level functions: `relplot()`, `catplot()`, `lmplot()` and `displot()`. In fact, it is strongly recommended to use one of these functions instead of using FacetGrid directly, particularly if you are using seaborn functions that infer semantic mappings from the dataset within your facets.

### PairGrid

PairGrid is another class provided by seaborn for faceting. It shows pairwise relationships between data elements. Unlike FacetGrid, it uses different variable pairs for each facet.  

The usage of PairGrid is similar to facetGrid. We start by calling the PairGrid constructor in order to initialize the facet grid and then call the plotting functions.  In the following example we are going to use the iris dataset from seaborn.

```python
# Load the iris dataset
iris = sns.load_dataset('iris')

# Set the plotting style
plt.style.use('seaborn-paper')

# Initiate the PairGrid instance
g = sns.PairGrid(iris, diag_sharey=False, hue="species", palette="Set2")

# Scatter plots on the upper triangle
g.map_upper(sns.scatterplot)
# Kde on the diagonal
g.map_diag(sns.histplot, hue=None, legend=False, color='palevioletred', kde=True, bins=8, lw=.3)


g.map_lower(sns.kdeplot, lw=.1)
g.add_legend();
```

<div class="imgcap">
<img src="/assets/4/pairGrid1.png" style="zoom:90%;" alt="Penguins Pairgrid using different plotting functions"/>
<div class="thecap"> </div></div>


PairGrid serves as the backbone to the figure-level function `pairplot()`. W, unless you need more flexibility in generating your plots.

### JointGrid

JointGrid is another facet class from seaborn. It combines univariate plots such as histograms, rug plots, and KDE plots with bivariate plots such as scatter and regression.  
Let's create a JointGrid instance without plots, to see how it looks like.

```python
g = sns.JointGrid(data=[])
```

<div class="imgcap">
<img src="/assets/4/jointGrid1.png" style="zoom:90%;" alt="JointGrid default layout"/>
<div class="thecap"> </div></div>

The middle, large axes, referred to as joint axes, is intended for bivariate plots such as scatter and regression. The other two axes, known as marginal axes, are meant for univariate plots.  
The simplest plot method, `plot()`, takes a pair of functions: one for the joint axes and one for the two marginal axes.

```python
tips = sns.load_dataset('tips')
g = sns.JointGrid(x="total_bill", y="tip", data=tips, hue="time", palette="magma")
g = g.plot(sns.scatterplot, sns.histplot)
```

<div class="imgcap">
<img src="/assets/4/jointGrid2.png" style="zoom:100%;" alt="JointGrid with tips dataset and time as hue parameter"/>
<div class="thecap"> </div></div>

If you prefer to use different arguments for each function then you should use the following `JointGrid` class methods: `plot_joint()` and `plot_marginals()`.

```python
# Let's use the penguins dataset again
g = sns.JointGrid(data=penguins, x="bill_length_mm", 
                  y="bill_depth_mm", hue="species", palette="Set1")
# Histplot with kde on both marginal axes
g.plot_marginals(sns.histplot,  kde=True, hue=None, legend=False, element='step')
# kde and scatter plots on the joint axes
g.plot_joint(sns.kdeplot, levels=4)
g.plot_joint(sns.scatterplot)
```

<div class="imgcap">
<img src="/assets/4/jointGrid3.png" style="zoom:100%;" alt="JointGrid with penguins dataset and species as hue parameter using plot_marginal() and plot_joint()"/>
<div class="thecap"> </div></div>



JointGrid serves also as a backbone for a figure-level function and it is recommended to use the figure-level function `jointplot()` instead of JointGrid.

Let me briefly show you an example. This time we are using the tips dataset.

```python 
tips = sns.load_dataset('tips')
```



[ ] Small Multiples
[ ] 





In the first part of this long tutorial, we first introduced seaborn and showed how easy it was to create elegant yet professional plots with little code compared to matplotlib. Then, using different datasets, we explained the main types of plots for exploratory and explanatory data analysis. Each time, providing clarification on which plot works best in the given context.   All plot functions we used were axes-level functions.  So that, each plot can be used as a subplot on a non-Seaborn figure.


## References

[1] Tufte, Edward (1990). Envisioning Information. Graphics Press. p. 67. ISBN 978-0961392116.