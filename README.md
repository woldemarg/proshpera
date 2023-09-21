This tool utilises sophisticated PCA with a cosine kernel to generate informative visualisations of multi-dimensional data in three-dimensional space.  Following the PCA process, the data is normalised by shifting each point to a centroid and making it the unit norm. To enhance the visualisation, vectors are additionally scaled with precision to move the farthest points closer to the surface of the sphere. The outcome is an engaging and instinctive representation of the data in spherical format. The tool initiates interactive visualisations in a new tab of your default web browser, facilitating data exploration and analysis.

## Basic Usage
#### Init visualizer
```python
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
from prosphera.projector import Projector

# Instantiate the class
visualizer = Projector()
```
#### Generated dataset
```python
# Generate data
data, labels = make_blobs(
    n_samples=5000,
    centers=50,
    n_features=25,
    random_state=1234)

# Call the visualize method to generate and display the visualization
visualizer.project(
    data=data,
    labels=labels)
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_blobs.gif?raw=true)
#### Wine dataset
```python
wine = datasets.load_wine()

visualizer.project(
    data=wine['data'],
    labels=wine['target'])
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_wine.gif?raw=true)
#### Cancer dataset
```python
cancer = datasets.load_breast_cancer()

visualizer.project(
    data=cancer['data'],
    labels=cancer['target'])
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_cancer.gif?raw=true)
#### Digits dataset (no labels)
```python
digits = datasets.load_digits(n_class=5)

visualizer.project(
    data=digits['data'],
    meta=digits['target'])
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_digits_no_labels.gif?raw=true)
#### Digits dataset (apply labels)
```python
visualizer.project(
    data=digits['data'],
    labels=digits['target'])
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_digits_labels.gif?raw=true)
#### Housing dataset (labels from 'age')
```python
housing = datasets.fetch_california_housing()

visualizer.project(
    data=housing['data'],
    labels=pd.qcut(housing['data'][:, 1], 5).astype(str))
```
#### Browser tab: 
![image](https://github.com/woldemarg/proshpera/blob/main/demo/gifs/crop_housing.gif?raw=true)

## Change renderer
You can set renderer as ```visualizer = Projector(renderer='iframe')``` to save the plot locally as HTML.
Available renderers:
- 'jupyterlab'
- 'vscode'
- 'notebook'
- 'kaggle'
- 'colab'
and [others](https://plotly.com/python/renderers/)



