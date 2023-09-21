import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
from prosphera.projector import Projector

# %%

# Instantiate the class
visualizer = Projector()

# %%

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

# %%

wine = datasets.load_wine()

visualizer.project(
    data=wine['data'],
    labels=wine['target'])

# %%

cancer = datasets.load_breast_cancer()

visualizer.project(
    data=cancer['data'],
    labels=cancer['target'])

# %%

digits = datasets.load_digits(n_class=5)

visualizer.project(
    data=digits['data'],
    meta=digits['target'])

# %%

visualizer.project(
    data=digits['data'],
    labels=digits['target'])

# %%

housing = datasets.fetch_california_housing()

visualizer.project(
    data=housing['data'],
    labels=pd.qcut(housing['data'][:, 1], 5).astype(str))
