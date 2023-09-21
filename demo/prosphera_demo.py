from sklearn.datasets import make_blobs
from prosphera.projector import Projector

# %%

# Generate data
data, labels = make_blobs(
    n_samples=5000,
    centers=50,
    n_features=25,
    random_state=1234)

# %%

# Instantiate the class
visualizer = Projector()

# Call the visualize method to generate and display the visualization
visualizer.project(data=data, labels=labels)
