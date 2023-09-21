import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import KernelPCA
from sklearn.utils import resample
from sklearn.preprocessing import robust_scale, minmax_scale


# %%

class Projector:
    def __init__(self,
                 random_state=1234,
                 max_points=5000,
                 renderer='browser'):

        self.random_state = random_state
        self.max_points = max_points
        self.renderer = renderer

    def _apply_pca(self, data):

        scaled_data = robust_scale(data, quantile_range=(5, 95))

        pca = KernelPCA(
            n_components=3,
            kernel='cosine',
            copy_X=False,
            random_state=self.random_state,
            n_jobs=-1)

        return pca.fit_transform(scaled_data)

    @staticmethod
    def _scale_vectors_on_sphere(data_pca, scaling_range=(0.1, 1)):

        vectors = data_pca - np.mean(data_pca, axis=0)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        normalized = vectors / norms

        scaled_magnitudes = minmax_scale(
            np.log(norms**2),
            feature_range=scaling_range)

        vecs = normalized * scaled_magnitudes

        return vecs, scaled_magnitudes

    @staticmethod
    def _create_colormap(labels):

        num_colors = len(np.unique(labels))

        palette = sns.color_palette(cc.glasbey_dark, n_colors=num_colors)

        colormap_dict = {
            label: LinearSegmentedColormap.from_list(
                '', [(1, 1, 1), color], N=100)
            for label, color in zip(sorted(np.unique(labels)), palette)}

        return colormap_dict

    def _create_colors(self, opacity_values, labels=None):

        if labels is None:
            colormap = plt.get_cmap('Blues')
            clrs = colormap(opacity_values)

        else:
            colormap_dict = self._create_colormap(labels)
            clrs = [colormap_dict[label](tone)
                    for label, tone in zip(labels, opacity_values)]

        return clrs

    def _get_sample(self, data, **kwargs):

        out = dict(data=data)

        for key in ('meta', 'labels'):
            out[key] = kwargs.get(key, range(data.shape[0]))

        max_points = min(self.max_points, data.shape[0])

        out = dict(zip(out.keys(),
                       resample(
                           *out.values(),
                           replace=False,
                           n_samples=max_points,
                           random_state=self.random_state)))

        return out

    def project(self, **kwargs):

        data = kwargs.pop('data')

        if data is None:
            print('Oops, got no data!')

        try:

            out = self._get_sample(data, **kwargs)

            data_pca = self._apply_pca(out['data'])

            vecs, opacities = self._scale_vectors_on_sphere(data_pca)

            df_output = pd.DataFrame(vecs, columns=list('xyz'))

            labels = out['labels'] if 'labels' in kwargs else None

            colors = self._create_colors(minmax_scale(
                opacities.flatten()), labels)

            if labels is None:
                hover_template = '<b>meta</b> %{text}<extra></extra>'
            else:
                hover_template = '<b>label</b> %{customdata}<br><b>meta</b> %{text}<extra></extra>'

            lines_coords = np.array(
                [[-1, 0, 0],
                 [1, 0, 0],
                 [0, -1, 0],
                 [0, 1, 0],
                 [0, 0, -1],
                 [0, 0, 1]])

            fig = px.scatter_3d(
                df_output,
                x='x',
                y='y',
                z='z',
                custom_data=[out['labels']])

            fig.update_traces(
                marker=dict(
                    size=5,
                    line=dict(width=0.1, color='white'),
                    color=colors),
                text=out['meta'],
                hovertemplate=hover_template,
                selector=dict(mode='markers'))

            for line_coords in lines_coords:

                fig.add_trace(
                    go.Scatter3d(
                        x=[0, line_coords[0]],
                        y=[0, line_coords[1]],
                        z=[0, line_coords[2]],
                        mode='lines',
                        line=dict(color='black', width=0.75),
                        showlegend=False,
                        hoverinfo='none'))

            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-1, 1],
                               showline=True,
                               showgrid=False,
                               showticklabels=False),
                    yaxis=dict(range=[-1, 1],
                               showline=True,
                               showgrid=False,
                               showticklabels=False),
                    zaxis=dict(range=[-1, 1],
                               showline=True,
                               showgrid=False,
                               showticklabels=False)),
                # https://plotly.com/python/figure-labels/
                font_family='PT Sans',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)')

            fig.update_scenes(
                aspectmode='cube',
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False)

            fig.show(renderer=self.renderer)

        except Exception:
            print('Oops, got an error!')
            raise
