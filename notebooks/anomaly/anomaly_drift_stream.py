# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Anomaly streams with concept drift
#
# This notebook shows how to make and plot an `AnomalyDriftStream`, a stream for anomaly detection where the normal data changes over time.
#
# * Instances from the concepts are **normal**.
# * A share of the stream is replaced with **anomalies**, drawn uniformly over a box.
# * The target is `is_anomaly`: `0` for a normal instance and `1` for an anomaly.
#
# In every plot, normal instances are blue dots and anomalies are red crosses.
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 07/10/2026**

# %% [markdown]
# ## Make the stream
#
# * An `AnomalyDriftStream` is defined like a `DriftStream`: a list of concepts with a drift between each pair.
# * Here a concept is two clusters of points. The clusters sit in different places in each concept, so the normal data moves at the drift.
# * Any stream can be a concept. Its class labels are not used.

# %%
import numpy as np
from sklearn.datasets import make_blobs

from capymoa.stream import NumpyStream
from capymoa.stream.drift import AbruptDrift, AnomalyDriftStream, GradualDrift


def blobs(centers, seed, num_instances=5000):
    """A concept with one cluster of normal data around each center."""
    x, _ = make_blobs(
        n_samples=num_instances, centers=centers, cluster_std=0.7, random_state=seed
    )
    labels = np.zeros(num_instances, dtype=int)
    return NumpyStream(x, labels, "Blobs", feature_names=["x1", "x2"])


def concepts_with(drift):
    """Two concepts with ``drift`` between them."""
    return [
        blobs(centers=[(-3, 0), (0, 3)], seed=1),
        drift,
        blobs(centers=[(3, 0), (0, -3)], seed=2),
    ]


stream = AnomalyDriftStream(
    stream=concepts_with(AbruptDrift(position=2000)),
    anomaly_rate=0.05,
)
print(stream.get_schema())

# %% [markdown]
# ## Read the stream
#
# * `read_stream` reads the first instances of a stream into arrays, so they are easy to plot.
# * About 5% of the instances are anomalies, as set by `anomaly_rate`.

# %%
import pandas as pd


def read_stream(stream, max_instances=4000):
    """Return the features and the labels of the first instances of a stream."""
    stream.restart()
    x, y = [], []
    while stream.has_more_instances() and len(y) < max_instances:
        instance = stream.next_instance()
        x.append(instance.x)
        y.append(instance.y_index)
    return np.array(x), np.array(y)


x, y = read_stream(stream)
print(f"{len(y)} instances, {y.sum()} anomalies ({y.mean():.1%})")
pd.DataFrame(x, columns=["x1", "x2"]).assign(is_anomaly=y).head(10)

# %% [markdown]
# ## Plot the data points
#
# * `plot_concepts` splits the stream at each drift and plots the first two features of each part.
# * The clusters of normal data move at the drift. The anomalies do not: they are spread over the same box before and after.
# * Some anomalies land inside a cluster. No detector can tell those from normal instances.

# %%
import matplotlib.pyplot as plt

NORMAL_COLOR = "#2a78d6"  # blue
ANOMALY_COLOR = "#e34948"  # red


def plot_points(ax, horizontal, vertical, y):
    """Draw normal instances as blue dots and anomalies as red crosses."""
    is_anomaly = y == 1
    ax.scatter(
        horizontal[~is_anomaly],
        vertical[~is_anomaly],
        s=8,
        color=NORMAL_COLOR,
        alpha=0.5,
        linewidths=0,
        label="normal",
    )
    ax.scatter(
        horizontal[is_anomaly],
        vertical[is_anomaly],
        s=36,
        color=ANOMALY_COLOR,
        marker="x",
        linewidths=1.5,
        label="anomaly",
        zorder=3,
    )
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def add_legend(fig, ax):
    """Put one legend under the plots."""
    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=2,
        frameon=False,
        markerscale=1.5,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1)


def plot_concepts(stream, max_instances=4000):
    """Plot the first two features, with one panel for each part of the stream."""
    x, y = read_stream(stream, max_instances)
    names = stream.get_schema().get_numeric_attributes()
    edges = [0, *(drift.position for drift in stream.get_drifts()), len(y)]
    num_parts = len(edges) - 1

    fig, axes = plt.subplots(
        1,
        num_parts,
        figsize=(5 * num_parts, 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, start, end in zip(axes[0], edges, edges[1:]):
        plot_points(ax, x[start:end, 0], x[start:end, 1], y[start:end])
        ax.set_title(f"Instances {start + 1} to {end}")
        ax.set_xlabel(names[0])
    axes[0, 0].set_ylabel(names[1])
    add_legend(fig, axes[0, 0])
    plt.show()


plot_concepts(stream)

# %% [markdown]
# ## Plot the stream over time
#
# * `plot_over_time` plots each feature against the position in the stream.
# * The drift is marked where the stream says it is, using `stream.get_drifts()`.
# * An anomaly takes the place of an instance, so the drift stays at position 2000.

# %%
def plot_over_time(stream, max_instances=4000):
    """Plot the first two features against the position in the stream."""
    x, y = read_stream(stream, max_instances)
    names = stream.get_schema().get_numeric_attributes()
    position = np.arange(1, len(y) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, layout="constrained")
    for feature, ax in enumerate(axes):
        plot_points(ax, position, x[:, feature], y)
        ax.set_ylabel(names[feature])
        for drift in stream.get_drifts():
            if drift.width:
                # A gradual drift is a window, centered on its position.
                ax.axvspan(
                    drift.position - drift.width / 2,
                    drift.position + drift.width / 2,
                    color="#898781",
                    alpha=0.18,
                    linewidth=0,
                )
            else:
                ax.axvline(drift.position, color="#52514e", linewidth=1)

    for drift in stream.get_drifts():
        axes[0].annotate(
            "drift",
            xy=(drift.position, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color="#52514e",
        )
    axes[1].set_xlabel("Position in the stream")
    add_legend(fig, axes[0])
    plt.show()


plot_over_time(stream)

# %% [markdown]
# ## Gradual drift
#
# * A `GradualDrift` mixes the two concepts over a window, shown here as a gray band.
# * Inside the window, normal instances come from both concepts. The old one fades out as the new one fades in.

# %%
gradual_stream = AnomalyDriftStream(
    stream=concepts_with(GradualDrift(position=2000, width=1000)),
    anomaly_rate=0.05,
)
plot_over_time(gradual_stream)

# %% [markdown]
# ## Change the anomalies
#
# * `anomaly_rate` sets how many anomalies there are.
# * `bounds` sets the box the anomalies are drawn from. When it is not given, the box is measured on the concepts, which is what the plots above show.
# * `random_seed` sets where the anomalies are. Without it, each new stream has different anomalies.
#
# One `(low, high)` pair gives every feature the same range. A wide box puts most anomalies far from the normal data, which makes them easy to find.

# %%
wide_stream = AnomalyDriftStream(
    stream=concepts_with(AbruptDrift(position=2000)),
    anomaly_rate=0.05,
    bounds=(-10, 10),
)
plot_concepts(wide_stream)

# %% [markdown]
# One pair for each feature gives each feature its own range. Here the anomalies stay in a band above the normal data.

# %%
band_stream = AnomalyDriftStream(
    stream=concepts_with(AbruptDrift(position=2000)),
    anomaly_rate=0.05,
    bounds=[(-6, 6), (5, 7)],
)
plot_concepts(band_stream)

# %% [markdown]
# ## Generators as concepts
#
# * A generator works as a concept too. `RandomRBFGenerator` draws points around a set of centers, and a different `model_random_seed` gives different centers.
# * The plot shows the first two features.

# %%
from capymoa.stream.generator import RandomRBFGenerator


def rbf(seed):
    return RandomRBFGenerator(
        model_random_seed=seed, number_of_attributes=2, number_of_centroids=3
    )


rbf_stream = AnomalyDriftStream(
    stream=[rbf(seed=3), AbruptDrift(position=2000), rbf(seed=4)],
    anomaly_rate=0.05,
)
plot_concepts(rbf_stream)

# %% [markdown]
# ### A drift that does not show
#
# * The class labels of the concepts are not used, so a drift only shows when the concepts differ in their **features**.
# * `SEA(function=1)` and `SEA(function=3)` draw features the same way and differ only in how they label them. The two parts of the stream look the same.
# * `SEA` also draws its features uniformly, like the anomalies. Blue and red cover the same square, so there is nothing to detect.

# %%
from capymoa.stream.generator import SEA

sea_stream = AnomalyDriftStream(
    stream=[SEA(function=1), AbruptDrift(position=2000), SEA(function=3)],
    anomaly_rate=0.05,
)
plot_concepts(sea_stream)
