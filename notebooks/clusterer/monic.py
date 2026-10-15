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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Monitoring cluster transitions with MONIC
#
# A clusterer tells you what the clusters are now. MONIC tells you what happened to
# them since last time: which clusters *survived*, which were *absorbed* by another,
# which were *split*, which *disappeared* and which *emerged*. For the survivors it
# also reports *internal* transitions: whether a cluster shrank or expanded, became
# more compact or more diffuse, or shifted.
#
# This tutorial runs MONIC on a stream whose clusters change at known moments, so
# that the reported transitions can be checked against what really happened.
#
# * Spiliopoulou, M., Ntoutsi, I., Theodoridis, Y., & Schult, R. (2006).
#   MONIC: modeling and monitoring cluster transitions. KDD '06.
#
# <span style="color:red">This is an experimental API might change significantly in the near future.</span>
#
# ---
#
# *More information about CapyMOA can be found at* https://www.capymoa.org.
#
# **last update on 16/10/2026**

# %% [markdown]
# ## A stream with known transitions
#
# The stream is a mixture of two-dimensional Gaussian clusters. Each instance is
# drawn from one of the clusters active at that moment.
#
# | Instance | What happens |
# |---|---|
# | 0 | Clusters at (0, 0), (12, 0), (0, 12) and (-12, 0) |
# | 2000 | A cluster emerges at (0, -12) |
# | 3000 | The cluster at (12, 0) is replaced by two, at (12, 3.5) and (12, -4.2) |
# | 5000 | The cluster at (0, 12) dies |
# | 7000 | The cluster at (-12, 0) is replaced by three, 4 away from it |
#
# The first replacement is lopsided on purpose: (12, 3.5) is nearer to the old
# cluster than (12, -4.2), so it inherits about three quarters of the old instances.

# %%
import math

import numpy as np

from capymoa.stream import NumpyStream

N_INSTANCES = 12_000
CHECK_INTERVAL = 1000


def active_centres(i: int) -> list[tuple[float, float]]:
    """The centres of the clusters that instance ``i`` can be drawn from."""
    centres = [(0.0, 0.0)]
    if i < 3000:
        centres += [(12.0, 0.0)]
    else:
        centres += [(12.0, 3.5), (12.0, -4.2)]
    if i < 5000:
        centres += [(0.0, 12.0)]
    if i < 7000:
        centres += [(-12.0, 0.0)]
    else:
        centres += [
            (-12.0 + 4 * math.cos(math.radians(a)), 4 * math.sin(math.radians(a)))
            for a in (90, 210, 330)
        ]
    if i >= 2000:
        centres += [(0.0, -12.0)]
    return centres


def drifting_stream(seed: int = 1) -> NumpyStream:
    rng = np.random.default_rng(seed)
    points = np.empty((N_INSTANCES, 2))
    for i in range(N_INSTANCES):
        centres = active_centres(i)
        points[i] = rng.normal(centres[rng.integers(len(centres))], 0.5)
    # NumpyStream needs a target. It is not used.
    return NumpyStream(
        points, np.zeros(N_INSTANCES, dtype=int), target_type="categorical"
    )


# %% [markdown]
# ## A clusterer that MONIC can monitor
#
# MONIC compares two clusterings instance by instance: the overlap of two clusters is
# the share of one cluster's instances found in the other. So a clusterer can be
# monitored only if it can say which instances are in which cluster. It says so by
# implementing `MonitorableClusterer`, whose one method, `get_clustering()`,
# returns a `Clustering` of the current state of its model:
#
# - `clusters` maps each cluster id to the ids of its instances. An instance must
#   keep its id from one clustering to the next.
# - `weights` gives every instance the model still holds an ageing weight in
#   `[0, 1]`, including instances that are in no cluster. An instance left out has
#   aged out.
# - `points` gives the instances' features. They are used for the internal
#   transitions.
# - `timepoint` must increase from one clustering to the next.
#
# The clusterers that come with CapyMOA report cluster centres but not their members,
# so this tutorial writes a small clusterer of its own.
#
# - It keeps the instances of the last two timepoints, each with weight 1, like the
#   sliding window of size 2 in the paper. An instance's id is `instance.id`, which
#   CapyMOA gives every instance when it is created.
# - Its clusters come from the most recent timepoint only. Each instance joins the
#   nearest centre within `radius`, moving it to the mean of its members, or else
#   starts a new centre. Centres with fewer than `min_size` members are dropped.
# - Every instance it keeps belongs to its nearest centre, unless that is farther
#   than `max_distance`. Such an instance is noise: it is still held, but in no
#   cluster.

# %%
from collections import deque

from capymoa.base import Clusterer, ClusteringResult
from capymoa.core import Instance
from capymoa.evaluation.monic import Clustering, MonitorableClusterer


class WindowedLeader(Clusterer, MonitorableClusterer):
    def __init__(
        self,
        schema,
        window: int = 2000,
        recent: int = 1000,
        radius: float = 4.0,
        max_distance: float = 6.0,
        min_size: int = 20,
    ):
        super().__init__(schema)
        self.recent = recent
        self.radius = radius
        self.max_distance = max_distance
        self.min_size = min_size
        self.instances_seen = 0
        # The id and the features of each instance held, oldest first.
        self.window: deque[tuple[int, np.ndarray]] = deque(maxlen=window)

    def __str__(self) -> str:
        return "WindowedLeader"

    def train(self, instance: Instance) -> None:
        self.window.append((instance.id, instance.x.copy()))
        self.instances_seen += 1

    def _leaders(self) -> tuple[list[np.ndarray], list[int]]:
        """The centres found in the most recent instances, and their sizes."""
        centres: list[np.ndarray] = []
        sizes: list[int] = []
        for _, x in list(self.window)[-self.recent :]:
            distances = [np.linalg.norm(x - centre) for centre in centres]
            if distances and min(distances) <= self.radius:
                nearest = int(np.argmin(distances))
                sizes[nearest] += 1
                centres[nearest] += (x - centres[nearest]) / sizes[nearest]
            else:
                centres.append(x.copy())
                sizes.append(1)
        kept = [i for i, size in enumerate(sizes) if size >= self.min_size]
        return [centres[i] for i in kept], [sizes[i] for i in kept]

    def get_clustering(self) -> Clustering:
        ids = [record for record, _ in self.window]
        points = np.array([x for _, x in self.window])
        centres = np.array(self._leaders()[0])
        clusters: dict[int, set[int]] = {}
        if len(ids) > 0 and len(centres) > 0:
            distances = np.linalg.norm(points[:, np.newaxis] - centres, axis=2)
            for record, nearest, distance in zip(
                ids, distances.argmin(axis=1), distances.min(axis=1)
            ):
                if distance <= self.max_distance:
                    clusters.setdefault(int(nearest), set()).add(record)
        return Clustering(
            timepoint=self.instances_seen,
            clusters=clusters,
            weights=dict.fromkeys(ids, 1.0),
            points=dict(zip(ids, points)),
        )

    # The rest of the Clusterer interface. The centres are macro-clusters.
    def implements_macro_clusters(self) -> bool:
        return True

    def implements_micro_clusters(self) -> bool:
        return False

    def _get_clusters_centers(self):
        return [centre.tolist() for centre in self._leaders()[0]]

    def _get_clusters_weights(self):
        return [float(size) for size in self._leaders()[1]]

    def _get_clusters_radii(self):
        return [self.radius] * len(self._leaders()[0])

    def get_clustering_result(self):
        return ClusteringResult(
            self._get_clusters_centers(),
            self._get_clusters_weights(),
            self._get_clusters_radii(),
            [],
        )

    def _get_micro_clusters_centers(self):
        return []

    def _get_micro_clusters_weights(self):
        return []

    def _get_micro_clusters_radii(self):
        return []

    def get_micro_clustering_result(self):
        return ClusteringResult([], [], [], [])


# %% [markdown]
# ## Running the monitor
#
# `MONICMonitor` trains the clusterer on the stream. Every `check_interval`
# instances is a *timepoint*, at which it asks the clusterer for its clustering,
# compares it with the one of the previous timepoint and returns a
# `TransitionReport`. `debug=True` checks the invariants of MONIC at every timepoint
# and raises an `AssertionError` if one is violated.

# %%
from capymoa.evaluation.monic import ExternalKind, MONICMonitor

stream = drifting_stream()
clusterer = WindowedLeader(stream.get_schema())
monitor = MONICMonitor(
    clusterer,
    check_interval=CHECK_INTERVAL,
    tau=0.5,
    tau_split=0.1,
    debug=True,
)


def where(clustering: Clustering, cluster_id: int) -> str:
    """Describe a cluster by where it is, since its id means nothing over time."""
    points = [clustering.points[record] for record in clustering.clusters[cluster_id]]
    x, y = np.mean(points, axis=0).round().astype(int)
    return f"({x}, {y})"


# %% [markdown]
# ### Transition log
#
# Each timepoint is headed by the passforward ratio of the previous clustering: the
# portion of its clusters that survived or were absorbed.

# %%
clusterings = []
reports = []
previous = None
for timepoint, report in enumerate(monitor.monitor(stream)):
    clustering = monitor.clustering
    clusterings.append(clustering)
    reports.append(report)

    internal = {
        t.source: sorted(kind.name for kind in t.kinds) for t in report.internal
    }
    print(
        f"timepoint {timepoint:2d}  "
        f"instances {monitor.instances_seen:5d}  "
        f"clusters {len(clustering.clusters)}  "
        f"passforward {report.ratios.passforward:.2f}"
    )
    for transition in report.external:
        source = (
            ""
            if previous is None or transition.source is None
            else where(previous, transition.source)
        )
        targets = ", ".join(where(clustering, target) for target in transition.targets)
        changes = " ".join(internal.get(transition.source, []))
        print(f"    {transition.kind.name:<12}{source:>10} -> {targets:<28}{changes}")
    previous = clustering

# %% [markdown]
# Compare the log with the table at the top.
#
# - **Timepoint 2**: the cluster at (0, -12) emerges.
# - **Timepoint 3**: the cluster at (12, 0) is reported as *surviving*, and a second
#   cluster as emerged, although the data says it was replaced by two. With `tau=0.5`
#   that is what the definition gives: three quarters of the old cluster went to one
#   place, which makes that place its match. It is a split only for a `tau` above
#   that share, as the sweep below shows.
# - **Timepoint 5**: the cluster at (0, 12) disappears. Its instances of the previous
#   timepoint are still held, but they are farther than `max_distance` from every
#   centre, so they are in no cluster.
# - **Timepoint 7**: the cluster at (-12, 0) is split in three.
#
# A split is seen because the clusterer holds the instances of two timepoints but
# finds its centres in the last one only. After a change, the instances of the
# previous timepoint are assigned to the new centres, and that is how MONIC traces
# where an old cluster went. MONIC describes what the clusterer's model does, so a
# transition is seen when the model reacts, not when the data changes.
#
# The clusters are located by the mean of their instances, which lags behind the
# data for one timepoint after a change: (12, 2) at timepoint 3 is the cluster the
# stream puts at (12, 3.5), still holding instances of the old one.
#
# The survivors also carry internal transitions. Most are `SHRUNK` or `EXPANDED`:
# each instance is drawn from one of the active clusters, so every cluster's share
# of the stream changes whenever the number of clusters does.

# %% [markdown]
# ### Passforward ratio per timepoint

# %%
import pandas as pd

pd.DataFrame(
    {
        "survival": [report.ratios.survival for report in reports],
        "absorption": [report.ratios.absorption for report in reports],
        "passforward": [report.ratios.passforward for report in reports],
    }
).rename_axis("timepoint").round(2)

# %% [markdown]
# ### Overlaps behind a transition
#
# Every report carries the overlap matrix it was derived from: one row per cluster of
# the previous timepoint, one column per cluster of the current one. A row is the
# share of a cluster's instances found in each current cluster.
#
# These are the rows of the two clusters that were replaced.


# %%
def overlaps_of(timepoint: int, location: str) -> pd.Series:
    source, target = clusterings[timepoint - 1], clusterings[timepoint]
    row = [where(source, cluster_id) for cluster_id in source.clusters].index(location)
    columns = [where(target, cluster_id) for cluster_id in target.clusters]
    overlaps = pd.Series(reports[timepoint].overlap[row], index=columns)
    return overlaps[overlaps > 0].round(2)


print(overlaps_of(3, "(12, 0)").to_string())
print()
print(overlaps_of(7, "(-12, 0)").to_string())

# %% [markdown]
# ## The effect of the thresholds
#
# `tau` is how much of a cluster must land in one place for that place to be its
# match. `tau_split` is how much must land in a place for it to count as a piece of
# a split.
#
# The clusterings do not depend on the thresholds, so there is no need to run the
# clusterer again. `MONIC` takes clusterings directly, and the ones collected above
# can be replayed with other thresholds.

# %%
from capymoa.evaluation.monic import MONIC


def count_transitions(tau: float, tau_split: float) -> dict[str, float]:
    """Count the transitions between timepoints for the given thresholds."""
    monic = MONIC(tau=tau, tau_split=tau_split, debug=True)
    counts = dict.fromkeys(ExternalKind, 0)
    for clustering in clusterings:
        report = monic.update(clustering)
        if report.source_timepoint is None:
            # The first clustering has nothing to compare with.
            continue
        for transition in report.external:
            counts[transition.kind] += 1
    return {"tau": tau, "tau_split": tau_split} | {
        kind.name.lower(): count for kind, count in counts.items()
    }


pd.DataFrame(
    [count_transitions(tau, 0.1) for tau in (0.5, 0.6, 0.7, 0.8, 0.9)]
).set_index(["tau", "tau_split"])

# %%
pd.DataFrame(
    [count_transitions(0.5, tau_split) for tau_split in (0.1, 0.2, 0.3, 0.31, 0.35)]
).set_index(["tau", "tau_split"])

# %% [markdown]
# Both tables can be read off the overlaps above.
#
# - **Raising `tau`** turns survivals into splits. The lopsided replacement has a
#   largest piece of 0.75, a match until `tau=0.7` and a split from `tau=0.8`. The
#   three-way replacement has a largest piece of 0.39, so it is a split at every
#   `tau` shown. Nothing disappears, because the instances of a replaced cluster all
#   end up in some new cluster.
# - **Raising `tau_split`** drops pieces from a split. The three-way pieces are 0.39,
#   0.31 and 0.30. At `tau_split=0.31` the piece of 0.30 stops counting and is
#   reported as emerged instead, but the other two still hold 0.70 of the old
#   cluster, at least `tau`, so it is still a split. At `tau_split=0.35` only the
#   piece of 0.39 is left, which is less than `tau`, so the cluster is reported as
#   disappeared and all three pieces as emerged.
#
# A larger `tau` can never add a match, and a larger `tau_split` can never add a
# split: that holds on any stream. Where the changes happen depends on the stream.
