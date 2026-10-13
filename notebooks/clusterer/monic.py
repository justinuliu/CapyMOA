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
# **last update on 12/10/2026**

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
# cluster than (12, -4.2), so it inherits about two thirds of the old instances.

# %%
import math

import numpy as np

from capymoa.stream import NumpyStream

N_INSTANCES = 12_000
SNAPSHOT_INTERVAL = 1000


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
# ## Running the monitor
#
# `MONICMonitor` wraps a clusterer. Every `snapshot_interval` instances is a
# *timepoint*, at which the monitor compares the current clusters with those of the
# previous timepoint and returns a `TransitionReport`.
#
# - CapyMOA clusterers report cluster centres, not which instance belongs to which
#   cluster. The monitor therefore keeps the instances of the last two timepoints and
#   assigns each to its nearest centre.
# - `max_distance` is the noise policy. An instance farther than this from every
#   centre belongs to no cluster. Without it, the instances left behind by a dead
#   cluster are handed to the nearest surviving one, and the death is reported as an
#   absorption.
# - `debug=True` checks the invariants of MONIC at every timepoint and raises an
#   `AssertionError` if one is violated.
# - `DenStream` is set to forget quickly (`lambda_option`, `speed`), so that its
#   clusters follow the data from one timepoint to the next. A transition is seen
#   when the clusterer reacts, not when the data changes.

# %%
from capymoa.cluster import Denstream_with_dbscan
from capymoa.evaluation.monic import ClusteringSnapshot, ExternalKind, MONICMonitor

stream = drifting_stream()
clusterer = Denstream_with_dbscan(
    schema=stream.get_schema(),
    epsilon=0.5,
    init_points=200,
    lambda_option=0.5,
    speed=30,
)
monitor = MONICMonitor(
    clusterer,
    snapshot_interval=SNAPSHOT_INTERVAL,
    max_distance=6.0,
    tau=0.5,
    tau_split=0.1,
    debug=True,
)


def where(snapshot: ClusteringSnapshot, cluster_id: int) -> str:
    """Describe a cluster by where it is, since its id means nothing over time."""
    points = [snapshot.points[record] for record in snapshot.clusters[cluster_id]]
    x, y = np.mean(points, axis=0).round().astype(int)
    return f"({x}, {y})"


# %% [markdown]
# ### Transition log
#
# Each timepoint is headed by the passforward ratio of the previous clustering: the
# portion of its clusters that survived or were absorbed.

# %%
snapshots = []
reports = []
previous = None
for report in monitor.monitor(stream):
    snapshot = monitor.snapshot
    snapshots.append(snapshot)
    reports.append(report)

    internal = {
        t.source: sorted(kind.name for kind in t.kinds) for t in report.internal
    }
    print(
        f"timepoint {report.target_timepoint:2d}  "
        f"instances {monitor.instances_seen:5d}  "
        f"clusters {len(snapshot.clusters)}  "
        f"passforward {report.ratios.passforward:.2f}"
    )
    for transition in report.external:
        source = (
            ""
            if previous is None or transition.source is None
            else where(previous, transition.source)
        )
        targets = ", ".join(where(snapshot, target) for target in transition.targets)
        changes = " ".join(internal.get(transition.source, []))
        print(f"    {transition.kind.name:<12}{source:>10} -> {targets:<28}{changes}")
    previous = snapshot

# %% [markdown]
# Compare the log with the table at the top.
#
# - **Timepoint 2**: the cluster at (0, -12) emerges.
# - **Timepoint 3**: the cluster at (12, 0) is reported as *surviving*, and a second
#   cluster as emerged, although the data says it was replaced by two. With `tau=0.5`
#   that is what the definition gives: two thirds of the old cluster went to one
#   place, which makes that place its match. It is a split only for a `tau` above
#   that share, as the sweep below shows.
# - **Timepoint 5**: the cluster at (0, 12) disappears.
# - **Timepoint 7**: the cluster at (-12, 0) is split in three.
# - **Timepoints 9 and 10**: nothing happens in the data, yet a cluster emerges and
#   is absorbed again. `DenStream` briefly reported one of the clusters as two. MONIC
#   describes what the clusterer's model does, so an unstable model produces
#   transitions of its own.
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
    source, target = snapshots[timepoint - 1], snapshots[timepoint]
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
# clusterer again. `MONIC` takes snapshots directly, and the ones collected above
# can be replayed with other thresholds.

# %%
import warnings

from capymoa.evaluation.monic import MONIC


def count_transitions(tau: float, tau_split: float) -> dict[str, float]:
    with warnings.catch_warnings():
        # MONIC warns that a tau below 0.5 lets less than half a cluster be a match.
        warnings.simplefilter("ignore", UserWarning)
        monic = MONIC(tau=tau, tau_split=tau_split, debug=True)
    counts = dict.fromkeys(ExternalKind, 0)
    for snapshot in snapshots:
        for transition in monic.update(snapshot).external:
            counts[transition.kind] += 1
    return {"tau": tau, "tau_split": tau_split} | {
        kind.name.lower(): count for kind, count in counts.items()
    }


pd.DataFrame(
    [count_transitions(tau, 0.1) for tau in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7)]
).set_index(["tau", "tau_split"])

# %%
pd.DataFrame(
    [
        count_transitions(0.5, tau_split)
        for tau_split in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35)
    ]
).set_index(["tau", "tau_split"])

# %% [markdown]
# Both tables can be read off the overlaps above.
#
# - **Raising `tau`** turns survivals into splits. The three-way replacement has a
#   largest piece of 0.48, a match only at `tau=0.45`. The lopsided replacement has a
#   largest piece of 0.67, a match until `tau=0.7`. Nothing disappears, because the
#   instances of a replaced cluster all end up in some new cluster.
# - **Raising `tau_split`** drops pieces from a split. The piece of 0.19 stops counting
#   at `tau_split=0.2` and is reported as emerged instead. At `tau_split=0.35` only the
#   piece of 0.48 is left, which is less than `tau`, so the cluster is reported as
#   disappeared and all three pieces as emerged.
#
# A larger `tau` can never add a match, and a larger `tau_split` can never add a
# split: that holds on any stream. Where the changes happen depends on the stream.
