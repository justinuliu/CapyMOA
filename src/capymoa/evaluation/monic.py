"""MONIC: modelling and monitoring cluster transitions.

MONIC [#f0]_ observes clusterings produced at successive timepoints and
classifies what happened to each cluster: whether it survived, was absorbed,
was split or disappeared, which clusters emerged, and how the survivors changed.

There are two ways in:

* :py:class:`MONIC` takes one :py:class:`Clustering` per timepoint. Use
  it when you cluster the data of each timepoint yourself, as the paper does.
* :py:class:`MONICMonitor` trains a clusterer on a stream and asks it for its
  clustering at every timepoint. The clusterer must implement
  :py:class:`MonitorableClusterer`.

See ``docs/MONIC-design.md`` for the decisions taken where the paper is silent.

..  [#f0] Spiliopoulou, M., Ntoutsi, I., Theodoridis, Y., & Schult, R. (2006).
    MONIC: modeling and monitoring cluster transitions. In Proceedings of the
    12th ACM SIGKDD international conference on Knowledge discovery and data
    mining (pp. 706-711).
"""

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np

from capymoa.base import Clusterer
from capymoa.core import Instance
from capymoa.stream import Stream

__all__ = [
    "MONIC",
    "Clustering",
    "ExternalKind",
    "ExternalTransition",
    "InternalKind",
    "InternalTransition",
    "Lifetime",
    "MONICMonitor",
    "MonitorableClusterer",
    "PassforwardRatios",
    "TransitionReport",
    "advance_lifetimes",
    "build_overlap_matrix",
    "detect_external_transitions",
    "detect_internal_transitions",
    "overlap",
    "passforward_ratios",
]


@dataclass(frozen=True)
class Clustering:
    """A clustering at one timepoint, with record-level membership.

    The row and column order of an overlap matrix follows the iteration order of
    :py:attr:`clusters`.

    >>> clustering = Clustering(
    ...     timepoint=0,
    ...     clusters={0: {1, 2}, 1: {3}},
    ...     weights={1: 1.0, 2: 1.0, 3: 0.5},
    ... )
    >>> clustering.cluster_ids
    (0, 1)

    :raises ValueError: If a record belongs to more than one cluster, a
        clustered record has no weight or no point, a weight is outside
        ``[0, 1]``, or the points do not all have the same number of features.
    """

    timepoint: float
    """The timepoint at which the clustering was built."""
    clusters: Mapping[int, AbstractSet[int]]
    """Maps a cluster id to the ids of its records. Cluster ids are only
    meaningful within this clustering. Clusters must be disjoint."""
    weights: Mapping[int, float]
    """Maps a record id to its ageing weight ``age(record, timepoint)`` in
    ``[0, 1]``. It covers every record alive at this timepoint, including
    records that belong to no cluster. A record that is absent has weight 0.
    A record of weight 0 has aged out even if it is in a cluster: it adds
    nothing to any overlap and has no weight in the statistics of its cluster."""
    points: Mapping[int, Sequence[float]] | None = field(default=None, compare=False)
    """Maps a record id to its feature vector at this timepoint. Only internal
    transitions need it. If either of two clusterings has no points, the
    internal analysis of every survived cluster between them is skipped, so its
    strict lifetime restarts at 1 (see :py:func:`advance_lifetimes`)."""

    def __post_init__(self) -> None:
        clusters = {
            cluster_id: frozenset(records)
            for cluster_id, records in self.clusters.items()
        }
        weights = dict(self.weights)

        n_memberships = sum(len(records) for records in clusters.values())
        clustered: frozenset[int] = frozenset().union(*clusters.values())
        if len(clustered) != n_memberships:
            raise ValueError("Clusters must be disjoint.")
        if not clustered <= weights.keys():
            raise ValueError("Every clustered record must have a weight.")
        if any(not 0.0 <= weight <= 1.0 for weight in weights.values()):
            raise ValueError("Weights must be in [0, 1].")

        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "weights", weights)

        if self.points is None:
            return
        points = {
            record: np.asarray(point, dtype=float)
            for record, point in self.points.items()
        }
        if not clustered <= points.keys():
            raise ValueError("Every clustered record must have a point.")
        if len({point.shape for point in points.values()}) > 1 or any(
            point.ndim != 1 for point in points.values()
        ):
            raise ValueError("Points must be vectors with the same number of features.")
        object.__setattr__(self, "points", points)

    @property
    def cluster_ids(self) -> tuple[int, ...]:
        """Cluster ids in overlap matrix order."""
        return tuple(self.clusters)

    @property
    def n_features(self) -> int | None:
        """Number of features of the points, or ``None`` if there are none."""
        if not self.points:
            return None
        return len(next(iter(self.points.values())))

    def without_clusters(self, cluster_ids: Iterable[int]) -> "Clustering":
        """Turn the records of some clusters into noise.

        Use it on a garbage cluster: one into which a clusterer puts the records
        it cannot place. Such a cluster is large, so left in place it becomes
        the match of clusters that merely scattered. Its records stay alive but
        belong to no cluster, so they still count against the overlap of the
        clusters they came from.

        >>> clustering = Clustering(0, {0: {1, 2}, 1: {3}}, {1: 1.0, 2: 1.0, 3: 1.0})
        >>> clustering.without_clusters([0]).clusters
        {1: frozenset({3})}

        :param cluster_ids: The ids of the clusters to dissolve.
        :return: A clustering without those clusters and with the same weights.
        """
        dropped = set(cluster_ids)
        clusters = {
            cluster_id: records
            for cluster_id, records in self.clusters.items()
            if cluster_id not in dropped
        }
        return Clustering(self.timepoint, clusters, self.weights, self.points)


def overlap(
    x: AbstractSet[int], y: AbstractSet[int], weights: Mapping[int, float]
) -> float:
    """Overlap of cluster ``x`` to cluster ``y`` (Definition 3).

    The normalised sum of the weights of the records in the intersection:
    ``sum(age(a, t_j) for a in x & y) / sum(age(a, t_j) for a in x)``. Both sums
    use the weights of the later timepoint ``t_j``. Overlap is not symmetric.

    >>> overlap({1, 2, 3, 4}, {3, 4, 5}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    0.5

    :param x: Record ids of a cluster found at the earlier timepoint ``t_i``.
    :param y: Record ids of a cluster found at the later timepoint ``t_j``.
    :param weights: Ageing weights at ``t_j``. An absent record has weight 0.
    :return: The overlap in ``[0, 1]``, or 0 if every record of ``x`` has aged
        out by ``t_j``.
    """
    denominator = sum(weights.get(record, 0.0) for record in x)
    if denominator == 0.0:
        return 0.0
    return sum(weights.get(record, 0.0) for record in x & y) / denominator


def build_overlap_matrix(source: Clustering, target: Clustering) -> np.ndarray:
    """Overlap of every cluster of ``source`` to every cluster of ``target``.

    Uses an inverted index from record id to the cluster of ``target``, then a
    single pass over the records of each cluster of ``source``. Besides
    allocating the matrix, the cost is linear in the number of records rather
    than in the number of cluster pairs.

    >>> source = Clustering(0, {0: {1, 2, 3, 4}}, dict.fromkeys(range(1, 5), 1.0))
    >>> target = Clustering(1, {7: {1, 2, 3}, 8: {4}}, dict.fromkeys(range(1, 5), 1.0))
    >>> build_overlap_matrix(source, target).tolist()
    [[0.75, 0.25]]

    :param source: The clustering at the earlier timepoint ``t_i``.
    :param target: The clustering at the later timepoint ``t_j``, whose weights
        are used for every cluster.
    :return: A matrix of shape ``(len(source.clusters), len(target.clusters))``
        where entry ``[i, j]`` is the overlap of the ``i``-th cluster of
        ``source`` to the ``j``-th cluster of ``target``. A row is all zeros if
        every record of its cluster has aged out.
    """
    intersections, totals = _intersection_weights(source, target)
    return _divide_or_zero(intersections, totals[:, np.newaxis])


def _intersection_weights(
    source: Clustering, target: Clustering
) -> tuple[np.ndarray, np.ndarray]:
    """Numerators and denominators of Definition 3, using the weights of ``target``.

    :return: The weight of the intersection of every cluster of ``source`` with
        every cluster of ``target``, and the weight of every cluster of
        ``source``.
    """
    column_of = {
        record: column
        for column, records in enumerate(target.clusters.values())
        for record in records
    }

    intersections = np.zeros((len(source.clusters), len(target.clusters)))
    totals = np.zeros(len(source.clusters))
    for row, records in enumerate(source.clusters.values()):
        for record in records:
            weight = target.weights.get(record, 0.0)
            totals[row] += weight
            column = column_of.get(record)
            if column is not None:
                intersections[row, column] += weight
    return intersections, totals


def _divide_or_zero(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, giving 0 wherever the denominator is 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


class ExternalKind(Enum):
    """The external transitions of a cluster (Table 1)."""

    SURVIVED = "survived"
    """The cluster has a match that is the match of no other cluster."""
    ABSORBED = "absorbed"
    """The cluster has a match that is also the match of another cluster."""
    SPLIT = "split"
    """The cluster has no match, but several clusters together form one."""
    DISAPPEARED = "disappeared"
    """None of the other cases hold for the cluster."""
    EMERGED = "emerged"
    """A new cluster that is not the result of another external transition."""


@dataclass(frozen=True)
class ExternalTransition:
    """An external transition between two clusterings."""

    kind: ExternalKind
    source: int | None
    """The id of the cluster of the earlier clustering that underwent the
    transition, or ``None`` for an emerged cluster."""
    targets: tuple[int, ...]
    """The ids of the resulting clusters of the later clustering: one for a
    survived, absorbed or emerged cluster, several for a split and none for a
    disappeared cluster."""


def detect_external_transitions(
    source: Clustering,
    target: Clustering,
    tau: float = 0.5,
    tau_split: float = 0.1,
) -> tuple[ExternalTransition, ...]:
    """Detect the external transitions from ``source`` to ``target`` (Table 1).

    A cluster of ``source`` with a match in ``target`` survives, or is absorbed
    if another cluster has the same match. The match is the cluster with the
    largest overlap, provided that overlap is at least ``tau``. Ties are broken
    by the larger reverse overlap, then by the order of the clusters in
    ``target``. A cluster without a match is split if the clusters it overlaps
    by at least ``tau_split`` together overlap it by at least ``tau``; otherwise
    it disappears.

    >>> source = Clustering(0, {0: {1, 2}, 1: {3, 4}}, dict.fromkeys(range(1, 5), 1.0))
    >>> target = Clustering(1, {7: {1, 2, 5}, 8: {6}}, {1: 1.0, 2: 1.0, 5: 1.0, 6: 1.0})
    >>> for transition in detect_external_transitions(source, target):
    ...     print(transition.kind.name, transition.source, transition.targets)
    SURVIVED 0 (7,)
    DISAPPEARED 1 ()
    EMERGED None (8,)

    :param source: The clustering at the earlier timepoint ``t_i``.
    :param target: The clustering at the later timepoint ``t_j``.
    :param tau: The least overlap for one cluster to be the match of another.
        Definition 4 restricts it to ``[0.5, 1]``; a smaller value is accepted
        with a warning.
    :param tau_split: The least overlap for a cluster to take part in a split.
    :raises ValueError: Unless ``0 < tau_split < tau <= 1``.
    :return: One transition for each cluster of ``source``, in order, followed
        by one for each emerged cluster of ``target``, in order.
    """
    _check_external_thresholds(tau, tau_split)
    return _external_transitions(source, target, tau, tau_split)


def _check_external_thresholds(tau: float, tau_split: float) -> None:
    """Validate the thresholds on behalf of a public function or constructor."""
    if not 0.0 < tau_split < tau <= 1.0:
        raise ValueError(
            f"Thresholds must satisfy 0 < tau_split < tau <= 1, but got "
            f"tau_split={tau_split} and tau={tau}."
        )
    if tau < 0.5:
        warnings.warn(
            f"tau={tau} is below 0.5, so a cluster may be matched by less than "
            "half of its weight.",
            stacklevel=3,
        )


def _external_transitions(
    source: Clustering, target: Clustering, tau: float, tau_split: float
) -> tuple[ExternalTransition, ...]:
    """:py:func:`detect_external_transitions` without the threshold checks."""
    intersections, totals = _intersection_weights(source, target)
    forward = _divide_or_zero(intersections, totals[:, np.newaxis])
    target_totals = np.array(
        [
            sum(target.weights[record] for record in records)
            for records in target.clusters.values()
        ]
    )
    reverse = _divide_or_zero(intersections, target_totals[np.newaxis, :])

    # Pass 1: find the match of each cluster of ``source``, or failing that the
    # clusters it is split into. A cluster with a match is never split.
    match_of: dict[int, int] = {}
    split_of: dict[int, list[int]] = {}
    for row in range(len(source.clusters)):
        candidates = np.flatnonzero(forward[row] >= tau).tolist()
        if candidates:
            match_of[row] = max(
                candidates,
                key=lambda column: (
                    forward[row, column],
                    reverse[row, column],
                    -column,
                ),
            )
            continue
        candidates = np.flatnonzero(forward[row] >= tau_split).tolist()
        # Clusters are disjoint, so the overlap to their union is the sum of
        # their intersection weights over the weight of the cluster.
        if candidates and intersections[row, candidates].sum() / totals[row] >= tau:
            split_of[row] = candidates

    # Pass 2: a cluster of ``target`` matched by one cluster is its survival,
    # and one matched by several has absorbed them.
    n_matched_by = Counter(match_of.values())

    target_ids = target.cluster_ids
    transitions = []
    produced: set[int] = set()
    for row, source_id in enumerate(source.clusters):
        if row in match_of:
            column = match_of[row]
            absorbed = n_matched_by[column] > 1
            kind = ExternalKind.ABSORBED if absorbed else ExternalKind.SURVIVED
            targets: tuple[int, ...] = (target_ids[column],)
        elif row in split_of:
            kind = ExternalKind.SPLIT
            targets = tuple(target_ids[column] for column in split_of[row])
        else:
            kind = ExternalKind.DISAPPEARED
            targets = ()
        produced.update(targets)
        transitions.append(ExternalTransition(kind, source_id, targets))

    transitions.extend(
        ExternalTransition(ExternalKind.EMERGED, None, (target_id,))
        for target_id in target_ids
        if target_id not in produced
    )
    return tuple(transitions)


class InternalKind(Enum):
    """The internal transitions of a survived cluster (Table 2).

    There are three groups: size, compactness and location. A cluster undergoes
    at most one transition of each group.
    """

    SHRUNK = "shrunk"
    """Size: the cluster lost weight."""
    EXPANDED = "expanded"
    """Size: the cluster gained weight."""
    COMPACTED = "compacted"
    """Compactness: the cluster became more compact."""
    DIFFUSED = "diffused"
    """Compactness: the cluster became more diffuse."""
    SHIFTED = "shifted"
    """Location: the mean or the skewness of the cluster moved."""


@dataclass(frozen=True)
class InternalTransition:
    """The internal transitions of one survived cluster."""

    source: int
    """The id of the cluster in the earlier clustering."""
    target: int
    """The id of the cluster in the later clustering."""
    kinds: frozenset[InternalKind]
    """The transitions the cluster underwent. Empty if it did not change, and
    also if the analysis was :py:attr:`skipped`."""
    skipped: bool = False
    """``True`` if the cluster could not be analysed, because the two
    clusterings do not both carry points with the same number of features or
    because the cluster has no weight."""


class _Statistics(NamedTuple):
    weight: float
    mean: np.ndarray
    spread: float
    skewness: np.ndarray


def _statistics(clustering: Clustering, cluster_id: int) -> _Statistics | None:
    """Weighted statistics of a cluster, using the weights of its own clustering.

    :return: The weight of the cluster, its mean, the root mean square distance
        of its records to the mean and its skewness in each feature, or ``None``
        if the cluster has no weight.
    """
    records = sorted(clustering.clusters[cluster_id])
    weights = np.array([clustering.weights[record] for record in records])
    total = weights.sum()
    if total == 0.0:
        return None

    points = np.array([clustering.points[record] for record in records])
    mean = weights @ points / total
    centred = points - mean
    variance = weights @ centred**2 / total
    third_moment = weights @ centred**3 / total
    # A constant feature can have a variance of rounding error, which would
    # turn its skewness into noise.
    has_variance = np.sqrt(variance) > 1e-12 * np.abs(points).max(axis=0)
    skewness = np.divide(
        third_moment,
        variance**1.5,
        out=np.zeros_like(variance),
        where=has_variance,
    )
    return _Statistics(total, mean, float(np.sqrt(variance.sum())), skewness)


def _check_internal_thresholds(
    epsilon: float, delta: float, tau_mean: float, tau_skew: float
) -> None:
    if min(epsilon, delta, tau_mean, tau_skew) < 0.0:
        raise ValueError("Thresholds must be non-negative.")


def detect_internal_transitions(
    source: Clustering,
    target: Clustering,
    transitions: Iterable[ExternalTransition],
    epsilon: float = 0.1,
    delta: float = 0.1,
    tau_mean: float = 0.5,
    tau_skew: float = 0.5,
) -> tuple[InternalTransition, ...]:
    """Detect the internal transitions of the survived clusters (Table 2).

    A cluster is described by its weight, its mean, its spread ``sigma`` (the
    root mean square distance of its records to the mean) and its skewness in
    each feature. Each cluster is weighted by its own clustering. Unlike in the
    paper, the thresholds on weight, spread and mean are relative, so that they
    do not depend on the size of the stream or the scale of the features.

    The statistics are undefined across a change of feature space. If the two
    clusterings do not both carry points with the same number of features, every
    cluster is reported as skipped instead.

    >>> source = Clustering(
    ...     0, {0: {1, 2}}, {1: 1.0, 2: 1.0}, {1: [0.0], 2: [2.0]}
    ... )
    >>> target = Clustering(
    ...     1, {7: {1, 2, 3}}, {1: 1.0, 2: 1.0, 3: 1.0}, {1: [0.0], 2: [2.0], 3: [1.0]}
    ... )
    >>> external = detect_external_transitions(source, target)
    >>> (internal,) = detect_internal_transitions(source, target, external)
    >>> sorted(kind.name for kind in internal.kinds)
    ['COMPACTED', 'EXPANDED']

    :param source: The clustering at the earlier timepoint ``t_i``.
    :param target: The clustering at the later timepoint ``t_j``.
    :param transitions: The external transitions from ``source`` to ``target``.
        Only the survived clusters are analysed.
    :param epsilon: The cluster shrinks or expands if its weight changes by more
        than this fraction.
    :param delta: The cluster becomes more compact or more diffuse if its
        spread changes by more than this fraction.
    :param tau_mean: The cluster shifts if its mean moves by more than this
        fraction of its spread.
    :param tau_skew: The cluster shifts if its skewness changes by more than
        this in any feature.
    :raises ValueError: If a threshold is negative.
    :return: One internal transition for each survived cluster, in order.
    """
    _check_internal_thresholds(epsilon, delta, tau_mean, tau_skew)

    comparable = (
        source.n_features is not None and source.n_features == target.n_features
    )

    result = []
    for transition in transitions:
        if transition.kind is not ExternalKind.SURVIVED:
            continue
        source_id = transition.source
        (target_id,) = transition.targets
        before = _statistics(source, source_id) if comparable else None
        after = _statistics(target, target_id) if comparable else None
        if before is None or after is None:
            result.append(
                InternalTransition(source_id, target_id, frozenset(), skipped=True)
            )
            continue

        kinds = set()
        if after.weight < (1 - epsilon) * before.weight:
            kinds.add(InternalKind.SHRUNK)
        elif after.weight > (1 + epsilon) * before.weight:
            kinds.add(InternalKind.EXPANDED)

        if after.spread < (1 - delta) * before.spread:
            kinds.add(InternalKind.COMPACTED)
        elif after.spread > (1 + delta) * before.spread:
            kinds.add(InternalKind.DIFFUSED)

        mean_shift = np.linalg.norm(after.mean - before.mean)
        skew_shift = np.abs(after.skewness - before.skewness).max()
        if mean_shift > tau_mean * before.spread or skew_shift > tau_skew:
            kinds.add(InternalKind.SHIFTED)

        result.append(InternalTransition(source_id, target_id, frozenset(kinds)))
    return tuple(result)


@dataclass(frozen=True)
class Lifetime:
    """The lifetimes of a cluster (Definition 5).

    Each is a number of timepoints counted forward from the emergence of the
    cluster, up to and including the timepoint of the clustering it belongs to.
    A new cluster has a lifetime of 1. ``strict <= internal <= absorption``
    always holds.
    """

    strict: int = 1
    """The number of timepoints since the cluster emerged or last underwent an
    internal transition (``lifetimeS``)."""
    internal: int = 1
    """The number of timepoints the cluster has survived, with or without
    internal transitions (``lifetimeI``)."""
    absorption: int = 1
    """Like :py:attr:`internal`, but also counting the timepoints at which the
    cluster absorbed others (``lifetimeA``)."""


def advance_lifetimes(
    lifetimes: Mapping[int, Lifetime],
    source: Clustering,
    target: Clustering,
    external: Iterable[ExternalTransition],
    internal: Iterable[InternalTransition],
) -> dict[int, Lifetime]:
    """Lifetimes of the clusters of ``target``, given those of ``source``.

    The paper computes lifetimes backward from the last clustering. A stream
    has no last clustering, so they are counted forward instead:

    * A cluster that survives passes all three lifetimes on, each extended by
      one. The strict lifetime restarts at 1 if the cluster underwent an
      internal transition, or if it was not shown to be free of one because
      its internal analysis was skipped or is missing from ``internal``.
    * A cluster that absorbed others continues the absorption lifetime of the
      one sharing the most weight with it, extended by one. Ties go to the
      longer absorption lifetime, then to the order of ``source``. Its other
      lifetimes restart at 1.
    * Every other cluster is new: it emerged, or it is the result of a split.
      A split extends no lifetime.

    Call it with an empty ``source`` to start a history:

    >>> nothing = Clustering(0, {}, {})
    >>> first = Clustering(1, {0: {1, 2}}, {1: 1.0, 2: 1.0})
    >>> external = detect_external_transitions(nothing, first)
    >>> advance_lifetimes({}, nothing, first, external, ())
    {0: Lifetime(strict=1, internal=1, absorption=1)}

    :param lifetimes: The lifetime of every cluster of ``source``, by cluster id.
    :param source: The clustering at the earlier timepoint ``t_i``.
    :param target: The clustering at the later timepoint ``t_j``.
    :param external: The external transitions from ``source`` to ``target``.
    :param internal: The internal transitions from ``source`` to ``target``.
    :return: The lifetime of every cluster of ``target``, by cluster id.
    """
    unchanged = {
        transition.source
        for transition in internal
        if not transition.skipped and not transition.kinds
    }
    intersections, _ = _intersection_weights(source, target)
    row_of = {cluster_id: row for row, cluster_id in enumerate(source.clusters)}
    column_of = {
        cluster_id: column for column, cluster_id in enumerate(target.clusters)
    }

    continued: dict[int, Lifetime] = {}
    absorbed_by: dict[int, list[int]] = {}
    for transition in external:
        if transition.kind is ExternalKind.SURVIVED:
            before = lifetimes[transition.source]
            continued[transition.targets[0]] = Lifetime(
                strict=before.strict + 1 if transition.source in unchanged else 1,
                internal=before.internal + 1,
                absorption=before.absorption + 1,
            )
        elif transition.kind is ExternalKind.ABSORBED:
            absorbed_by.setdefault(transition.targets[0], []).append(transition.source)

    for target_id, source_ids in absorbed_by.items():
        dominant = max(
            source_ids,
            key=lambda source_id: (
                intersections[row_of[source_id], column_of[target_id]],
                lifetimes[source_id].absorption,
                -row_of[source_id],
            ),
        )
        continued[target_id] = Lifetime(absorption=lifetimes[dominant].absorption + 1)

    return {
        target_id: continued.get(target_id, Lifetime()) for target_id in target.clusters
    }


@dataclass(frozen=True)
class PassforwardRatios:
    """The survival and absorption ratios of a clustering (Definition 6)."""

    survival: float
    """The portion of the clusters that survived into the next clustering."""
    absorption: float
    """The portion of the clusters absorbed by a cluster of the next clustering."""

    @property
    def passforward(self) -> float:
        """The sum of the survival and absorption ratios."""
        return self.survival + self.absorption


def passforward_ratios(external: Iterable[ExternalTransition]) -> PassforwardRatios:
    """Survival and absorption ratios of a clustering (Definition 6).

    These are ratios, divided by the number of clusters of the clustering.
    Table 3 of the paper is headed "passforward ratios" but reports absolute
    numbers of clusters, so the values returned here do not match it directly.

    >>> source = Clustering(0, {0: {1, 2}, 1: {3, 4}}, dict.fromkeys(range(1, 5), 1.0))
    >>> target = Clustering(1, {7: {1, 2}}, {1: 1.0, 2: 1.0})
    >>> passforward_ratios(detect_external_transitions(source, target))
    PassforwardRatios(survival=0.5, absorption=0.0)

    :param external: The external transitions from the clustering to the next.
    :return: The ratios of the earlier clustering, both 0 if it has no clusters.
    """
    kinds = Counter(
        transition.kind
        for transition in external
        if transition.kind is not ExternalKind.EMERGED
    )
    n_clusters = sum(kinds.values())
    if n_clusters == 0:
        return PassforwardRatios(0.0, 0.0)
    return PassforwardRatios(
        survival=kinds[ExternalKind.SURVIVED] / n_clusters,
        absorption=kinds[ExternalKind.ABSORBED] / n_clusters,
    )


@dataclass(frozen=True)
class TransitionReport:
    """Everything MONIC found between two successive clusterings."""

    source_timepoint: float | None
    """The timepoint of the earlier clustering, or ``None`` for the report on
    the first clustering, in which every cluster has emerged."""
    target_timepoint: float
    """The timepoint of the later clustering."""
    overlap: np.ndarray = field(compare=False)
    """The overlap of every cluster of the earlier clustering to every cluster
    of the later one. See :py:func:`build_overlap_matrix`."""
    external: tuple[ExternalTransition, ...]
    """See :py:func:`detect_external_transitions`."""
    internal: tuple[InternalTransition, ...]
    """See :py:func:`detect_internal_transitions`."""
    lifetimes: Mapping[int, Lifetime]
    """The lifetime of every cluster of the later clustering, by cluster id."""
    ratios: PassforwardRatios
    """The survival and absorption ratios of the earlier clustering."""
    overlap_weighting: str = "target"
    """Overlap weights every record by its weight at the later timepoint."""
    statistic_weighting: str = "own"
    """Internal transitions weight every cluster by its own timepoint."""


class MONIC:
    """Monitor the transitions of clusters over a sequence of clusterings.

    Give it one :py:class:`Clustering` per timepoint. Each is compared
    with the one before it.

    >>> monic = MONIC()
    >>> report = monic.update(Clustering(0, {0: {1, 2}, 1: {3, 4}}, dict.fromkeys(range(1, 5), 1.0)))
    >>> [transition.kind.name for transition in report.external]
    ['EMERGED', 'EMERGED']
    >>> report = monic.update(Clustering(1, {7: {1, 2, 5}}, {1: 1.0, 2: 1.0, 5: 1.0}))
    >>> [(t.kind.name, t.source, t.targets) for t in report.external]
    [('SURVIVED', 0, (7,)), ('DISAPPEARED', 1, ())]
    >>> report.lifetimes[7].internal
    2
    """

    overlap_weighting = "target"
    """Overlap weights every record by its weight at the later timepoint, as in
    Definition 3."""
    statistic_weighting = "own"
    """Internal transitions weight every cluster by its own timepoint, as in
    Table 2."""

    def __init__(
        self,
        tau: float = 0.5,
        tau_split: float = 0.1,
        epsilon: float = 0.1,
        delta: float = 0.1,
        tau_mean: float = 0.5,
        tau_skew: float = 0.5,
        debug: bool = False,
    ) -> None:
        """Construct a monitor.

        :param tau: See :py:func:`detect_external_transitions`.
        :param tau_split: See :py:func:`detect_external_transitions`.
        :param epsilon: See :py:func:`detect_internal_transitions`.
        :param delta: See :py:func:`detect_internal_transitions`.
        :param tau_mean: See :py:func:`detect_internal_transitions`.
        :param tau_skew: See :py:func:`detect_internal_transitions`.
        :param debug: Check every report against the invariants of MONIC and
            raise an ``AssertionError`` if one is violated: the rows of the
            overlap matrix sum to 1 where they should and never to more, every
            cluster of the earlier clustering has exactly one external
            transition, the survived, absorbed, split and disappeared clusters
            add up to that clustering, and the lifetimes are ordered.
        :raises ValueError: Unless ``0 < tau_split < tau <= 1`` and the other
            thresholds are non-negative.
        """
        _check_external_thresholds(tau, tau_split)
        _check_internal_thresholds(epsilon, delta, tau_mean, tau_skew)
        self.debug = debug
        self.tau = tau
        self.tau_split = tau_split
        self.epsilon = epsilon
        self.delta = delta
        self.tau_mean = tau_mean
        self.tau_skew = tau_skew
        self._clustering: Clustering | None = None
        self._lifetimes: dict[int, Lifetime] = {}

    @property
    def clustering(self) -> Clustering | None:
        """The latest clustering, or ``None`` before the first update."""
        return self._clustering

    @property
    def lifetimes(self) -> dict[int, Lifetime]:
        """The lifetime of every cluster of the latest clustering."""
        return dict(self._lifetimes)

    def update(self, clustering: Clustering) -> TransitionReport:
        """Compare a new clustering with the previous one.

        :param clustering: The clustering at the next timepoint.
        :raises ValueError: If its timepoint is not later than the previous one.
        :return: The transitions from the previous clustering to this one. For
            the first clustering, every cluster has emerged.
        """
        source = self._clustering
        if source is None:
            source_timepoint = None
            source = Clustering(clustering.timepoint, {}, {})
        elif clustering.timepoint <= source.timepoint:
            raise ValueError(
                f"Timepoint {clustering.timepoint} is not later than the previous "
                f"timepoint {source.timepoint}."
            )
        else:
            source_timepoint = source.timepoint

        external = _external_transitions(source, clustering, self.tau, self.tau_split)
        internal = detect_internal_transitions(
            source,
            clustering,
            external,
            self.epsilon,
            self.delta,
            self.tau_mean,
            self.tau_skew,
        )
        self._lifetimes = advance_lifetimes(
            self._lifetimes, source, clustering, external, internal
        )
        self._clustering = clustering
        report = TransitionReport(
            source_timepoint=source_timepoint,
            target_timepoint=clustering.timepoint,
            overlap=build_overlap_matrix(source, clustering),
            external=external,
            internal=internal,
            lifetimes=dict(self._lifetimes),
            ratios=passforward_ratios(external),
        )
        if self.debug:
            _check_invariants(source, clustering, report)
        return report


def _check_invariants(
    source: Clustering, target: Clustering, report: TransitionReport
) -> None:
    """Raise an ``AssertionError`` if a report violates an invariant of MONIC.

    :param source: The earlier clustering of the report.
    :param target: The later clustering of the report.
    :param report: The report to check.
    """
    tolerance = 1e-9
    timepoint = report.target_timepoint

    # Overlap matrix row sums.
    assigned = frozenset().union(*target.clusters.values())
    for (cluster_id, records), row_sum in zip(
        source.clusters.items(), report.overlap.sum(axis=1).tolist()
    ):
        alive = {record for record in records if target.weights.get(record, 0.0) > 0}
        if not alive:
            expected, holds = "0", row_sum == 0.0
        elif alive <= assigned:
            expected, holds = "1", abs(row_sum - 1.0) <= tolerance
        else:
            expected, holds = "at most 1", row_sum <= 1.0 + tolerance
        if not holds:
            raise AssertionError(
                f"Timepoint {timepoint}: the overlaps of cluster {cluster_id} sum "
                f"to {row_sum}, expected {expected}."
            )

    # Every cluster of the earlier clustering has exactly one external label.
    labelled = [t.source for t in report.external if t.source is not None]
    if labelled != list(source.clusters):
        raise AssertionError(
            f"Timepoint {timepoint}: external transitions are for clusters "
            f"{labelled}, expected one each for {list(source.clusters)}."
        )

    # The four outcomes account for the whole of the earlier clustering.
    kinds = Counter(t.kind for t in report.external)
    n_accounted = (
        kinds[ExternalKind.SURVIVED]
        + kinds[ExternalKind.ABSORBED]
        + kinds[ExternalKind.SPLIT]
        + kinds[ExternalKind.DISAPPEARED]
    )
    if n_accounted != len(source.clusters):
        raise AssertionError(
            f"Timepoint {timepoint}: {n_accounted} clusters survived, were "
            f"absorbed, were split or disappeared, out of {len(source.clusters)}."
        )

    # Lifetime ordering.
    if report.lifetimes.keys() != target.clusters.keys():
        raise AssertionError(
            f"Timepoint {timepoint}: lifetimes are for clusters "
            f"{list(report.lifetimes)}, expected {list(target.clusters)}."
        )
    for cluster_id, lifetime in report.lifetimes.items():
        if not 1 <= lifetime.strict <= lifetime.internal <= lifetime.absorption:
            raise AssertionError(
                f"Timepoint {timepoint}: cluster {cluster_id} has {lifetime}, "
                "expected 1 <= strict <= internal <= absorption."
            )


class MonitorableClusterer(ABC):
    """A clusterer that can say which records are in which of its clusters.

    MONIC compares two clusterings record by record (Definition 3), so it needs
    the members of every cluster, not only its centre. A clusterer that
    implements this interface can be monitored by :py:class:`MONICMonitor`:

    .. code-block:: python

        class MyClusterer(Clusterer, MonitorableClusterer):
            ...

            def get_clustering(self) -> Clustering:
                return Clustering(
                    timepoint=self.instances_seen,
                    clusters=...,  # cluster id -> ids of its records
                    weights=...,  # record id -> ageing weight in [0, 1]
                    points=...,  # record id -> feature vector
                )
    """

    @abstractmethod
    def get_clustering(self) -> Clustering:
        """The clustering of the current state of the model.

        For the transitions between two calls to be meaningful:

        * A record keeps its id from one call to the next.
        * The weights cover every record the model still holds, including
          those in no cluster, and are their ageing weights now (Definition 2).
          A record that is left out, or has weight 0, has aged out, even if it
          is in a cluster.
        * The timepoint is later than that of the previous call.

        The points are optional. Without them the internal analysis of every
        survived cluster is skipped: no internal transitions are reported, and
        the strict lifetime of every cluster stays at 1.

        :return: The clusters of the model now, with their records.
        """


class MONICMonitor:
    """Monitor the cluster transitions of a clusterer on a stream.

    Every ``check_interval`` instances is a timepoint. At each timepoint the
    monitor asks the clusterer for its current clustering (see
    :py:class:`MonitorableClusterer`) and compares it with the clustering of the
    previous timepoint (see :py:class:`MONIC`).

    A transition describes how the clusterer's model changed between two
    timepoints. It reflects a change in the data only once the model has
    reacted to it.
    """

    def __init__(
        self,
        clusterer: Clusterer,
        check_interval: int = 1000,
        tau: float = 0.5,
        tau_split: float = 0.1,
        epsilon: float = 0.1,
        delta: float = 0.1,
        tau_mean: float = 0.5,
        tau_skew: float = 0.5,
        debug: bool = False,
    ) -> None:
        """Construct a monitor.

        :param clusterer: The clusterer to train and monitor. It must implement
            :py:class:`MonitorableClusterer`.
        :param check_interval: The number of instances between timepoints.
        :param tau: See :py:func:`detect_external_transitions`.
        :param tau_split: See :py:func:`detect_external_transitions`.
        :param epsilon: See :py:func:`detect_internal_transitions`.
        :param delta: See :py:func:`detect_internal_transitions`.
        :param tau_mean: See :py:func:`detect_internal_transitions`.
        :param tau_skew: See :py:func:`detect_internal_transitions`.
        :param debug: Check the invariants of MONIC at every timepoint. See
            :py:class:`MONIC`.
        :raises TypeError: If the clusterer does not implement
            :py:class:`MonitorableClusterer`.
        :raises ValueError: If ``check_interval`` is less than 1, or a threshold
            is out of range. See :py:class:`MONIC`.
        """
        if not isinstance(clusterer, MonitorableClusterer):
            raise TypeError(f"{clusterer} does not implement MonitorableClusterer.")
        if check_interval < 1:
            raise ValueError("The check interval must be at least 1.")
        self.clusterer = clusterer
        self.check_interval = check_interval
        self.monic = MONIC(
            tau, tau_split, epsilon, delta, tau_mean, tau_skew, debug=debug
        )
        self.instances_seen = 0

    @property
    def clustering(self) -> Clustering | None:
        """The latest clustering, or ``None`` before the first timepoint."""
        return self.monic.clustering

    def train(self, instance: Instance) -> TransitionReport | None:
        """Train the clusterer on an instance.

        :param instance: The next instance of the stream.
        :return: A report if the instance completes a timepoint, else ``None``.
        """
        self.clusterer.train(instance)
        self.instances_seen += 1
        if self.instances_seen % self.check_interval != 0:
            return None
        return self.monic.update(self.clusterer.get_clustering())

    def monitor(
        self, stream: Stream, max_instances: int | None = None
    ) -> Iterator[TransitionReport]:
        """Train on a stream, yielding a report at every timepoint.

        :param stream: The stream to consume.
        :param max_instances: Stop after this many instances, or ``None`` to
            consume the whole stream.
        """
        n_instances = 0
        while stream.has_more_instances() and (
            max_instances is None or n_instances < max_instances
        ):
            report = self.train(stream.next_instance())
            n_instances += 1
            if report is not None:
                yield report
