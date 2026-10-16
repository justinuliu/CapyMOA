"""MONIC: modelling and monitoring cluster transitions.

MONIC [#f0]_ observes clusterings produced at successive timepoints and
classifies what happened to each cluster: whether it survived, was absorbed,
was split or disappeared, which clusters emerged, and how the survivors changed.

:py:class:`MONIC` takes one :py:class:`Clustering` per timepoint. The
clusterings can come from clustering the data of each timepoint yourself, as
the paper does, or from a clusterer on a stream that implements
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
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np

__all__ = [
    "MONIC",
    "Clustering",
    "ExternalTransition",
    "InternalTransition",
    "Lifetime",
    "MonitorableClusterer",
    "PassforwardRatios",
    "TransitionReport",
    "TransitionType",
    "build_overlap_matrix",
    "overlap",
]


@dataclass(frozen=True)
class Clustering:
    """A clustering at one timepoint, with instance-level membership.

    The row and column order of an overlap matrix follows the iteration order of
    :py:attr:`clusters`.

    >>> clustering = Clustering(
    ...     timepoint=0,
    ...     clusters={0: {1, 2}, 1: {3}},
    ...     weights={1: 1.0, 2: 1.0, 3: 0.5},
    ... )
    >>> clustering.cluster_ids
    (0, 1)

    :raises ValueError: If an instance belongs to more than one cluster, a
        clustered instance has no weight or no point, a weight is outside
        ``[0, 1]``, or the points do not all have the same number of features.
    """

    timepoint: float
    """The timepoint at which the clustering was built."""
    clusters: Mapping[int, AbstractSet[int]]
    """Maps a cluster id to the ids of its instances. Cluster ids are only
    meaningful within this clustering. Clusters must be disjoint."""
    weights: Mapping[int, float]
    """Maps an instance id to its ageing weight ``age(instance, timepoint)`` in
    ``[0, 1]``. It covers every instance alive at this timepoint, including
    instances that belong to no cluster. An instance that is absent has weight 0.
    An instance of weight 0 has aged out even if it is in a cluster: it adds
    nothing to any overlap and has no weight in the statistics of its cluster."""
    points: Mapping[int, Sequence[float]] | None = field(default=None, compare=False)
    """Maps an instance id to its feature vector at this timepoint. Only internal
    transitions need it. If either of two clusterings has no points, the
    internal analysis of every survived cluster between them is skipped, so its
    strict lifetime restarts at 1 (see :py:class:`Lifetime`)."""

    def __post_init__(self) -> None:
        clusters = {
            cluster_id: frozenset(instances)
            for cluster_id, instances in self.clusters.items()
        }
        weights = dict(self.weights)

        n_memberships = sum(len(instances) for instances in clusters.values())
        clustered: frozenset[int] = frozenset().union(*clusters.values())
        if len(clustered) != n_memberships:
            raise ValueError("Clusters must be disjoint.")
        if not clustered <= weights.keys():
            raise ValueError("Every clustered instance must have a weight.")
        if any(not 0.0 <= weight <= 1.0 for weight in weights.values()):
            raise ValueError("Weights must be in [0, 1].")

        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "weights", weights)

        if self.points is None:
            return
        points = {
            instance: np.asarray(point, dtype=float)
            for instance, point in self.points.items()
        }
        if not clustered <= points.keys():
            raise ValueError("Every clustered instance must have a point.")
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
        """Turn the instances of some clusters into noise.

        Use it on a garbage cluster: one into which a clusterer puts the instances
        it cannot place. Such a cluster is large, so left in place it becomes
        the match of clusters that merely scattered. Its instances stay alive but
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
            cluster_id: instances
            for cluster_id, instances in self.clusters.items()
            if cluster_id not in dropped
        }
        return Clustering(self.timepoint, clusters, self.weights, self.points)


def overlap(
    x: AbstractSet[int], y: AbstractSet[int], weights: Mapping[int, float]
) -> float:
    """Overlap of cluster ``x`` to cluster ``y`` (Definition 3).

    The normalised sum of the weights of the instances in the intersection:
    ``sum(age(a, t_j) for a in x & y) / sum(age(a, t_j) for a in x)``. Both sums
    use the weights of the later timepoint ``t_j``. Overlap is not symmetric.

    >>> overlap({1, 2, 3, 4}, {3, 4, 5}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    0.5

    :param x: Instance ids of a cluster found at the earlier timepoint ``t_i``.
    :param y: Instance ids of a cluster found at the later timepoint ``t_j``.
    :param weights: Ageing weights at ``t_j``. An absent instance has weight 0.
    :return: The overlap in ``[0, 1]``, or 0 if every instance of ``x`` has aged
        out by ``t_j``.
    """
    denominator = sum(weights.get(instance, 0.0) for instance in x)
    if denominator == 0.0:
        return 0.0
    return sum(weights.get(instance, 0.0) for instance in x & y) / denominator


def build_overlap_matrix(source: Clustering, target: Clustering) -> np.ndarray:
    """Overlap of every cluster of ``source`` to every cluster of ``target``.

    Uses an inverted index from instance id to the cluster of ``target``, then a
    single pass over the instances of each cluster of ``source``. Besides
    allocating the matrix, the cost is linear in the number of instances rather
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
        every instance of its cluster has aged out.
    """
    return _overlap_weights(source, target).forward


class _OverlapWeights(NamedTuple):
    """Numerators and denominators of Definition 3 for every pair of clusters of
    two clusterings, using the weights of the later one."""

    intersections: np.ndarray
    """Entry ``[i, j]`` is the weight of the instances that the ``i``-th cluster
    of the earlier clustering shares with the ``j``-th cluster of the later one."""
    source_totals: np.ndarray
    """Entry ``i`` is the weight of the ``i``-th cluster of the earlier clustering."""
    target_totals: np.ndarray
    """Entry ``j`` is the weight of the ``j``-th cluster of the later clustering."""

    @property
    def forward(self) -> np.ndarray:
        """The overlap matrix: entry ``[i, j]`` is the overlap of the ``i``-th
        cluster of the earlier clustering to the ``j``-th cluster of the later one."""
        return _divide_or_zero(self.intersections, self.source_totals[:, np.newaxis])

    @property
    def reverse(self) -> np.ndarray:
        """Entry ``[i, j]`` is the overlap of the ``j``-th cluster of the later
        clustering to the ``i``-th cluster of the earlier one."""
        return _divide_or_zero(self.intersections, self.target_totals[np.newaxis, :])


def _overlap_weights(source: Clustering, target: Clustering) -> _OverlapWeights:
    """Numerators and denominators of Definition 3, using the weights of ``target``.

    See :py:func:`build_overlap_matrix` for how they are computed.
    """
    column_of: dict[int, int] = {}
    target_totals = np.zeros(len(target.clusters))
    for column, instances in enumerate(target.clusters.values()):
        for instance in instances:
            column_of[instance] = column
            target_totals[column] += target.weights[instance]

    intersections = np.zeros((len(source.clusters), len(target.clusters)))
    source_totals = np.zeros(len(source.clusters))
    for row, instances in enumerate(source.clusters.values()):
        for instance in instances:
            weight = target.weights.get(instance, 0.0)
            source_totals[row] += weight
            column = column_of.get(instance)
            if column is not None:
                intersections[row, column] += weight
    return _OverlapWeights(intersections, source_totals, target_totals)


def _divide_or_zero(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, giving 0 wherever the denominator is 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


class TransitionType(Enum):
    """The transitions of a cluster.

    The external transitions (Table 1) say what became of a cluster: it
    survived, was absorbed, was split or disappeared. A new cluster emerged.

    The internal transitions (Table 2) say how a survived cluster changed. There
    are three groups: size, compactness and location. A cluster undergoes at
    most one transition of each group.
    """

    SURVIVED = "survived"
    """External: the cluster has a match that is the match of no other cluster."""
    ABSORBED = "absorbed"
    """External: the cluster has a match that is also the match of another
    cluster."""
    SPLIT = "split"
    """External: the cluster has no match, but several clusters together form
    one."""
    DISAPPEARED = "disappeared"
    """External: none of the other cases hold for the cluster."""
    EMERGED = "emerged"
    """External: a new cluster that is not the result of another external
    transition."""
    SHRUNK = "shrunk"
    """Internal, size: the cluster lost weight."""
    EXPANDED = "expanded"
    """Internal, size: the cluster gained weight."""
    COMPACTED = "compacted"
    """Internal, compactness: the cluster became more compact."""
    DIFFUSED = "diffused"
    """Internal, compactness: the cluster became more diffuse."""
    SHIFTED = "shifted"
    """Internal, location: the mean or the skewness of the cluster moved."""


@dataclass(frozen=True)
class ExternalTransition:
    """An external transition between two clusterings."""

    kind: TransitionType
    """One of the external transitions: survived, absorbed, split, disappeared
    or emerged."""
    source: int | None
    """The id of the cluster of the earlier clustering that underwent the
    transition, or ``None`` for an emerged cluster."""
    targets: tuple[int, ...]
    """The ids of the resulting clusters of the later clustering: one for a
    survived, absorbed or emerged cluster, several for a split and none for a
    disappeared cluster."""


@dataclass(frozen=True)
class InternalTransition:
    """The internal transitions of one survived cluster (Table 2).

    A cluster is described by its weight, its mean, its spread ``sigma`` (the
    root mean square distance of its instances to the mean) and its skewness in
    each feature. Each cluster is weighted by its own clustering. Unlike in the
    paper, the thresholds on weight, spread and mean are relative, so that they
    do not depend on the size of the stream or the scale of the features. The
    thresholds are set on :py:class:`MONIC`.

    The statistics are undefined across a change of feature space. If the two
    clusterings do not both carry points with the same number of features, every
    survived cluster is reported as skipped instead.

    >>> monic = MONIC()
    >>> _ = monic.update(Clustering(0, {0: {1, 2}}, {1: 1.0, 2: 1.0}, {1: [0.0], 2: [2.0]}))
    >>> report = monic.update(
    ...     Clustering(
    ...         1, {7: {1, 2, 3}}, dict.fromkeys([1, 2, 3], 1.0), {1: [0.0], 2: [2.0], 3: [1.0]}
    ...     )
    ... )
    >>> (internal,) = report.internal
    >>> sorted(kind.name for kind in internal.kinds)
    ['COMPACTED', 'EXPANDED']
    """

    source: int
    """The id of the cluster in the earlier clustering."""
    target: int
    """The id of the cluster in the later clustering."""
    kinds: frozenset[TransitionType]
    """The internal transitions the cluster underwent: at most one of shrunk
    and expanded, one of compacted and diffused, and shifted. Empty if it did
    not change, and also if the analysis was :py:attr:`skipped`."""
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

    :param clustering: A clustering that carries points.
    :return: The weight of the cluster, its mean, the root mean square distance
        of its instances to the mean and its skewness in each feature, or ``None``
        if the cluster has no weight.
    """
    vectors = clustering.points
    assert vectors is not None, "Statistics need a clustering with points."
    instances = sorted(clustering.clusters[cluster_id])
    weights = np.array([clustering.weights[instance] for instance in instances])
    total = weights.sum()
    if total == 0.0:
        return None

    points = np.array([vectors[instance] for instance in instances])
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


@dataclass(frozen=True)
class Lifetime:
    """The lifetimes of a cluster (Definition 5).

    Each is a number of timepoints counted forward from the emergence of the
    cluster, up to and including the timepoint of the clustering it belongs to.
    A new cluster has a lifetime of 1. ``strict <= internal <= absorption``
    always holds.

    The paper computes lifetimes backward from the last clustering. A stream
    has no last clustering, so they are counted forward instead:

    * A cluster that survives passes all three lifetimes on, each extended by
      one. The strict lifetime restarts at 1 if the cluster underwent an
      internal transition, or if it was not shown to be free of one because
      its internal analysis was skipped.
    * A cluster that absorbed others continues the absorption lifetime of the
      one sharing the most weight with it, extended by one. Ties go to the
      longer absorption lifetime, then to the order of the earlier clustering.
      Its other lifetimes restart at 1.
    * Every other cluster is new: it emerged, or it is the result of a split.
      A split extends no lifetime.
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


@dataclass(frozen=True)
class PassforwardRatios:
    """The survival and absorption ratios of a clustering (Definition 6).

    These are ratios, divided by the number of clusters of the clustering, and
    both are 0 for a clustering with no clusters. Table 3 of the paper is headed
    "passforward ratios" but reports absolute numbers of clusters, so these
    values do not match it directly.
    """

    survival: float
    """The portion of the clusters that survived into the next clustering."""
    absorption: float
    """The portion of the clusters absorbed by a cluster of the next clustering."""

    @property
    def passforward(self) -> float:
        """The sum of the survival and absorption ratios."""
        return self.survival + self.absorption


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
    """The external transitions (Table 1): one for each cluster of the earlier
    clustering, in order, followed by one for each emerged cluster of the later
    clustering, in order."""
    internal: tuple[InternalTransition, ...]
    """The internal transitions (Table 2): one for each survived cluster of the
    earlier clustering, in order."""
    lifetimes: Mapping[int, Lifetime]
    """The lifetime of every cluster of the later clustering, by cluster id."""
    ratios: PassforwardRatios
    """The survival and absorption ratios of the earlier clustering."""
    overlap_weighting: str = "target"
    """Overlap weights every instance by its weight at the later timepoint."""
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
    >>> report.ratios
    PassforwardRatios(survival=0.5, absorption=0.0)
    """

    overlap_weighting = "target"
    """Overlap weights every instance by its weight at the later timepoint, as in
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

        :param tau: The least overlap for one cluster to be the match of another
            (Definition 4). A cluster with a match survives, or is absorbed if
            another cluster has the same match. Definition 4 restricts ``tau``
            to ``[0.5, 1]``; a smaller value is accepted with a warning.
        :param tau_split: The least overlap for a cluster to take part in a
            split. A cluster without a match is split if the clusters it
            overlaps by at least ``tau_split`` together overlap it by at least
            ``tau``; otherwise it disappears.
        :param epsilon: A survived cluster shrinks or expands if its weight
            changes by more than this fraction.
        :param delta: A survived cluster becomes more compact or more diffuse if
            its spread changes by more than this fraction.
        :param tau_mean: A survived cluster shifts if its mean moves by more than
            this fraction of its spread.
        :param tau_skew: A survived cluster shifts if its skewness changes by
            more than this in any feature.
        :param debug: Check every report against the invariants of MONIC and
            raise an ``AssertionError`` if one is violated: the rows of the
            overlap matrix sum to 1 where they should and never to more, every
            cluster of the earlier clustering has exactly one external
            transition, the survived, absorbed, split and disappeared clusters
            add up to that clustering, and the lifetimes are ordered.
        :raises ValueError: Unless ``0 < tau_split < tau <= 1`` and the other
            thresholds are non-negative.
        """
        if not 0.0 < tau_split < tau <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < tau_split < tau <= 1, but got "
                f"tau_split={tau_split} and tau={tau}."
            )
        if tau < 0.5:
            warnings.warn(
                f"tau={tau} is below 0.5, so a cluster may be matched by less "
                "than half of its weight.",
                stacklevel=2,
            )
        if min(epsilon, delta, tau_mean, tau_skew) < 0.0:
            raise ValueError("Thresholds must be non-negative.")
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
        previous = self._clustering
        if previous is not None and clustering.timepoint <= previous.timepoint:
            raise ValueError(
                f"Timepoint {clustering.timepoint} is not later than the previous "
                f"timepoint {previous.timepoint}."
            )
        # The first clustering is compared with an empty one.
        if previous is None:
            source = Clustering(clustering.timepoint, {}, {})
        else:
            source = previous

        overlap_weights = _overlap_weights(source, clustering)
        external = self._detect_external_transitions(
            source, clustering, overlap_weights
        )
        internal = self._detect_internal_transitions(source, clustering, external)
        self._lifetimes = self._advance_lifetimes(
            self._lifetimes, source, clustering, overlap_weights, external, internal
        )
        self._clustering = clustering
        report = TransitionReport(
            source_timepoint=None if previous is None else previous.timepoint,
            target_timepoint=clustering.timepoint,
            overlap=overlap_weights.forward,
            external=external,
            internal=internal,
            lifetimes=dict(self._lifetimes),
            ratios=self._passforward_ratios(external),
        )
        if self.debug:
            _check_invariants(source, clustering, report)
        return report

    def _detect_external_transitions(
        self,
        source: Clustering,
        target: Clustering,
        overlap_weights: _OverlapWeights,
    ) -> tuple[ExternalTransition, ...]:
        """Detect the external transitions from ``source`` to ``target`` (Table 1).

        A cluster of ``source`` with a match in ``target`` survives, or is
        absorbed if another cluster has the same match. The match is the cluster
        with the largest overlap, provided that overlap is at least
        :py:attr:`tau`. Ties are broken by the larger reverse overlap, then by
        the order of the clusters in ``target``. A cluster without a match is
        split if the clusters it overlaps by at least :py:attr:`tau_split`
        together overlap it by at least :py:attr:`tau`; otherwise it disappears.

        :param source: The clustering at the earlier timepoint ``t_i``.
        :param target: The clustering at the later timepoint ``t_j``.
        :param overlap_weights: The overlap weights of ``source`` to ``target``.
        :return: One transition for each cluster of ``source``, in order,
            followed by one for each emerged cluster of ``target``, in order.
        """
        forward = overlap_weights.forward
        reverse = overlap_weights.reverse

        # Pass 1: find the match of each cluster of ``source``, or failing that
        # the clusters it is split into. A cluster with a match is never split.
        match_of: dict[int, int] = {}
        split_of: dict[int, list[int]] = {}
        for row in range(len(source.clusters)):
            candidates = np.flatnonzero(forward[row] >= self.tau).tolist()
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
            candidates = np.flatnonzero(forward[row] >= self.tau_split).tolist()
            # Clusters are disjoint, so the overlap to their union is the sum of
            # their intersection weights over the weight of the cluster.
            union_weight = overlap_weights.intersections[row, candidates].sum()
            total_weight = overlap_weights.source_totals[row]
            if candidates and union_weight / total_weight >= self.tau:
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
                kind = TransitionType.ABSORBED if absorbed else TransitionType.SURVIVED
                targets: tuple[int, ...] = (target_ids[column],)
            elif row in split_of:
                kind = TransitionType.SPLIT
                targets = tuple(target_ids[column] for column in split_of[row])
            else:
                kind = TransitionType.DISAPPEARED
                targets = ()
            produced.update(targets)
            transitions.append(ExternalTransition(kind, source_id, targets))

        transitions.extend(
            ExternalTransition(TransitionType.EMERGED, None, (target_id,))
            for target_id in target_ids
            if target_id not in produced
        )
        return tuple(transitions)

    def _detect_internal_transitions(
        self,
        source: Clustering,
        target: Clustering,
        external: Iterable[ExternalTransition],
    ) -> tuple[InternalTransition, ...]:
        """Detect the internal transitions of the survived clusters (Table 2).

        See :py:class:`InternalTransition` for the statistics compared.

        :param source: The clustering at the earlier timepoint ``t_i``.
        :param target: The clustering at the later timepoint ``t_j``.
        :param external: The external transitions from ``source`` to ``target``.
            Only the survived clusters are analysed.
        :return: One internal transition for each survived cluster, in order.
        """
        comparable = (
            source.n_features is not None and source.n_features == target.n_features
        )

        result = []
        for transition in external:
            if transition.kind is not TransitionType.SURVIVED:
                continue
            source_id = transition.source
            assert source_id is not None, "Only an emerged cluster has no source."
            (target_id,) = transition.targets
            before = _statistics(source, source_id) if comparable else None
            after = _statistics(target, target_id) if comparable else None
            if before is None or after is None:
                result.append(
                    InternalTransition(source_id, target_id, frozenset(), skipped=True)
                )
            else:
                kinds = self._internal_kinds(before, after)
                result.append(InternalTransition(source_id, target_id, kinds))
        return tuple(result)

    def _internal_kinds(
        self, before: _Statistics, after: _Statistics
    ) -> frozenset[TransitionType]:
        """The internal transitions of a cluster, given its statistics at ``t_i``
        and ``t_j``."""
        kinds = set()
        if after.weight < (1 - self.epsilon) * before.weight:
            kinds.add(TransitionType.SHRUNK)
        elif after.weight > (1 + self.epsilon) * before.weight:
            kinds.add(TransitionType.EXPANDED)

        if after.spread < (1 - self.delta) * before.spread:
            kinds.add(TransitionType.COMPACTED)
        elif after.spread > (1 + self.delta) * before.spread:
            kinds.add(TransitionType.DIFFUSED)

        mean_shift = np.linalg.norm(after.mean - before.mean)
        skew_shift = np.abs(after.skewness - before.skewness).max()
        if mean_shift > self.tau_mean * before.spread or skew_shift > self.tau_skew:
            kinds.add(TransitionType.SHIFTED)
        return frozenset(kinds)

    @staticmethod
    def _advance_lifetimes(
        lifetimes: Mapping[int, Lifetime],
        source: Clustering,
        target: Clustering,
        overlap_weights: _OverlapWeights,
        external: Iterable[ExternalTransition],
        internal: Iterable[InternalTransition],
    ) -> dict[int, Lifetime]:
        """Lifetimes of the clusters of ``target``, given those of ``source``.

        See :py:class:`Lifetime` for the rules. A survived cluster whose
        internal analysis is missing from ``internal`` is treated like one whose
        analysis was skipped.

        :param lifetimes: The lifetime of every cluster of ``source``, by
            cluster id.
        :param source: The clustering at the earlier timepoint ``t_i``.
        :param target: The clustering at the later timepoint ``t_j``.
        :param overlap_weights: The overlap weights of ``source`` to ``target``.
        :param external: The external transitions from ``source`` to ``target``.
        :param internal: The internal transitions from ``source`` to ``target``.
        :return: The lifetime of every cluster of ``target``, by cluster id.
        """
        unchanged = {
            transition.source
            for transition in internal
            if not transition.skipped and not transition.kinds
        }
        intersections = overlap_weights.intersections
        row_of = {cluster_id: row for row, cluster_id in enumerate(source.clusters)}
        column_of = {
            cluster_id: column for column, cluster_id in enumerate(target.clusters)
        }

        continued: dict[int, Lifetime] = {}
        absorbed_by: dict[int, list[int]] = {}
        for transition in external:
            source_id = transition.source
            if source_id is None:
                # An emerged cluster continues no lifetime.
                continue
            if transition.kind is TransitionType.SURVIVED:
                before = lifetimes[source_id]
                continued[transition.targets[0]] = Lifetime(
                    strict=before.strict + 1 if source_id in unchanged else 1,
                    internal=before.internal + 1,
                    absorption=before.absorption + 1,
                )
            elif transition.kind is TransitionType.ABSORBED:
                absorbed_by.setdefault(transition.targets[0], []).append(source_id)

        for target_id, source_ids in absorbed_by.items():
            dominant = max(
                source_ids,
                key=lambda source_id: (
                    intersections[row_of[source_id], column_of[target_id]],
                    lifetimes[source_id].absorption,
                    -row_of[source_id],
                ),
            )
            continued[target_id] = Lifetime(
                absorption=lifetimes[dominant].absorption + 1
            )

        return {
            target_id: continued.get(target_id, Lifetime())
            for target_id in target.clusters
        }

    @staticmethod
    def _passforward_ratios(
        external: Iterable[ExternalTransition],
    ) -> PassforwardRatios:
        """Survival and absorption ratios of a clustering (Definition 6).

        :param external: The external transitions from the clustering to the
            next.
        :return: The ratios of the earlier clustering.
        """
        kinds = Counter(
            transition.kind
            for transition in external
            if transition.kind is not TransitionType.EMERGED
        )
        n_clusters = sum(kinds.values())
        if n_clusters == 0:
            return PassforwardRatios(0.0, 0.0)
        return PassforwardRatios(
            survival=kinds[TransitionType.SURVIVED] / n_clusters,
            absorption=kinds[TransitionType.ABSORBED] / n_clusters,
        )


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
    for (cluster_id, instances), row_sum in zip(
        source.clusters.items(), report.overlap.sum(axis=1).tolist()
    ):
        alive = {
            instance for instance in instances if target.weights.get(instance, 0.0) > 0
        }
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
        kinds[TransitionType.SURVIVED]
        + kinds[TransitionType.ABSORBED]
        + kinds[TransitionType.SPLIT]
        + kinds[TransitionType.DISAPPEARED]
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
    """A clusterer that can say which instances are in which of its clusters.

    MONIC compares two clusterings instance by instance (Definition 3), so it needs
    the members of every cluster, not only its centre. A clusterer that
    implements this interface can be monitored with :py:class:`MONIC`:

    .. code-block:: python

        class MyClusterer(Clusterer, MonitorableClusterer):
            ...

            def get_clustering(self) -> Clustering:
                return Clustering(
                    timepoint=self.instances_seen,
                    clusters=...,  # cluster id -> ids of its instances
                    weights=...,  # instance id -> ageing weight in [0, 1]
                    points=...,  # instance id -> feature vector
                )
    """

    @abstractmethod
    def get_clustering(self) -> Clustering:
        """The clustering of the current state of the model.

        For the transitions between two calls to be meaningful:

        * An instance keeps its id from one call to the next.
        * The weights cover every instance the model still holds, including
          those in no cluster, and are their ageing weights now (Definition 2).
          An instance that is left out, or has weight 0, has aged out, even if it
          is in a cluster.
        * The timepoint is later than that of the previous call.

        The points are optional. Without them the internal analysis of every
        survived cluster is skipped: no internal transitions are reported, and
        the strict lifetime of every cluster stays at 1.

        :return: The clusters of the model now, with their instances.
        """
