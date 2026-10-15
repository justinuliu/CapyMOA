import dataclasses

import numpy as np
import pytest

from capymoa.base import Clusterer, ClusteringResult
from capymoa.core import Instance
from capymoa.evaluation.monic import (
    MONIC,
    Clustering,
    ExternalTransition,
    InternalTransition,
    Lifetime,
    MONICMonitor,
    MonitorableClusterer,
    PassforwardRatios,
    TransitionType,
    _check_invariants,
    advance_lifetimes,
    build_overlap_matrix,
    detect_external_transitions,
    detect_internal_transitions,
    overlap,
    passforward_ratios,
)
from capymoa.stream import NumpyStream

TOLERANCE = 1e-9


def _clustering(
    timepoint: float,
    clusters: dict[int, set[int]],
    weights: dict[int, float] | None = None,
) -> Clustering:
    """Build a clustering, giving every clustered instance weight 1 by default."""
    if weights is None:
        weights = {
            instance: 1.0 for instances in clusters.values() for instance in instances
        }
    return Clustering(timepoint, clusters, weights)


# Five clusters at t_i over instances 1..19.
SOURCE_CLUSTERS = {
    0: {1, 2, 3, 4},
    1: {5, 6, 7, 8},
    2: {9, 10, 11, 12, 13},
    3: {14, 15},
    4: {16, 17, 18, 19},
}
# Four clusters at t_j over the same instances plus the new instances 20 and 21.
TARGET_CLUSTERS = {
    0: {1, 2, 3, 4, 16, 20, 21},
    1: {5, 6, 7, 8, 14, 15},
    2: {9, 10, 11},
    3: {12, 13, 17, 18, 19},
}


def test_overlap_matrix_known_values():
    matrix = build_overlap_matrix(
        _clustering(0, SOURCE_CLUSTERS), _clustering(1, TARGET_CLUSTERS)
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.4],
            [0.0, 1.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.75],
        ]
    )
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=TOLERANCE)


def test_overlap_matrix_matches_pairwise_overlap():
    source = _clustering(0, SOURCE_CLUSTERS)
    target = _clustering(1, TARGET_CLUSTERS)
    matrix = build_overlap_matrix(source, target)

    for row, x in enumerate(source.clusters.values()):
        for column, y in enumerate(target.clusters.values()):
            assert matrix[row, column] == pytest.approx(
                overlap(x, y, target.weights), abs=TOLERANCE
            )


@pytest.mark.parametrize(
    "old_weight", [1.0, 0.5, 0.125], ids=["uniform", "half", "eighth"]
)
def test_rows_sum_to_one_when_all_instances_survive(old_weight: float):
    # Every t_i instance keeps a positive weight at t_j and is assigned to a cluster.
    target_weights = {instance: old_weight for instance in range(1, 20)}
    target_weights |= {20: 1.0, 21: 1.0}
    # Unequal weights inside a cluster, so the weights do not simply cancel out.
    target_weights |= {9: 1.0, 16: 1.0}

    matrix = build_overlap_matrix(
        _clustering(0, SOURCE_CLUSTERS), _clustering(1, TARGET_CLUSTERS, target_weights)
    )

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=TOLERANCE)


def test_row_sums_below_one_when_surviving_instance_is_unassigned():
    # Instance 4 is still alive at t_j but belongs to no cluster.
    source = _clustering(0, {0: {1, 2, 3, 4}})
    target = _clustering(1, {0: {1, 2, 3}}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

    assert build_overlap_matrix(source, target).tolist() == [[0.75]]


@pytest.mark.parametrize(
    "aged_out_weights",
    [{}, {1: 0.0, 2: 0.0, 3: 0.0}],
    ids=["absent", "zero-weight"],
)
def test_fully_aged_out_cluster_has_zero_overlap(aged_out_weights: dict[int, float]):
    # Cluster 0 of t_i has aged out completely by t_j; cluster 1 has not.
    source = _clustering(0, {0: {1, 2, 3}, 1: {4, 5}})
    target = _clustering(1, {0: {4, 5, 6}, 1: {7}}, {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0})
    weights = {**target.weights, **aged_out_weights}

    with np.errstate(all="raise"):
        matrix = build_overlap_matrix(source, Clustering(1, target.clusters, weights))

    assert not np.isnan(matrix).any()
    assert matrix.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    for y in target.clusters.values():
        assert overlap(source.clusters[0], y, weights) == 0.0


def test_partial_ageing_uses_target_weights():
    # At t_i all four instances of X weigh 1. By t_j instance 1 has aged out and
    # instance 2 has decayed to 0.5.
    source = _clustering(0, {0: {1, 2, 3, 4}})
    target = _clustering(1, {0: {2, 3}, 1: {4}}, {2: 0.5, 3: 1.0, 4: 1.0})

    matrix = build_overlap_matrix(source, target)

    # t_j weights: (0.5 + 1) / 2.5 and 1 / 2.5.
    np.testing.assert_allclose(matrix, [[0.6, 0.4]], rtol=0, atol=TOLERANCE)
    # t_i weights would instead give 2 / 4 and 1 / 4.
    t_i_weighted = [
        overlap(source.clusters[0], y, source.weights) for y in target.clusters.values()
    ]
    assert t_i_weighted == [0.5, 0.25]


def test_overlap_to_union_equals_sum_of_overlaps():
    # The identity at the end of Section 4.1, which makes splitUnion unnecessary.
    x = {1, 2, 3, 4, 5, 6, 7, 8}
    partition = {0: {1, 2, 20}, 1: {3, 21}, 2: {4, 5, 6}, 3: {22, 23}}
    weights = {
        1: 1.0,
        2: 0.5,
        3: 0.25,
        4: 1.0,
        5: 0.5,
        6: 0.125,
        7: 1.0,  # alive but unassigned
        20: 1.0,
        21: 1.0,
        22: 1.0,
        23: 1.0,
    }  # instance 8 has aged out
    target = Clustering(1, partition, weights)
    matrix = build_overlap_matrix(_clustering(0, {0: x}), target)

    for cluster_ids in [(0, 1), (0, 2), (1, 2, 3), (0, 1, 2, 3)]:
        union = set().union(*(partition[cluster_id] for cluster_id in cluster_ids))
        summed = sum(
            overlap(x, partition[cluster_id], weights) for cluster_id in cluster_ids
        )

        assert overlap(x, union, weights) == pytest.approx(summed, abs=TOLERANCE)
        assert matrix[0, list(cluster_ids)].sum() == pytest.approx(
            summed, abs=TOLERANCE
        )

    # Sum of intersection weights 3.375 over the surviving weight of X, 4.375.
    assert overlap(x, set().union(*partition.values()), weights) == pytest.approx(
        3.375 / 4.375, abs=TOLERANCE
    )


@pytest.mark.parametrize(
    ("clusters", "weights", "message"),
    [
        ({0: {1, 2}, 1: {2, 3}}, {1: 1.0, 2: 1.0, 3: 1.0}, "disjoint"),
        ({0: {1, 2}}, {1: 1.0}, "must have a weight"),
        ({0: {1}}, {1: 1.5}, r"in \[0, 1\]"),
        ({0: {1}}, {1: -0.1}, r"in \[0, 1\]"),
    ],
    ids=["overlapping", "missing-weight", "weight-above-one", "negative-weight"],
)
def test_clustering_rejects_invalid_input(
    clusters: dict[int, set[int]], weights: dict[int, float], message: str
):
    with pytest.raises(ValueError, match=message):
        Clustering(0, clusters, weights)


# External transitions (Section 4.1). Expected labels follow Table 1.

SURVIVED = TransitionType.SURVIVED
ABSORBED = TransitionType.ABSORBED
SPLIT = TransitionType.SPLIT
DISAPPEARED = TransitionType.DISAPPEARED
EMERGED = TransitionType.EMERGED


def _detect(
    source_clusters: dict[int, set[int]],
    target_clusters: dict[int, set[int]],
    alive: set[int] | None = None,
    tau: float = 0.5,
    tau_split: float = 0.1,
) -> tuple[ExternalTransition, ...]:
    """Detect transitions with unit weights.

    :param alive: Instances alive at t_j. Defaults to the clustered instances, so
        that every other instance of t_i has aged out.
    """
    if alive is None:
        alive = set().union(*target_clusters.values())
    return detect_external_transitions(
        _clustering(0, source_clusters),
        Clustering(1, target_clusters, dict.fromkeys(alive, 1.0)),
        tau=tau,
        tau_split=tau_split,
    )


def test_survival():
    transitions = _detect({0: {1, 2, 3, 4}}, {7: {1, 2, 3, 4, 5}})

    assert transitions == (ExternalTransition(SURVIVED, 0, (7,)),)


def test_survival_with_another_cluster_below_tau():
    # Cluster 2 overlaps cluster 5 by 0.4, which is no match, so cluster 0 is
    # the only cluster matched by 5 and survives rather than being absorbed.
    transitions = _detect(
        {0: {1, 2, 3}, 2: {7, 8, 9, 10, 11}},
        {5: {1, 2, 3, 7, 8}},
        alive=set(range(1, 12)),
    )

    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(DISAPPEARED, 2, ()),
    )


def test_absorption_of_three_clusters():
    transitions = _detect(
        {0: {1, 2}, 1: {3, 4}, 2: {5, 6, 7}}, {9: {1, 2, 3, 4, 5, 6, 7}}
    )

    assert transitions == (
        ExternalTransition(ABSORBED, 0, (9,)),
        ExternalTransition(ABSORBED, 1, (9,)),
        ExternalTransition(ABSORBED, 2, (9,)),
    )


def test_absorption_excludes_cluster_matched_below_tau():
    # Cluster 5 is the match of clusters 0 and 1. It is also the cluster that
    # cluster 2 overlaps most, but only by 0.4, so 2 is not absorbed.
    transitions = _detect(
        {0: {1, 2, 3}, 1: {4, 5, 6}, 2: {7, 8, 9, 10, 11}},
        {5: {1, 2, 3, 4, 5, 6, 7, 8}},
        alive=set(range(1, 12)),
    )

    assert transitions == (
        ExternalTransition(ABSORBED, 0, (5,)),
        ExternalTransition(ABSORBED, 1, (5,)),
        ExternalTransition(DISAPPEARED, 2, ()),
    )


def test_two_way_split():
    # Overlaps 0.5 and 0.5: neither is a match at tau = 0.6, together they are.
    transitions = _detect(
        {0: {1, 2, 3, 4}}, {5: {1, 2}, 6: {3, 4}}, tau=0.6, tau_split=0.2
    )

    assert transitions == (ExternalTransition(SPLIT, 0, (5, 6)),)


def test_split_excludes_cluster_below_tau_split():
    # Overlaps 0.4, 0.4 and 0.1; instance 10 is alive but unassigned.
    transitions = _detect(
        {0: set(range(1, 11))},
        {5: {1, 2, 3, 4}, 6: {5, 6, 7, 8}, 7: {9}},
        alive=set(range(1, 11)),
        tau=0.6,
        tau_split=0.2,
    )

    assert transitions == (
        ExternalTransition(SPLIT, 0, (5, 6)),
        ExternalTransition(EMERGED, None, (7,)),
    )


def test_failed_split_disappears():
    # Overlaps 0.2 and 0.2 reach tau_split, but their sum 0.4 is below tau.
    transitions = _detect(
        {0: set(range(1, 11))}, {5: {1, 2}, 6: {3, 4}}, alive=set(range(1, 11))
    )

    assert transitions == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


def test_failed_split_ignores_overlap_below_tau_split():
    # Overlaps 0.3, 0.1, 0.1, 0.1, 0.1 sum to 0.7, but only the first reaches
    # tau_split, so the split candidates sum to 0.3.
    transitions = _detect(
        {0: set(range(1, 11))},
        {5: {1, 2, 3}, 6: {4}, 7: {5}, 8: {6}, 9: {7}},
        alive=set(range(1, 11)),
        tau_split=0.2,
    )

    assert transitions == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
        ExternalTransition(EMERGED, None, (7,)),
        ExternalTransition(EMERGED, None, (8,)),
        ExternalTransition(EMERGED, None, (9,)),
    )


def test_no_candidates_disappears_when_aged_out():
    transitions = _detect({0: {1, 2, 3}}, {5: {10, 11}})

    assert transitions == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
    )


def test_no_candidates_disappears_when_scattered():
    # Four overlaps of 0.25, all below tau_split.
    transitions = _detect(
        {0: {1, 2, 3, 4}}, {5: {1}, 6: {2}, 7: {3}, 8: {4}}, tau_split=0.3
    )

    assert transitions == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
        ExternalTransition(EMERGED, None, (7,)),
        ExternalTransition(EMERGED, None, (8,)),
    )


def test_emergence():
    transitions = _detect({0: {1, 2}}, {5: {1, 2}, 6: {10, 11}})

    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


def test_match_takes_priority_over_split_candidates():
    # Overlaps 0.6 and 0.4: cluster 5 is a match and cluster 6 reaches
    # tau_split. A cluster with a match survives; it is never split.
    transitions = _detect({0: {1, 2, 3, 4, 5}}, {5: {1, 2, 3}, 6: {4, 5}})

    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


def test_absorption_takes_priority_over_split_candidates():
    transitions = _detect(
        {0: {1, 2, 3, 4, 5}, 1: {6, 7}}, {5: {1, 2, 3, 6, 7}, 6: {4, 5}}
    )

    assert transitions == (
        ExternalTransition(ABSORBED, 0, (5,)),
        ExternalTransition(ABSORBED, 1, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


@pytest.mark.parametrize(
    ("target_clusters", "match", "emerged"),
    [
        # Reverse overlaps 1 and 2/3.
        ({5: {1, 2}, 6: {3, 4, 50}}, 5, 6),
        # Reverse overlaps 2/3 and 1.
        ({5: {1, 2, 50}, 6: {3, 4}}, 6, 5),
        # Equal reverse overlaps: the first cluster of the clustering wins.
        ({5: {1, 2}, 6: {3, 4}}, 5, 6),
        ({6: {3, 4}, 5: {1, 2}}, 6, 5),
    ],
    ids=["reverse-first", "reverse-second", "order-first", "order-second"],
)
def test_tie_at_tau_one_half(
    target_clusters: dict[int, set[int]], match: int, emerged: int
):
    # Both clusters overlap cluster 0 by exactly 0.5 = tau.
    transitions = _detect({0: {1, 2, 3, 4}}, target_clusters)

    assert transitions == (
        ExternalTransition(SURVIVED, 0, (match,)),
        ExternalTransition(EMERGED, None, (emerged,)),
    )


def test_cluster_can_be_match_and_split_child():
    # Cluster 5 is the match of cluster 0 and one half of the split of cluster 1.
    transitions = _detect(
        {0: {1, 2, 3, 4}, 1: {10, 11, 12, 13, 14, 15}},
        {5: {1, 2, 3, 4, 10, 11, 12}, 6: {13, 14, 15}},
        tau=0.7,
    )

    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(SPLIT, 1, (5, 6)),
    )


def test_empty_target_clustering():
    transitions = _detect({0: {1, 2}, 1: {3}}, {})

    assert transitions == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(DISAPPEARED, 1, ()),
    )


def test_empty_source_clustering():
    transitions = _detect({}, {5: {1, 2}, 6: {3}})

    assert transitions == (
        ExternalTransition(EMERGED, None, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


def test_both_clusterings_empty():
    assert _detect({}, {}) == ()


@pytest.mark.parametrize(
    ("tau", "tau_split"),
    [(0.5, 0.5), (0.5, 0.6), (0.5, 0.0), (1.1, 0.1)],
    ids=["equal", "split-above-tau", "zero-split", "tau-above-one"],
)
def test_invalid_thresholds_are_rejected(tau: float, tau_split: float):
    with pytest.raises(ValueError, match="0 < tau_split < tau <= 1"):
        _detect({0: {1}}, {5: {1}}, tau=tau, tau_split=tau_split)


def test_tau_below_one_half_warns():
    with pytest.warns(UserWarning, match="below 0.5"):
        transitions = _detect({0: {1, 2, 3, 4}}, {5: {1, 2}, 6: {3}}, tau=0.45)

    # Overlap 2 / 3: instance 4 has aged out.
    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


# Internal transitions (Section 4.2, Table 2) of survived clusters.

SHRUNK = TransitionType.SHRUNK
EXPANDED = TransitionType.EXPANDED
COMPACTED = TransitionType.COMPACTED
DIFFUSED = TransitionType.DIFFUSED
SHIFTED = TransitionType.SHIFTED

Points = dict[int, list[float]]


def _cross(instances: range, radius: float = 1.0, centre: float = 0.0) -> Points:
    """Points at ``radius`` from ``(centre, 0)`` along both axes, repeated to
    cover ``instances``: mean ``(centre, 0)``, sigma ``radius`` and no skewness,
    provided the number of instances is a multiple of four."""
    arms = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    return {
        instance: [centre + radius * arms[i % 4][0], radius * arms[i % 4][1]]
        for i, instance in enumerate(instances)
    }


def _internal(
    source_points: Points,
    target_points: Points,
    source_weights: dict[int, float] | None = None,
    target_weights: dict[int, float] | None = None,
    **thresholds: float,
) -> InternalTransition:
    """Internal transition of cluster 0, which survives as cluster 5."""
    source = Clustering(
        0,
        {0: set(source_points)},
        source_weights or dict.fromkeys(source_points, 1.0),
        source_points,
    )
    target = Clustering(
        1,
        {5: set(target_points)},
        target_weights or dict.fromkeys(target_points, 1.0),
        target_points,
    )
    external = detect_external_transitions(source, target)
    assert external == (ExternalTransition(TransitionType.SURVIVED, 0, (5,)),)

    with np.errstate(all="raise"):
        (internal,) = detect_internal_transitions(
            source, target, external, **thresholds
        )
    return internal


def test_internal_no_change_is_empty_set():
    internal = _internal(_cross(range(4)), _cross(range(4)))

    assert internal == InternalTransition(0, 5, frozenset(), skipped=False)


def test_internal_shrinks():
    # Eight instances of weight 1 become four; mean, sigma and skewness are unchanged.
    assert _internal(_cross(range(8)), _cross(range(4))).kinds == {SHRUNK}


def test_internal_expands():
    assert _internal(_cross(range(4)), _cross(range(8))).kinds == {EXPANDED}


def test_internal_becomes_more_compact():
    # Sigma goes from 1 to 0.5.
    assert _internal(_cross(range(4)), _cross(range(4), radius=0.5)).kinds == {
        COMPACTED
    }


def test_internal_becomes_more_diffuse():
    # Sigma goes from 1 to 2.
    assert _internal(_cross(range(4)), _cross(range(4), radius=2.0)).kinds == {DIFFUSED}


@pytest.mark.parametrize(
    ("shift", "expected"),
    [(0.4, set()), (0.6, {SHIFTED}), (-0.6, {SHIFTED}), (3.0, {SHIFTED})],
)
def test_internal_mean_shift(shift: float, expected: set[TransitionType]):
    # The mean moves by `shift` while sigma stays 1, against tau_mean = 0.5.
    internal = _internal(_cross(range(4)), _cross(range(4), centre=shift))

    assert internal.kinds == expected


def test_internal_skewness_shift():
    # Both clusters have mean 0 and sigma 1. The skewness goes from 0 to
    # 2 / sqrt(3), about 1.15, against tau_skew = 0.5.
    root = 3**0.5
    source_points = {0: [-1.0], 1: [-1.0], 2: [1.0], 3: [1.0]}
    target_points = {0: [root], 1: [-1 / root], 2: [-1 / root], 3: [-1 / root]}

    assert _internal(source_points, target_points).kinds == {SHIFTED}
    assert _internal(source_points, target_points, tau_skew=1.2).kinds == set()


def test_internal_groups_combine():
    # Twice the instances, half the sigma, and a mean moved by 3.
    internal = _internal(_cross(range(4)), _cross(range(8), radius=0.5, centre=3.0))

    assert internal.kinds == {EXPANDED, COMPACTED, SHIFTED}


def test_internal_statistics_use_each_clusters_own_weights():
    # Every instance decays from weight 1 at t_i to 0.5 at t_j, so the cluster
    # weighs 4 and then 2. With t_j weights on both sides nothing would change.
    points = _cross(range(4))
    internal = _internal(points, points, target_weights=dict.fromkeys(points, 0.5))

    assert internal.kinds == {SHRUNK}


def test_internal_weighted_moments():
    # At t_j the instance at (1, 0) weighs 1 and the other three weigh 0.25. The
    # weighted mean moves to (3 / 7, 0), about 0.43, and sigma becomes
    # sqrt(40) / 7, about 0.90. tau_skew is raised to isolate the mean.
    points = _cross(range(4))
    target_weights = {0: 1.0, 1: 0.25, 2: 0.25, 3: 0.25}

    def kinds(tau_mean: float) -> frozenset[TransitionType]:
        return _internal(
            points,
            points,
            target_weights=target_weights,
            tau_mean=tau_mean,
            tau_skew=1.0,
        ).kinds

    assert kinds(0.4) == {SHRUNK, SHIFTED}
    assert kinds(0.45) == {SHRUNK}


@pytest.mark.parametrize("n_instances", [4, 400, 40_000])
def test_internal_size_thresholds_are_relative(n_instances: int):
    # epsilon = 0.1 is a fraction of the cluster weight: 5% growth is no
    # change and 25% growth is an expansion, whatever the absolute difference.
    source_points = _cross(range(n_instances))

    grown_5_percent = _cross(range(n_instances * 105 // 100))
    grown_25_percent = _cross(range(n_instances * 125 // 100))

    assert EXPANDED not in _internal(source_points, grown_5_percent).kinds
    assert EXPANDED in _internal(source_points, grown_25_percent).kinds


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_internal_feature_thresholds_are_scale_invariant(scale: float):
    # delta and tau_mean are fractions of sigma(X), so rescaling the features
    # of both clusters changes no label.
    def scaled(points: Points) -> Points:
        return {
            instance: [scale * v for v in point] for instance, point in points.items()
        }

    source_points = scaled(_cross(range(4)))

    assert (
        _internal(source_points, scaled(_cross(range(4), radius=1.05))).kinds == set()
    )
    assert _internal(source_points, scaled(_cross(range(4), radius=1.2))).kinds == {
        DIFFUSED
    }
    assert _internal(source_points, scaled(_cross(range(4), centre=0.4))).kinds == set()
    assert _internal(source_points, scaled(_cross(range(4), centre=0.6))).kinds == {
        SHIFTED
    }


def test_internal_zero_spread_cluster():
    # sigma(X) = 0: any spread is more diffuse and any move is a shift.
    point = {instance: [2.0, 3.0] for instance in range(4)}
    moved = {instance: [2.0, 3.5] for instance in range(4)}
    spread = {0: [3.0, 3.0], 1: [1.0, 3.0], 2: [2.0, 4.0], 3: [2.0, 2.0]}

    assert _internal(point, point).kinds == set()
    assert _internal(point, moved).kinds == {SHIFTED}
    assert _internal(point, spread).kinds == {DIFFUSED}
    assert _internal(spread, point).kinds == {COMPACTED}


def test_internal_constant_feature_has_no_skewness():
    # 0.1 is not exactly representable, so the second feature has a variance
    # of rounding error only. Its skewness must be 0, not noise.
    source_points = {instance: [float(instance), 0.1] for instance in range(7)}
    target_points = {instance: [float(instance), 0.1 * 3] for instance in range(7)}

    assert _internal(source_points, target_points, tau_mean=1.0).kinds == set()


def test_internal_skipped_when_feature_dimensions_differ():
    source_points = _cross(range(4))
    target_points = {
        instance: [*point, 0.0] for instance, point in _cross(range(8)).items()
    }

    internal = _internal(source_points, target_points)

    assert internal == InternalTransition(0, 5, frozenset(), skipped=True)


@pytest.mark.parametrize("missing", ["source", "target", "both"])
def test_internal_skipped_without_points(missing: str):
    points = _cross(range(4))
    weights = dict.fromkeys(points, 1.0)
    source = Clustering(
        0, {0: set(points)}, weights, None if missing != "target" else points
    )
    target = Clustering(
        1, {5: set(points)}, weights, None if missing != "source" else points
    )

    internal = detect_internal_transitions(
        source, target, detect_external_transitions(source, target)
    )

    assert internal == (InternalTransition(0, 5, frozenset(), skipped=True),)


def test_internal_only_for_survived_clusters():
    # Cluster 0 survives as 5, clusters 1 and 2 are absorbed by 6, cluster 3
    # disappears and cluster 7 emerges.
    points = {instance: [float(instance), 0.0] for instance in range(12)}
    weights = dict.fromkeys(points, 1.0)
    source = Clustering(
        0, {0: {0, 1}, 1: {2, 3}, 2: {4, 5}, 3: {6, 7}}, weights, points
    )
    target = Clustering(
        1,
        {5: {0, 1}, 6: {2, 3, 4, 5}, 7: {10, 11}},
        {r: 1.0 for r in (0, 1, 2, 3, 4, 5, 10, 11)},
        points,
    )
    external = detect_external_transitions(source, target)
    assert [t.kind for t in external] == [
        SURVIVED,
        ABSORBED,
        ABSORBED,
        DISAPPEARED,
        EMERGED,
    ]

    internal = detect_internal_transitions(source, target, external)

    assert internal == (InternalTransition(0, 5, frozenset(), skipped=False),)


@pytest.mark.parametrize("name", ["epsilon", "delta", "tau_mean", "tau_skew"])
def test_internal_rejects_negative_threshold(name: str):
    with pytest.raises(ValueError, match="non-negative"):
        _internal(_cross(range(4)), _cross(range(4)), **{name: -0.1})


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ({1: [0.0, 0.0]}, "must have a point"),
        ({1: [0.0, 0.0], 2: [0.0]}, "same number of features"),
        ({1: [[0.0]], 2: [[0.0]]}, "same number of features"),
    ],
    ids=["missing-point", "ragged", "not-a-vector"],
)
def test_clustering_rejects_invalid_points(points: dict, message: str):
    with pytest.raises(ValueError, match=message):
        Clustering(0, {0: {1, 2}}, {1: 1.0, 2: 1.0}, points)


# Lifetimes (Definition 5) and ratios (Definition 6).


def _unit_clustering(timepoint: float, clusters: dict[int, set[int]]) -> Clustering:
    """Clustering whose alive instances are the clustered ones, all of weight 1 and
    all at the same point, so that only a change of size is an internal
    transition."""
    instances = set().union(*clusters.values())
    return Clustering(
        timepoint,
        clusters,
        dict.fromkeys(instances, 1.0),
        {instance: [0.0] for instance in instances},
    )


def _lifetime_history(
    clusterings: list[dict[int, set[int]]], tau: float = 0.5
) -> list[dict[int, Lifetime]]:
    """Lifetimes of the clusters of each clustering, starting from nothing."""
    history = []
    lifetimes: dict[int, Lifetime] = {}
    source = Clustering(-1, {}, {})
    for timepoint, clusters in enumerate(clusterings):
        target = _unit_clustering(timepoint, clusters)
        external = detect_external_transitions(source, target, tau=tau)
        internal = detect_internal_transitions(source, target, external)
        lifetimes = advance_lifetimes(lifetimes, source, target, external, internal)
        history.append(lifetimes)
        source = target
    return history


# Five timepoints, with tau = 0.6. Cluster ids are 10 * timepoint + position.
#
# t0 -> t1: 0 survives unchanged as 10; 1 survives as 11 and expands;
#           2 and 3 are absorbed by 12, 2 being the larger.
# t1 -> t2: 10, 11 and 12 survive unchanged as 20, 21 and 22; 23 emerges.
# t2 -> t3: 20 survives as 30 and expands; 21 is split into 31 and 32;
#           22 and 23 are absorbed by 33, 22 being the larger.
# t3 -> t4: 30, 31 and 33 survive unchanged as 40, 41 and 43; 32 disappears.
FIVE_TIMEPOINTS = [
    {0: {1, 2, 3, 4}, 1: {5, 6, 7, 8}, 2: {9, 10, 11}, 3: {12, 13}},
    {10: {1, 2, 3, 4}, 11: {5, 6, 7, 8, 20, 21}, 12: {9, 10, 11, 12, 13}},
    {
        20: {1, 2, 3, 4},
        21: {5, 6, 7, 8, 20, 21},
        22: {9, 10, 11, 12, 13},
        23: {30, 31, 32},
    },
    {
        30: {1, 2, 3, 4, 40, 41},
        31: {5, 6, 7},
        32: {8, 20, 21},
        33: {9, 10, 11, 12, 13, 30, 31, 32},
    },
    {
        40: {1, 2, 3, 4, 40, 41},
        41: {5, 6, 7},
        43: {9, 10, 11, 12, 13, 30, 31, 32},
    },
]


def test_lifetimes_over_five_timepoints():
    history = _lifetime_history(FIVE_TIMEPOINTS, tau=0.6)

    # Lifetime(strict, internal, absorption)
    assert history == [
        {
            0: Lifetime(1, 1, 1),
            1: Lifetime(1, 1, 1),
            2: Lifetime(1, 1, 1),
            3: Lifetime(1, 1, 1),
        },
        {
            10: Lifetime(2, 2, 2),
            11: Lifetime(1, 2, 2),
            12: Lifetime(1, 1, 2),
        },
        {
            20: Lifetime(3, 3, 3),
            21: Lifetime(2, 3, 3),
            22: Lifetime(2, 2, 3),
            23: Lifetime(1, 1, 1),
        },
        {
            30: Lifetime(1, 4, 4),
            31: Lifetime(1, 1, 1),
            32: Lifetime(1, 1, 1),
            33: Lifetime(1, 1, 4),
        },
        {
            40: Lifetime(2, 5, 5),
            41: Lifetime(2, 2, 2),
            43: Lifetime(2, 2, 5),
        },
    ]


def test_five_timepoint_history_has_the_intended_transitions():
    kinds = []
    source = Clustering(-1, {}, {})
    for timepoint, clusters in enumerate(FIVE_TIMEPOINTS):
        target = _unit_clustering(timepoint, clusters)
        external = detect_external_transitions(source, target, tau=0.6)
        kinds.append([t.kind for t in external])
        source = target

    assert kinds == [
        [EMERGED, EMERGED, EMERGED, EMERGED],
        [SURVIVED, SURVIVED, ABSORBED, ABSORBED],
        [SURVIVED, SURVIVED, SURVIVED, EMERGED],
        [SURVIVED, SPLIT, ABSORBED, ABSORBED],
        [SURVIVED, SURVIVED, DISAPPEARED, SURVIVED],
    ]


def test_lifetime_invariant_over_five_timepoints():
    for lifetimes in _lifetime_history(FIVE_TIMEPOINTS, tau=0.6):
        for lifetime in lifetimes.values():
            assert 1 <= lifetime.strict <= lifetime.internal <= lifetime.absorption


def test_absorption_continues_the_heavier_cluster():
    # Cluster 0 is older but contributes 2 instances to cluster 5; cluster 1 is
    # younger and contributes 3.
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4, 5}})
    target = _unit_clustering(1, {5: {1, 2, 3, 4, 5}})
    external = detect_external_transitions(source, target)

    lifetimes = advance_lifetimes(
        {0: Lifetime(4, 6, 7), 1: Lifetime(1, 2, 2)}, source, target, external, ()
    )

    assert lifetimes == {5: Lifetime(1, 1, 3)}


def test_absorption_tie_continues_the_longer_lived_cluster():
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4}})
    target = _unit_clustering(1, {5: {1, 2, 3, 4}})
    external = detect_external_transitions(source, target)

    lifetimes = advance_lifetimes(
        {0: Lifetime(1, 2, 2), 1: Lifetime(4, 6, 7)}, source, target, external, ()
    )

    assert lifetimes == {5: Lifetime(1, 1, 8)}


def test_split_extends_no_lifetime():
    source = _unit_clustering(0, {0: {1, 2, 3, 4}})
    target = _unit_clustering(1, {5: {1, 2}, 6: {3, 4}})
    external = detect_external_transitions(source, target, tau=0.6)
    assert [t.kind for t in external] == [SPLIT]

    lifetimes = advance_lifetimes({0: Lifetime(4, 6, 7)}, source, target, external, ())

    assert lifetimes == {5: Lifetime(1, 1, 1), 6: Lifetime(1, 1, 1)}


def test_match_takes_priority_over_split_child_for_lifetime():
    # Cluster 5 is the survival of cluster 0 and a split child of cluster 1.
    source = _unit_clustering(0, {0: {1, 2, 3, 4}, 1: {10, 11, 12, 13, 14, 15}})
    target = _unit_clustering(1, {5: {1, 2, 3, 4, 10, 11, 12}, 6: {13, 14, 15}})
    external = detect_external_transitions(source, target, tau=0.7)
    internal = detect_internal_transitions(source, target, external)
    assert [t.kind for t in external] == [SURVIVED, SPLIT]

    lifetimes = advance_lifetimes(
        {0: Lifetime(3, 3, 3), 1: Lifetime(5, 5, 5)}, source, target, external, internal
    )

    # Cluster 5 expanded, so its strict lifetime restarts.
    assert lifetimes == {5: Lifetime(1, 4, 4), 6: Lifetime(1, 1, 1)}


def test_skipped_internal_analysis_restarts_strict_lifetime():
    # Without points the internal analysis is skipped, so the survival cannot
    # be shown to be free of internal transitions.
    source = _clustering(0, {0: {1, 2}})
    target = _clustering(1, {5: {1, 2}})
    external = detect_external_transitions(source, target)
    internal = detect_internal_transitions(source, target, external)
    assert internal == (InternalTransition(0, 5, frozenset(), skipped=True),)

    lifetimes = advance_lifetimes(
        {0: Lifetime(3, 3, 3)}, source, target, external, internal
    )

    assert lifetimes == {5: Lifetime(1, 4, 4)}


def test_ratios_when_everything_survives():
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4}, 2: {5}})
    target = _unit_clustering(1, {5: {1, 2}, 6: {3, 4}, 7: {5}, 8: {9}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=1.0, absorption=0.0)
    assert ratios.passforward == 1.0


def test_ratios_when_everything_dies():
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4}, 2: {5}})
    target = _unit_clustering(1, {5: {10, 11}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=0.0, absorption=0.0)
    assert ratios.passforward == 0.0


def test_ratios_mixed():
    # Of the four clusters, 0 survives, 1 and 2 are absorbed and 3 disappears.
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4}, 2: {5, 6}, 3: {7, 8}})
    target = _unit_clustering(1, {5: {1, 2}, 6: {3, 4, 5, 6}, 7: {9}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=0.25, absorption=0.5)
    assert ratios.passforward == 0.75


def test_ratios_do_not_count_splits():
    # Cluster 0 survives and cluster 1 is split.
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4, 5, 6}})
    target = _unit_clustering(1, {5: {1, 2}, 6: {3, 4}, 7: {5, 6}})

    ratios = passforward_ratios(detect_external_transitions(source, target, tau=0.6))

    assert ratios == PassforwardRatios(survival=0.5, absorption=0.0)


def test_ratios_of_empty_clustering_are_zero():
    target = _unit_clustering(1, {5: {1, 2}})
    external = detect_external_transitions(Clustering(0, {}, {}), target)

    assert passforward_ratios(external) == PassforwardRatios(0.0, 0.0)


# MONIC over a sequence of clusterings.


def test_monic_reports_first_clustering_as_emerged():
    report = MONIC().update(_unit_clustering(3, {0: {1, 2}, 1: {3}}))

    assert report.source_timepoint is None
    assert report.target_timepoint == 3
    assert report.overlap.shape == (0, 2)
    assert report.external == (
        ExternalTransition(EMERGED, None, (0,)),
        ExternalTransition(EMERGED, None, (1,)),
    )
    assert report.internal == ()
    assert report.lifetimes == {0: Lifetime(1, 1, 1), 1: Lifetime(1, 1, 1)}
    assert report.ratios == PassforwardRatios(0.0, 0.0)


def test_monic_matches_step_by_step_lifetimes():
    monic = MONIC(tau=0.6)
    reports = [
        monic.update(_unit_clustering(timepoint, clusters))
        for timepoint, clusters in enumerate(FIVE_TIMEPOINTS)
    ]

    assert [report.lifetimes for report in reports] == _lifetime_history(
        FIVE_TIMEPOINTS, tau=0.6
    )
    assert [report.source_timepoint for report in reports] == [None, 0, 1, 2, 3]
    assert monic.lifetimes == reports[-1].lifetimes
    # t2 -> t3: one survival, one split and two absorptions out of four clusters.
    assert reports[3].ratios == PassforwardRatios(survival=0.25, absorption=0.5)
    assert reports[3].overlap.shape == (4, 4)


def test_monic_rejects_timepoint_that_is_not_later():
    monic = MONIC()
    monic.update(_unit_clustering(1, {0: {1}}))

    with pytest.raises(ValueError, match="not later"):
        monic.update(_unit_clustering(1, {0: {1}}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0.5, "tau_split": 0.5},
        {"tau": 0.5, "tau_split": 0.7},
        {"tau_split": 0.0},
        {"tau": 1.5},
        {"epsilon": -1.0},
    ],
)
def test_monic_validates_thresholds_at_construction(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        MONIC(**kwargs)


def test_garbage_cluster_wins_matches_unless_dissolved():
    # Cluster 0 scatters: at t_j all of its instances are in the garbage cluster
    # 9, along with the leftovers of other clusters.
    source = _unit_clustering(0, {0: {1, 2, 3, 4}})
    target = _unit_clustering(1, {9: {1, 2, 3, 4, 20, 21, 22, 23, 24}, 5: {30, 31}})

    assert detect_external_transitions(source, target) == (
        ExternalTransition(SURVIVED, 0, (9,)),
        ExternalTransition(EMERGED, None, (5,)),
    )
    assert detect_external_transitions(source, target.without_clusters([9])) == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
    )
    # The dissolved instances are still alive, so they keep their weights.
    assert target.without_clusters([9]).weights == target.weights


# Monitoring a clusterer.


def _stream(n_instances: int) -> NumpyStream:
    """A stream whose instance ``i`` has the single feature ``i``."""
    return NumpyStream(
        np.arange(n_instances, dtype=float)[:, np.newaxis],
        np.zeros(n_instances, dtype=int),
        target_type="categorical",
    )


class _PlainClusterer(Clusterer):
    """A clusterer that records what it is trained on and has no clusters."""

    def __init__(self):
        super().__init__(_stream(1).get_schema())
        self.trained: list[float] = []

    def __str__(self):
        return "PlainClusterer"

    def train(self, instance: Instance):
        self.trained.append(float(instance.x[0]))

    def implements_micro_clusters(self) -> bool:
        return False

    def implements_macro_clusters(self) -> bool:
        return False

    def _get_micro_clusters_centers(self):
        return []

    def _get_micro_clusters_radii(self):
        return []

    def _get_micro_clusters_weights(self):
        return []

    def _get_clusters_centers(self):
        return []

    def _get_clusters_radii(self):
        return []

    def _get_clusters_weights(self):
        return []

    def get_clustering_result(self):
        return ClusteringResult([], [], [], [])

    def get_micro_clustering_result(self):
        return ClusteringResult([], [], [], [])


class _ScriptedClusterer(_PlainClusterer, MonitorableClusterer):
    """A clusterer whose clusterings are given in advance, one per call."""

    def __init__(self, clusterings: list[Clustering]):
        super().__init__()
        self.clusterings = iter(clusterings)

    def get_clustering(self) -> Clustering:
        return next(self.clusterings)


# Cluster 0 splits in two at timepoint 2, while cluster 1 survives. At
# timepoint 3 instances 1 and 2 have aged out, so the cluster holding them
# disappears, and the other two are absorbed by one cluster.
SCRIPT = [
    _unit_clustering(1, {0: {1, 2, 3, 4}, 1: {5, 6}}),
    _unit_clustering(2, {0: {1, 2}, 1: {3, 4}, 2: {5, 6, 7}}),
    _unit_clustering(3, {0: {3, 4, 5, 6, 7}}),
]


def test_monitor_asks_for_a_clustering_at_every_interval():
    clusterer = _ScriptedClusterer(SCRIPT)
    monitor = MONICMonitor(clusterer, check_interval=2, tau=0.6, debug=True)
    stream = _stream(7)

    reports = [monitor.train(stream.next_instance()) for _ in range(7)]

    assert [report is not None for report in reports] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert clusterer.trained == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert monitor.instances_seen == 7
    assert monitor.clustering is SCRIPT[2]
    assert [report.external for report in reports if report is not None] == [
        (
            ExternalTransition(EMERGED, None, (0,)),
            ExternalTransition(EMERGED, None, (1,)),
        ),
        (
            ExternalTransition(SPLIT, 0, (0, 1)),
            ExternalTransition(SURVIVED, 1, (2,)),
        ),
        (
            ExternalTransition(DISAPPEARED, 0, ()),
            ExternalTransition(ABSORBED, 1, (0,)),
            ExternalTransition(ABSORBED, 2, (0,)),
        ),
    ]


def test_monitor_stops_at_max_instances():
    monitor = MONICMonitor(_ScriptedClusterer(SCRIPT), check_interval=2, tau=0.6)

    reports = list(monitor.monitor(_stream(10), max_instances=5))

    assert [report.target_timepoint for report in reports] == [1, 2]
    assert monitor.instances_seen == 5


def test_monitor_rejects_clusterer_without_clusterings():
    with pytest.raises(TypeError, match="does not implement MonitorableClusterer"):
        MONICMonitor(_PlainClusterer())


@pytest.mark.parametrize("kwargs", [{"check_interval": 0}, {"tau_split": 0.9}])
def test_monitor_validates_arguments(kwargs: dict):
    with pytest.raises(ValueError):
        MONICMonitor(_ScriptedClusterer([]), **kwargs)


# The runtime invariant checks of debug mode.


def _report_for_invariants():
    """A report with a survival, a split, two disappearances and an emergence.
    Instance 10 of cluster 2 is alive at t_j but unassigned, and cluster 3 has
    aged out."""
    monic = MONIC(tau=0.6, debug=True)
    source = _unit_clustering(0, {0: {1, 2}, 1: {3, 4, 5, 6}, 2: {10, 11}, 3: {30}})
    target = Clustering(
        1,
        {5: {1, 2}, 6: {3, 4}, 7: {5, 6}, 8: {20}},
        dict.fromkeys([1, 2, 3, 4, 5, 6, 10, 20], 1.0),
    )
    monic.update(source)
    return source, target, monic.update(target)


def test_debug_mode_accepts_valid_reports():
    source, target, report = _report_for_invariants()

    assert [t.kind for t in report.external] == [
        SURVIVED,
        SPLIT,
        DISAPPEARED,
        DISAPPEARED,
        EMERGED,
    ]
    assert report.overlap.sum(axis=1).tolist() == [1.0, 1.0, 0.0, 0.0]
    _check_invariants(source, target, report)


def test_debug_mode_runs_over_a_history():
    monic = MONIC(tau=0.6, debug=True)

    for timepoint, clusters in enumerate(FIVE_TIMEPOINTS):
        monic.update(_unit_clustering(timepoint, clusters))


def _corrupt_row_sum_above_one(report):
    overlap = report.overlap.copy()
    overlap[0, 1] = 0.5
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_row_sum_below_one(report):
    overlap = report.overlap.copy()
    overlap[1, 2] = 0.25
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_row_sum_with_unassigned_instance(report):
    overlap = report.overlap.copy()
    overlap[2, 0] = 1.5
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_row_sum_of_aged_out_cluster(report):
    overlap = report.overlap.copy()
    overlap[3, 0] = 0.5
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_unlabelled_cluster(report):
    return dataclasses.replace(report, external=report.external[1:])


def _corrupt_cluster_labelled_twice(report):
    return dataclasses.replace(report, external=(report.external[0], *report.external))


def _corrupt_counts(report):
    # Cluster 2 is still labelled once, but as emerged, which is not one of
    # the four outcomes of a cluster of the earlier clustering.
    relabelled = ExternalTransition(EMERGED, 2, ())
    return dataclasses.replace(
        report, external=(*report.external[:2], relabelled, *report.external[3:])
    )


def _corrupt_lifetime_order(report):
    return dataclasses.replace(
        report, lifetimes={**report.lifetimes, 5: Lifetime(3, 2, 2)}
    )


def _corrupt_missing_lifetime(report):
    lifetimes = dict(report.lifetimes)
    del lifetimes[8]
    return dataclasses.replace(report, lifetimes=lifetimes)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_corrupt_row_sum_above_one, "overlaps of cluster 0 sum to 1.5, expected 1"),
        (_corrupt_row_sum_below_one, "overlaps of cluster 1 sum to 0.75, expected 1"),
        (
            _corrupt_row_sum_with_unassigned_instance,
            "overlaps of cluster 2 sum to 1.5, expected at most 1",
        ),
        (
            _corrupt_row_sum_of_aged_out_cluster,
            "overlaps of cluster 3 sum to 0.5, expected 0",
        ),
        (_corrupt_unlabelled_cluster, "expected one each for"),
        (_corrupt_cluster_labelled_twice, "expected one each for"),
        (_corrupt_counts, "3 clusters survived, .* out of 4"),
        (_corrupt_lifetime_order, "expected 1 <= strict <= internal <= absorption"),
        (_corrupt_missing_lifetime, "lifetimes are for clusters"),
    ],
)
def test_debug_mode_rejects_corrupt_reports(corrupt, message: str):
    source, target, report = _report_for_invariants()

    with pytest.raises(AssertionError, match=message):
        _check_invariants(source, target, corrupt(report))


def test_debug_mode_row_sum_may_be_below_one_with_unassigned_instances():
    # Instance 4 is alive at t_j but in no cluster, so the row sums to 0.75.
    monic = MONIC(debug=True)
    monic.update(_unit_clustering(0, {0: {1, 2, 3, 4}}))

    report = monic.update(
        Clustering(1, {5: {1, 2, 3}}, dict.fromkeys([1, 2, 3, 4], 1.0))
    )

    assert report.overlap.tolist() == [[0.75]]
