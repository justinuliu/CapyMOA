import dataclasses
from collections import deque

import numpy as np
import pytest

from capymoa.base import Clusterer, ClusteringResult
from capymoa.cluster import Denstream_with_dbscan
from capymoa.core import Instance
from capymoa.evaluation.monic import (
    MONIC,
    ClusteringSnapshot,
    ExponentialDecay,
    ExternalKind,
    ExternalTransition,
    InternalKind,
    InternalTransition,
    Lifetime,
    MONICMonitor,
    PassforwardRatios,
    RecordBuffer,
    SlidingWindow,
    _check_invariants,
    advance_lifetimes,
    build_overlap_matrix,
    detect_external_transitions,
    detect_internal_transitions,
    overlap,
    passforward_ratios,
    snapshot_from_clusterer,
)
from capymoa.stream import NumpyStream, Schema

TOLERANCE = 1e-9


def _snapshot(
    timepoint: float,
    clusters: dict[int, set[int]],
    weights: dict[int, float] | None = None,
) -> ClusteringSnapshot:
    """Build a snapshot, weighting every clustered record 1 unless told otherwise."""
    if weights is None:
        weights = {record: 1.0 for records in clusters.values() for record in records}
    return ClusteringSnapshot(timepoint, clusters, weights)


# Five clusters at t_i over records 1..19.
SOURCE_CLUSTERS = {
    0: {1, 2, 3, 4},
    1: {5, 6, 7, 8},
    2: {9, 10, 11, 12, 13},
    3: {14, 15},
    4: {16, 17, 18, 19},
}
# Four clusters at t_j over the same records plus the new records 20 and 21.
TARGET_CLUSTERS = {
    0: {1, 2, 3, 4, 16, 20, 21},
    1: {5, 6, 7, 8, 14, 15},
    2: {9, 10, 11},
    3: {12, 13, 17, 18, 19},
}


def test_overlap_matrix_known_values():
    matrix = build_overlap_matrix(
        _snapshot(0, SOURCE_CLUSTERS), _snapshot(1, TARGET_CLUSTERS)
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
    source = _snapshot(0, SOURCE_CLUSTERS)
    target = _snapshot(1, TARGET_CLUSTERS)
    matrix = build_overlap_matrix(source, target)

    for row, x in enumerate(source.clusters.values()):
        for column, y in enumerate(target.clusters.values()):
            assert matrix[row, column] == pytest.approx(
                overlap(x, y, target.weights), abs=TOLERANCE
            )


@pytest.mark.parametrize(
    "old_weight", [1.0, 0.5, 0.125], ids=["uniform", "half", "eighth"]
)
def test_rows_sum_to_one_when_all_records_survive(old_weight: float):
    # Every t_i record keeps a positive weight at t_j and is assigned to a cluster.
    target_weights = {record: old_weight for record in range(1, 20)}
    target_weights |= {20: 1.0, 21: 1.0}
    # Unequal weights inside a cluster, so the weights do not simply cancel out.
    target_weights |= {9: 1.0, 16: 1.0}

    matrix = build_overlap_matrix(
        _snapshot(0, SOURCE_CLUSTERS), _snapshot(1, TARGET_CLUSTERS, target_weights)
    )

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=TOLERANCE)


def test_row_sums_below_one_when_surviving_record_is_unassigned():
    # Record 4 is still alive at t_j but belongs to no cluster.
    source = _snapshot(0, {0: {1, 2, 3, 4}})
    target = _snapshot(1, {0: {1, 2, 3}}, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})

    assert build_overlap_matrix(source, target).tolist() == [[0.75]]


@pytest.mark.parametrize(
    "aged_out_weights",
    [{}, {1: 0.0, 2: 0.0, 3: 0.0}],
    ids=["absent", "zero-weight"],
)
def test_fully_aged_out_cluster_has_zero_overlap(aged_out_weights: dict[int, float]):
    # Cluster 0 of t_i has aged out completely by t_j; cluster 1 has not.
    source = _snapshot(0, {0: {1, 2, 3}, 1: {4, 5}})
    target = _snapshot(1, {0: {4, 5, 6}, 1: {7}}, {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0})
    weights = {**target.weights, **aged_out_weights}

    with np.errstate(all="raise"):
        matrix = build_overlap_matrix(
            source, ClusteringSnapshot(1, target.clusters, weights)
        )

    assert not np.isnan(matrix).any()
    assert matrix.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    for y in target.clusters.values():
        assert overlap(source.clusters[0], y, weights) == 0.0


def test_partial_ageing_uses_target_weights():
    # At t_i all four records of X weigh 1. By t_j record 1 has aged out and
    # record 2 has decayed to 0.5.
    source = _snapshot(0, {0: {1, 2, 3, 4}})
    target = _snapshot(1, {0: {2, 3}, 1: {4}}, {2: 0.5, 3: 1.0, 4: 1.0})

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
    }  # record 8 has aged out
    target = ClusteringSnapshot(1, partition, weights)
    matrix = build_overlap_matrix(_snapshot(0, {0: x}), target)

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
def test_snapshot_rejects_invalid_input(
    clusters: dict[int, set[int]], weights: dict[int, float], message: str
):
    with pytest.raises(ValueError, match=message):
        ClusteringSnapshot(0, clusters, weights)


# External transitions (Section 4.1). Expected labels follow Table 1.

SURVIVED = ExternalKind.SURVIVED
ABSORBED = ExternalKind.ABSORBED
SPLIT = ExternalKind.SPLIT
DISAPPEARED = ExternalKind.DISAPPEARED
EMERGED = ExternalKind.EMERGED


def _detect(
    source_clusters: dict[int, set[int]],
    target_clusters: dict[int, set[int]],
    alive: set[int] | None = None,
    tau: float = 0.5,
    tau_split: float = 0.1,
) -> tuple[ExternalTransition, ...]:
    """Detect transitions with unit weights.

    :param alive: Records alive at t_j. Defaults to the clustered records, so
        that every other record of t_i has aged out.
    """
    if alive is None:
        alive = set().union(*target_clusters.values())
    return detect_external_transitions(
        _snapshot(0, source_clusters),
        ClusteringSnapshot(1, target_clusters, dict.fromkeys(alive, 1.0)),
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
    # Overlaps 0.4, 0.4 and 0.1; record 10 is alive but unassigned.
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
        # Equal reverse overlaps: the first cluster of the snapshot wins.
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

    # Overlap 2 / 3: record 4 has aged out.
    assert transitions == (
        ExternalTransition(SURVIVED, 0, (5,)),
        ExternalTransition(EMERGED, None, (6,)),
    )


# Internal transitions (Section 4.2, Table 2) of survived clusters.

SHRUNK = InternalKind.SHRUNK
EXPANDED = InternalKind.EXPANDED
COMPACTED = InternalKind.COMPACTED
DIFFUSED = InternalKind.DIFFUSED
SHIFTED = InternalKind.SHIFTED

Points = dict[int, list[float]]


def _cross(records: range, radius: float = 1.0, centre: float = 0.0) -> Points:
    """Points at ``radius`` from ``(centre, 0)`` along both axes, repeated to
    cover ``records``: mean ``(centre, 0)``, sigma ``radius`` and no skewness,
    provided the number of records is a multiple of four."""
    arms = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    return {
        record: [centre + radius * arms[i % 4][0], radius * arms[i % 4][1]]
        for i, record in enumerate(records)
    }


def _internal(
    source_points: Points,
    target_points: Points,
    source_weights: dict[int, float] | None = None,
    target_weights: dict[int, float] | None = None,
    **thresholds: float,
) -> InternalTransition:
    """Internal transition of cluster 0, which survives as cluster 5."""
    source = ClusteringSnapshot(
        0,
        {0: set(source_points)},
        source_weights or dict.fromkeys(source_points, 1.0),
        source_points,
    )
    target = ClusteringSnapshot(
        1,
        {5: set(target_points)},
        target_weights or dict.fromkeys(target_points, 1.0),
        target_points,
    )
    external = detect_external_transitions(source, target)
    assert external == (ExternalTransition(ExternalKind.SURVIVED, 0, (5,)),)

    with np.errstate(all="raise"):
        (internal,) = detect_internal_transitions(
            source, target, external, **thresholds
        )
    return internal


def test_internal_no_change_is_empty_set():
    internal = _internal(_cross(range(4)), _cross(range(4)))

    assert internal == InternalTransition(0, 5, frozenset(), skipped=False)


def test_internal_shrinks():
    # Eight records of weight 1 become four; mean, sigma and skewness are unchanged.
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
def test_internal_mean_shift(shift: float, expected: set[InternalKind]):
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
    # Twice the records, half the sigma, and a mean moved by 3.
    internal = _internal(_cross(range(4)), _cross(range(8), radius=0.5, centre=3.0))

    assert internal.kinds == {EXPANDED, COMPACTED, SHIFTED}


def test_internal_statistics_use_each_clusters_own_weights():
    # Every record decays from weight 1 at t_i to 0.5 at t_j, so the cluster
    # weighs 4 and then 2. With t_j weights on both sides nothing would change.
    points = _cross(range(4))
    internal = _internal(points, points, target_weights=dict.fromkeys(points, 0.5))

    assert internal.kinds == {SHRUNK}


def test_internal_weighted_moments():
    # At t_j the record at (1, 0) weighs 1 and the other three weigh 0.25. The
    # weighted mean moves to (3 / 7, 0), about 0.43, and sigma becomes
    # sqrt(40) / 7, about 0.90. tau_skew is raised to isolate the mean.
    points = _cross(range(4))
    target_weights = {0: 1.0, 1: 0.25, 2: 0.25, 3: 0.25}

    def kinds(tau_mean: float) -> frozenset[InternalKind]:
        return _internal(
            points,
            points,
            target_weights=target_weights,
            tau_mean=tau_mean,
            tau_skew=1.0,
        ).kinds

    assert kinds(0.4) == {SHRUNK, SHIFTED}
    assert kinds(0.45) == {SHRUNK}


@pytest.mark.parametrize("n_records", [4, 400, 40_000])
def test_internal_size_thresholds_are_relative(n_records: int):
    # epsilon = 0.1 is a fraction of the cluster weight: 5% growth is no
    # change and 25% growth is an expansion, whatever the absolute difference.
    source_points = _cross(range(n_records))

    grown_5_percent = _cross(range(n_records * 105 // 100))
    grown_25_percent = _cross(range(n_records * 125 // 100))

    assert EXPANDED not in _internal(source_points, grown_5_percent).kinds
    assert EXPANDED in _internal(source_points, grown_25_percent).kinds


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_internal_feature_thresholds_are_scale_invariant(scale: float):
    # delta and tau_mean are fractions of sigma(X), so rescaling the features
    # of both clusters changes no label.
    def scaled(points: Points) -> Points:
        return {record: [scale * v for v in point] for record, point in points.items()}

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
    point = {record: [2.0, 3.0] for record in range(4)}
    moved = {record: [2.0, 3.5] for record in range(4)}
    spread = {0: [3.0, 3.0], 1: [1.0, 3.0], 2: [2.0, 4.0], 3: [2.0, 2.0]}

    assert _internal(point, point).kinds == set()
    assert _internal(point, moved).kinds == {SHIFTED}
    assert _internal(point, spread).kinds == {DIFFUSED}
    assert _internal(spread, point).kinds == {COMPACTED}


def test_internal_constant_feature_has_no_skewness():
    # 0.1 is not exactly representable, so the second feature has a variance
    # of rounding error only. Its skewness must be 0, not noise.
    source_points = {record: [float(record), 0.1] for record in range(7)}
    target_points = {record: [float(record), 0.1 * 3] for record in range(7)}

    assert _internal(source_points, target_points, tau_mean=1.0).kinds == set()


def test_internal_skipped_when_feature_dimensions_differ():
    source_points = _cross(range(4))
    target_points = {
        record: [*point, 0.0] for record, point in _cross(range(8)).items()
    }

    internal = _internal(source_points, target_points)

    assert internal == InternalTransition(0, 5, frozenset(), skipped=True)


@pytest.mark.parametrize("missing", ["source", "target", "both"])
def test_internal_skipped_without_points(missing: str):
    points = _cross(range(4))
    weights = dict.fromkeys(points, 1.0)
    source = ClusteringSnapshot(
        0, {0: set(points)}, weights, None if missing != "target" else points
    )
    target = ClusteringSnapshot(
        1, {5: set(points)}, weights, None if missing != "source" else points
    )

    internal = detect_internal_transitions(
        source, target, detect_external_transitions(source, target)
    )

    assert internal == (InternalTransition(0, 5, frozenset(), skipped=True),)


def test_internal_only_for_survived_clusters():
    # Cluster 0 survives as 5, clusters 1 and 2 are absorbed by 6, cluster 3
    # disappears and cluster 7 emerges.
    points = {record: [float(record), 0.0] for record in range(12)}
    weights = dict.fromkeys(points, 1.0)
    source = ClusteringSnapshot(
        0, {0: {0, 1}, 1: {2, 3}, 2: {4, 5}, 3: {6, 7}}, weights, points
    )
    target = ClusteringSnapshot(
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
def test_snapshot_rejects_invalid_points(points: dict, message: str):
    with pytest.raises(ValueError, match=message):
        ClusteringSnapshot(0, {0: {1, 2}}, {1: 1.0, 2: 1.0}, points)


# Lifetimes (Definition 5) and ratios (Definition 6).


def _unit_snapshot(
    timepoint: float, clusters: dict[int, set[int]]
) -> ClusteringSnapshot:
    """Snapshot whose alive records are the clustered ones, all of weight 1 and
    all at the same point, so that only a change of size is an internal
    transition."""
    records = set().union(*clusters.values())
    return ClusteringSnapshot(
        timepoint,
        clusters,
        dict.fromkeys(records, 1.0),
        {record: [0.0] for record in records},
    )


def _lifetime_history(
    clusterings: list[dict[int, set[int]]], tau: float = 0.5
) -> list[dict[int, Lifetime]]:
    """Lifetimes of the clusters of each clustering, starting from nothing."""
    history = []
    lifetimes: dict[int, Lifetime] = {}
    source = ClusteringSnapshot(-1, {}, {})
    for timepoint, clusters in enumerate(clusterings):
        target = _unit_snapshot(timepoint, clusters)
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
    source = ClusteringSnapshot(-1, {}, {})
    for timepoint, clusters in enumerate(FIVE_TIMEPOINTS):
        target = _unit_snapshot(timepoint, clusters)
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
    # Cluster 0 is older but contributes 2 records to cluster 5; cluster 1 is
    # younger and contributes 3.
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4, 5}})
    target = _unit_snapshot(1, {5: {1, 2, 3, 4, 5}})
    external = detect_external_transitions(source, target)

    lifetimes = advance_lifetimes(
        {0: Lifetime(4, 6, 7), 1: Lifetime(1, 2, 2)}, source, target, external, ()
    )

    assert lifetimes == {5: Lifetime(1, 1, 3)}


def test_absorption_tie_continues_the_longer_lived_cluster():
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4}})
    target = _unit_snapshot(1, {5: {1, 2, 3, 4}})
    external = detect_external_transitions(source, target)

    lifetimes = advance_lifetimes(
        {0: Lifetime(1, 2, 2), 1: Lifetime(4, 6, 7)}, source, target, external, ()
    )

    assert lifetimes == {5: Lifetime(1, 1, 8)}


def test_split_extends_no_lifetime():
    source = _unit_snapshot(0, {0: {1, 2, 3, 4}})
    target = _unit_snapshot(1, {5: {1, 2}, 6: {3, 4}})
    external = detect_external_transitions(source, target, tau=0.6)
    assert [t.kind for t in external] == [SPLIT]

    lifetimes = advance_lifetimes({0: Lifetime(4, 6, 7)}, source, target, external, ())

    assert lifetimes == {5: Lifetime(1, 1, 1), 6: Lifetime(1, 1, 1)}


def test_match_takes_priority_over_split_child_for_lifetime():
    # Cluster 5 is the survival of cluster 0 and a split child of cluster 1.
    source = _unit_snapshot(0, {0: {1, 2, 3, 4}, 1: {10, 11, 12, 13, 14, 15}})
    target = _unit_snapshot(1, {5: {1, 2, 3, 4, 10, 11, 12}, 6: {13, 14, 15}})
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
    source = _snapshot(0, {0: {1, 2}})
    target = _snapshot(1, {5: {1, 2}})
    external = detect_external_transitions(source, target)
    internal = detect_internal_transitions(source, target, external)
    assert internal == (InternalTransition(0, 5, frozenset(), skipped=True),)

    lifetimes = advance_lifetimes(
        {0: Lifetime(3, 3, 3)}, source, target, external, internal
    )

    assert lifetimes == {5: Lifetime(1, 4, 4)}


def test_ratios_when_everything_survives():
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4}, 2: {5}})
    target = _unit_snapshot(1, {5: {1, 2}, 6: {3, 4}, 7: {5}, 8: {9}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=1.0, absorption=0.0)
    assert ratios.passforward == 1.0


def test_ratios_when_everything_dies():
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4}, 2: {5}})
    target = _unit_snapshot(1, {5: {10, 11}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=0.0, absorption=0.0)
    assert ratios.passforward == 0.0


def test_ratios_mixed():
    # Of the four clusters, 0 survives, 1 and 2 are absorbed and 3 disappears.
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4}, 2: {5, 6}, 3: {7, 8}})
    target = _unit_snapshot(1, {5: {1, 2}, 6: {3, 4, 5, 6}, 7: {9}})

    ratios = passforward_ratios(detect_external_transitions(source, target))

    assert ratios == PassforwardRatios(survival=0.25, absorption=0.5)
    assert ratios.passforward == 0.75


def test_ratios_do_not_count_splits():
    # Cluster 0 survives and cluster 1 is split.
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4, 5, 6}})
    target = _unit_snapshot(1, {5: {1, 2}, 6: {3, 4}, 7: {5, 6}})

    ratios = passforward_ratios(detect_external_transitions(source, target, tau=0.6))

    assert ratios == PassforwardRatios(survival=0.5, absorption=0.0)


def test_ratios_of_empty_clustering_are_zero():
    target = _unit_snapshot(1, {5: {1, 2}})
    external = detect_external_transitions(ClusteringSnapshot(0, {}, {}), target)

    assert passforward_ratios(external) == PassforwardRatios(0.0, 0.0)


# MONIC over a sequence of snapshots.


def test_monic_reports_first_clustering_as_emerged():
    report = MONIC().update(_unit_snapshot(3, {0: {1, 2}, 1: {3}}))

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
        monic.update(_unit_snapshot(timepoint, clusters))
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
    monic.update(_unit_snapshot(1, {0: {1}}))

    with pytest.raises(ValueError, match="not later"):
        monic.update(_unit_snapshot(1, {0: {1}}))


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
    # Cluster 0 scatters: at t_j all of its records are in the garbage cluster
    # 9, along with the leftovers of other clusters.
    source = _unit_snapshot(0, {0: {1, 2, 3, 4}})
    target = _unit_snapshot(1, {9: {1, 2, 3, 4, 20, 21, 22, 23, 24}, 5: {30, 31}})

    assert detect_external_transitions(source, target) == (
        ExternalTransition(SURVIVED, 0, (9,)),
        ExternalTransition(EMERGED, None, (5,)),
    )
    assert detect_external_transitions(source, target.without_clusters([9])) == (
        ExternalTransition(DISAPPEARED, 0, ()),
        ExternalTransition(EMERGED, None, (5,)),
    )
    # The dissolved records are still alive, so they keep their weights.
    assert target.without_clusters([9]).weights == target.weights


# Ageing and the record buffer.


def test_sliding_window_weights():
    assert SlidingWindow(2).weight(np.array([0, 1, 2, 5])).tolist() == [1, 1, 0, 0]


def test_exponential_decay_weights():
    ageing = ExponentialDecay(rate=0.5, min_weight=0.1)

    # 2 ** -3.5 is about 0.088, which is below the minimum weight.
    assert ageing.weight(np.array([0, 2, 4, 6, 7])).tolist() == [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0,
    ]


@pytest.mark.parametrize(
    "construct",
    [
        lambda: SlidingWindow(0),
        lambda: ExponentialDecay(rate=0.0),
        lambda: ExponentialDecay(rate=1.0, min_weight=0.0),
    ],
)
def test_ageing_rejects_invalid_parameters(construct):
    with pytest.raises(ValueError):
        construct()


def test_record_buffer_ids_count_instances_and_records_age_out():
    buffer = RecordBuffer(SlidingWindow(2))

    assert [buffer.add([float(i)]) for i in range(3)] == [0, 1, 2]
    buffer.advance()
    assert [buffer.add([float(i)]) for i in range(3, 5)] == [3, 4]
    record_ids, points, weights = buffer.records()
    assert record_ids.tolist() == [0, 1, 2, 3, 4]
    assert points.tolist() == [[0.0], [1.0], [2.0], [3.0], [4.0]]
    assert weights.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]

    buffer.advance()
    assert buffer.timepoint == 2
    assert buffer.add([5.0]) == 5
    record_ids, _, weights = buffer.records()
    assert record_ids.tolist() == [3, 4, 5]
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_record_buffer_decaying_weights():
    buffer = RecordBuffer(ExponentialDecay(rate=1.0, min_weight=0.2))
    for _ in range(4):
        buffer.add([0.0])
        buffer.advance()

    record_ids, _, weights = buffer.records()

    # Ages 4, 3, 2 and 1: the two oldest are below the minimum weight.
    assert record_ids.tolist() == [2, 3]
    assert weights.tolist() == [0.25, 0.5]


def test_record_buffer_copies_points():
    buffer = RecordBuffer(SlidingWindow(1))
    point = np.array([1.0, 2.0])
    buffer.add(point)
    point[0] = 99.0

    assert buffer.records()[1].tolist() == [[1.0, 2.0]]


# Wiring to CapyMOA clusterers.


def _schema(n_features: int = 2) -> Schema:
    return NumpyStream(
        np.zeros((1, n_features)), np.zeros(1, dtype=int), target_type="categorical"
    ).get_schema()


class _FixedClusterer(Clusterer):
    """A clusterer with given centres, at the macro or at the micro level."""

    def __init__(self, centres: list[list[float]], macro: bool = True):
        super().__init__(_schema())
        self.centres = centres
        self.macro = macro

    def __str__(self):
        return "FixedClusterer"

    def train(self, instance: Instance):
        pass

    def implements_micro_clusters(self) -> bool:
        return not self.macro

    def implements_macro_clusters(self) -> bool:
        return self.macro

    def _get_micro_clusters_centers(self):
        return [] if self.macro else self.centres

    def _get_micro_clusters_radii(self):
        return [0.0] * len(self._get_micro_clusters_centers())

    def _get_micro_clusters_weights(self):
        return [1.0] * len(self._get_micro_clusters_centers())

    def _get_clusters_centers(self):
        return self.centres if self.macro else []

    def _get_clusters_radii(self):
        return [0.0] * len(self._get_clusters_centers())

    def _get_clusters_weights(self):
        return [1.0] * len(self._get_clusters_centers())

    def get_clustering_result(self):
        return ClusteringResult(
            self._get_clusters_centers(),
            self._get_clusters_weights(),
            self._get_clusters_radii(),
            [],
        )

    def get_micro_clustering_result(self):
        return ClusteringResult(
            self._get_micro_clusters_centers(),
            self._get_micro_clusters_weights(),
            self._get_micro_clusters_radii(),
            [],
        )


def _buffer(points: list[list[float]]) -> RecordBuffer:
    buffer = RecordBuffer(SlidingWindow(2))
    for point in points:
        buffer.add(point)
    return buffer


POINTS = [[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [10.0, 1.0], [5.0, 0.0], [5.0, 40.0]]


def test_adapter_assigns_records_to_nearest_centre():
    # Record 4 is equally far from both centres and goes to the first.
    snapshot = snapshot_from_clusterer(
        _FixedClusterer([[0.0, 0.0], [10.0, 0.0]]), _buffer(POINTS)
    )

    assert snapshot.timepoint == 0
    assert snapshot.clusters == {0: {0, 1, 4, 5}, 1: {2, 3}}
    assert snapshot.weights == dict.fromkeys(range(6), 1.0)
    assert {r: point.tolist() for r, point in snapshot.points.items()} == dict(
        enumerate(POINTS)
    )


def test_adapter_leaves_distant_records_unassigned():
    # Record 5 is 40.3 from the nearest centre. It is noise: alive, unassigned.
    snapshot = snapshot_from_clusterer(
        _FixedClusterer([[0.0, 0.0], [10.0, 0.0]]), _buffer(POINTS), max_distance=5.0
    )

    assert snapshot.clusters == {0: {0, 1, 4}, 1: {2, 3}}
    assert snapshot.weights == dict.fromkeys(range(6), 1.0)


def test_adapter_omits_clusters_without_records():
    snapshot = snapshot_from_clusterer(
        _FixedClusterer([[0.0, 0.0], [500.0, 500.0], [10.0, 0.0]]), _buffer(POINTS)
    )

    assert snapshot.clusters == {0: {0, 1, 4, 5}, 2: {2, 3}}


def test_adapter_without_clusters_keeps_records_alive():
    # A clusterer that has not produced clusters yet.
    snapshot = snapshot_from_clusterer(_FixedClusterer([]), _buffer(POINTS))

    assert snapshot.clusters == {}
    assert snapshot.weights == dict.fromkeys(range(6), 1.0)


def test_adapter_without_records():
    snapshot = snapshot_from_clusterer(_FixedClusterer([[0.0, 0.0]]), _buffer([]))

    assert snapshot.clusters == {}
    assert snapshot.weights == {}


@pytest.mark.parametrize(
    ("macro", "level", "expected"),
    [
        (True, "auto", {0: {0, 1, 2, 3, 4, 5}}),
        (False, "auto", {0: {0, 1, 2, 3, 4, 5}}),
        (True, "macro", {0: {0, 1, 2, 3, 4, 5}}),
        (False, "micro", {0: {0, 1, 2, 3, 4, 5}}),
        (True, "micro", None),
        (False, "macro", None),
    ],
)
def test_adapter_level(macro: bool, level: str, expected: dict | None):
    clusterer = _FixedClusterer([[0.0, 0.0]], macro=macro)

    if expected is None:
        with pytest.raises(ValueError, match=f"does not implement {level}-clusters"):
            snapshot_from_clusterer(clusterer, _buffer(POINTS), level=level)
    else:
        snapshot = snapshot_from_clusterer(clusterer, _buffer(POINTS), level=level)
        assert snapshot.clusters == expected


def test_adapter_rejects_centres_of_another_dimension():
    with pytest.raises(ValueError, match="records have 2 features"):
        snapshot_from_clusterer(_FixedClusterer([[0.0, 0.0, 0.0]]), _buffer(POINTS))


class _LeaderClusterer(_FixedClusterer):
    """A deterministic clusterer that re-clusters its last ``window`` instances.

    An instance joins the first cluster whose first instance is within
    ``radius`` of it, and otherwise starts a cluster. The centres are the means.
    """

    def __init__(self, window: int, radius: float):
        super().__init__([])
        self.radius = radius
        self.recent: deque[np.ndarray] = deque(maxlen=window)

    def train(self, instance: Instance):
        self.recent.append(instance.x.copy())

    def _get_clusters_centers(self):
        groups: list[list[np.ndarray]] = []
        for point in self.recent:
            for group in groups:
                if np.linalg.norm(point - group[0]) <= self.radius:
                    group.append(point)
                    break
            else:
                groups.append([point])
        return [np.mean(group, axis=0).tolist() for group in groups]


# Where the clusters of the drifting stream are. B is replaced by B1 and B2 at
# instance 3000, and C receives its last instance just before instance 5000.
CENTRES = {
    "A": (0.0, 0.0),
    "B": (10.0, 0.0),
    "B1": (10.0, 4.0),
    "B2": (10.0, -4.0),
    "C": (0.0, 10.0),
}
OFFSETS = [(0.5, 0.25), (-0.5, -0.25), (0.25, -0.5), (-0.25, 0.5)]
SPLIT_AT = 3000
DEATH_AT = 5000
N_INSTANCES = 8000
SNAPSHOT_INTERVAL = 1000


def _drifting_stream() -> NumpyStream:
    """Clusters taking turns, each instance at its centre plus a small offset.

    Half the offsets are above the centre and half below, so the instances of B
    divide evenly between B1 and B2.
    """
    points = np.empty((N_INSTANCES, 2))
    for i in range(N_INSTANCES):
        if i < SPLIT_AT:
            active = ["A", "B", "C"]
        elif i < DEATH_AT:
            active = ["A", "B1", "B2", "C"]
        else:
            active = ["A", "B1", "B2"]
        centre = CENTRES[active[i % len(active)]]
        offset = OFFSETS[(i // len(active)) % len(OFFSETS)]
        points[i] = (centre[0] + offset[0], centre[1] + offset[1])
    return NumpyStream(
        points, np.zeros(N_INSTANCES, dtype=int), target_type="categorical"
    )


def _named_transitions(max_distance: float | None) -> list[list[tuple]]:
    """External transitions at each timepoint of the drifting stream, with
    clusters named after the centre they are at."""
    names = {tuple(np.round(centre)): name for name, centre in CENTRES.items()}
    clusterer = _LeaderClusterer(window=SNAPSHOT_INTERVAL, radius=2.0)
    monitor = MONICMonitor(
        clusterer,
        snapshot_interval=SNAPSHOT_INTERVAL,
        max_distance=max_distance,
        tau=0.6,
        tau_split=0.2,
        debug=True,
    )

    history = []
    source_names: dict[int, str] = {}
    for report in monitor.monitor(_drifting_stream()):
        target_names = {
            cluster_id: names[tuple(np.round(centre))]
            for cluster_id, centre in enumerate(clusterer._get_clusters_centers())
        }
        history.append(
            sorted(
                (
                    t.kind.name,
                    source_names.get(t.source),
                    tuple(sorted(target_names[target] for target in t.targets)),
                )
                for t in report.external
            )
        )
        source_names = target_names
    assert monitor.instances_seen == N_INSTANCES
    return history


def test_monitor_detects_split_and_death_on_drifting_stream():
    history = _named_transitions(max_distance=6.0)

    stable = [
        ("SURVIVED", "A", ("A",)),
        ("SURVIVED", "B", ("B",)),
        ("SURVIVED", "C", ("C",)),
    ]
    after_death = [
        ("SURVIVED", "A", ("A",)),
        ("SURVIVED", "B1", ("B1",)),
        ("SURVIVED", "B2", ("B2",)),
    ]
    assert history == [
        # Timepoint 0, after 1000 instances.
        [
            ("EMERGED", None, ("A",)),
            ("EMERGED", None, ("B",)),
            ("EMERGED", None, ("C",)),
        ],
        stable,
        stable,
        # Timepoint 3, after 4000 instances: the first since the split at 3000.
        [
            ("SPLIT", "B", ("B1", "B2")),
            ("SURVIVED", "A", ("A",)),
            ("SURVIVED", "C", ("C",)),
        ],
        [
            ("SURVIVED", "A", ("A",)),
            ("SURVIVED", "B1", ("B1",)),
            ("SURVIVED", "B2", ("B2",)),
            ("SURVIVED", "C", ("C",)),
        ],
        # Timepoint 5, after 6000 instances: the first with no instance of C.
        [("DISAPPEARED", "C", ()), *after_death],
        after_death,
        after_death,
    ]


def test_monitor_without_max_distance_reports_death_as_absorption():
    # The leftover instances of C are assigned to A, the nearest centre left,
    # so A is the match of both A and C.
    history = _named_transitions(max_distance=None)

    assert history[5] == [
        ("ABSORBED", "A", ("A",)),
        ("ABSORBED", "C", ("A",)),
        ("SURVIVED", "B1", ("B1",)),
        ("SURVIVED", "B2", ("B2",)),
    ]
    assert history[3] == [
        ("SPLIT", "B", ("B1", "B2")),
        ("SURVIVED", "A", ("A",)),
        ("SURVIVED", "C", ("C",)),
    ]


def test_monitor_trains_the_clusterer_and_reports_at_intervals():
    clusterer = _LeaderClusterer(window=10, radius=2.0)
    monitor = MONICMonitor(clusterer, snapshot_interval=4)
    stream = _drifting_stream()

    reports = [monitor.train(stream.next_instance()) for _ in range(9)]

    assert [report is not None for report in reports] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert len(clusterer.recent) == 9
    assert [reports[3].target_timepoint, reports[7].target_timepoint] == [0, 1]
    assert monitor.snapshot.timepoint == 1
    # Instances 0..7 are alive at timepoint 1 under the default window of 2.
    assert sorted(monitor.snapshot.weights) == list(range(8))


def test_monitor_stops_at_max_instances():
    monitor = MONICMonitor(_LeaderClusterer(window=10, radius=2.0), snapshot_interval=4)

    reports = list(monitor.monitor(_drifting_stream(), max_instances=10))

    assert len(reports) == 2
    assert monitor.instances_seen == 10


@pytest.mark.parametrize(
    "kwargs", [{"snapshot_interval": 0}, {"max_distance": -1.0}, {"tau_split": 0.9}]
)
def test_monitor_validates_arguments(kwargs: dict):
    with pytest.raises(ValueError):
        MONICMonitor(_FixedClusterer([]), **kwargs)


def test_monitor_detects_split_and_death_with_denstream():
    # A MOA clusterer set to forget within one snapshot interval, so that its
    # clusters follow the data from one timepoint to the next.
    stream = _drifting_stream()
    clusterer = Denstream_with_dbscan(
        schema=stream.get_schema(),
        epsilon=0.5,
        init_points=200,
        lambda_option=1.0,
        speed=10,
    )
    monitor = MONICMonitor(
        clusterer,
        snapshot_interval=SNAPSHOT_INTERVAL,
        max_distance=6.0,
        tau=0.6,
        tau_split=0.2,
        debug=True,
    )

    kinds = [
        sorted(t.kind.name for t in report.external)
        for report in monitor.monitor(stream)
    ]

    assert kinds == [
        ["EMERGED", "EMERGED", "EMERGED"],
        ["SURVIVED", "SURVIVED", "SURVIVED"],
        ["SURVIVED", "SURVIVED", "SURVIVED"],
        # Timepoint 3: the first since the split at instance 3000.
        ["SPLIT", "SURVIVED", "SURVIVED"],
        ["SURVIVED", "SURVIVED", "SURVIVED", "SURVIVED"],
        # Timepoint 5: the first with no instance of C.
        ["DISAPPEARED", "SURVIVED", "SURVIVED", "SURVIVED"],
        ["SURVIVED", "SURVIVED", "SURVIVED"],
        ["SURVIVED", "SURVIVED", "SURVIVED"],
    ]


# The runtime invariant checks of debug mode.


def _report_for_invariants():
    """A report with a survival, a split, two disappearances and an emergence.
    Record 10 of cluster 2 is alive at t_j but unassigned, and cluster 3 has
    aged out."""
    monic = MONIC(tau=0.6, debug=True)
    source = _unit_snapshot(0, {0: {1, 2}, 1: {3, 4, 5, 6}, 2: {10, 11}, 3: {30}})
    target = ClusteringSnapshot(
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
        monic.update(_unit_snapshot(timepoint, clusters))


def _corrupt_row_sum_above_one(report):
    overlap = report.overlap.copy()
    overlap[0, 1] = 0.5
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_row_sum_below_one(report):
    overlap = report.overlap.copy()
    overlap[1, 2] = 0.25
    return dataclasses.replace(report, overlap=overlap)


def _corrupt_row_sum_with_unassigned_record(report):
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
            _corrupt_row_sum_with_unassigned_record,
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


def test_debug_mode_row_sum_may_be_below_one_with_unassigned_records():
    # Record 4 is alive at t_j but in no cluster, so the row sums to 0.75.
    monic = MONIC(debug=True)
    monic.update(_unit_snapshot(0, {0: {1, 2, 3, 4}}))

    report = monic.update(
        ClusteringSnapshot(1, {5: {1, 2, 3}}, dict.fromkeys([1, 2, 3, 4], 1.0))
    )

    assert report.overlap.tolist() == [[0.75]]
