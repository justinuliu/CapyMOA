import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from capymoa.evaluation.monic import (
    MONIC,
    Clustering,
    ExternalTransition,
    TransitionType,
    _overlap_weights,
    overlap,
)

WEIGHTS = [0.0, 0.25, 0.5, 1.0]


def _external(
    source: Clustering, target: Clustering, tau: float, tau_split: float
) -> tuple[ExternalTransition, ...]:
    """External transitions from ``source`` to ``target``, as detected by a MONIC
    with the given thresholds."""
    monic = MONIC(tau=tau, tau_split=tau_split)
    return monic._detect_external_transitions(
        source, target, _overlap_weights(source, target)
    )


@st.composite
def clustering_pairs(draw: st.DrawFn) -> tuple[Clustering, Clustering]:
    """Two clusterings over a shared pool of instances.

    Each instance is independently placed in a cluster of each clustering or left
    out of it, and given a weight at the later timepoint. A weight of 0 or being
    left out of both the clusters and the weights models ageing out.
    """
    n_instances = draw(st.integers(0, 24))
    cluster = st.none() | st.integers(0, 4)

    source_clusters: dict[int, set[int]] = {}
    target_clusters: dict[int, set[int]] = {}
    target_weights: dict[int, float] = {}
    for instance in range(n_instances):
        source_cluster = draw(cluster)
        if source_cluster is not None:
            source_clusters.setdefault(source_cluster, set()).add(instance)
        target_cluster = draw(cluster)
        if target_cluster is not None:
            target_clusters.setdefault(target_cluster, set()).add(instance)
        if target_cluster is not None or draw(st.booleans()):
            target_weights[instance] = draw(st.sampled_from(WEIGHTS))

    source_weights = dict.fromkeys(range(n_instances), 1.0)
    return (
        Clustering(0, source_clusters, source_weights),
        Clustering(1, target_clusters, target_weights),
    )


@settings(max_examples=500, deadline=None, derandomize=True, database=None)
@given(
    pair=clustering_pairs(),
    tau=st.sampled_from([0.5, 0.6, 0.75, 1.0]),
    tau_split=st.sampled_from([0.1, 0.25, 0.4]),
)
def test_external_transitions_label_every_cluster_once(
    pair: tuple[Clustering, Clustering], tau: float, tau_split: float
):
    source, target = pair
    transitions = _external(source, target, tau, tau_split)
    external = [t for t in transitions if t.kind is not TransitionType.EMERGED]
    emerged = [t for t in transitions if t.kind is TransitionType.EMERGED]

    # Every cluster of zeta_i receives exactly one label.
    assert [t.source for t in external] == list(source.cluster_ids)

    # Every cluster of zeta_j is either emerged or the result of a transition.
    produced = {target_id for t in external for target_id in t.targets}
    emerged_ids = [t.targets[0] for t in emerged]
    assert all(t.source is None and len(t.targets) == 1 for t in emerged)
    assert len(emerged_ids) == len(set(emerged_ids))
    assert produced.isdisjoint(emerged_ids)
    assert produced | set(emerged_ids) == set(target.cluster_ids)

    # Each label satisfies its indicator in Table 1.
    def overlaps_of(source_id: int) -> dict[int, float]:
        return {
            target_id: overlap(source.clusters[source_id], instances, target.weights)
            for target_id, instances in target.clusters.items()
        }

    n_matched_by: dict[int, int] = {}
    for t in external:
        if t.kind in (TransitionType.SURVIVED, TransitionType.ABSORBED):
            n_matched_by[t.targets[0]] = n_matched_by.get(t.targets[0], 0) + 1

    for t in external:
        overlaps = overlaps_of(t.source)
        best = max(overlaps.values(), default=0.0)
        if t.kind in (TransitionType.SURVIVED, TransitionType.ABSORBED):
            (match,) = t.targets
            assert overlaps[match] == best >= tau
            assert (n_matched_by[match] > 1) == (t.kind is TransitionType.ABSORBED)
        else:
            assert best < tau
            candidates = {y for y, value in overlaps.items() if value >= tau_split}
            if t.kind is TransitionType.SPLIT:
                assert set(t.targets) == candidates
                assert len(t.targets) >= 2
            else:
                assert t.targets == ()


@st.composite
def histories(draw: st.DrawFn) -> list[Clustering]:
    """Clusterings at successive timepoints over a shared pool of instances.

    At each timepoint an instance is either clustered, with a weight and a point,
    or has aged out.
    """
    n_instances = draw(st.integers(1, 16))
    clusterings = []
    for timepoint in range(draw(st.integers(2, 6))):
        clusters: dict[int, set[int]] = {}
        weights: dict[int, float] = {}
        points: dict[int, list[float]] = {}
        for instance in range(n_instances):
            cluster = draw(st.none() | st.integers(0, 3))
            if cluster is not None:
                clusters.setdefault(cluster, set()).add(instance)
                weights[instance] = draw(st.sampled_from([0.5, 1.0]))
                points[instance] = [float(draw(st.integers(0, 2)))]
        clusterings.append(Clustering(timepoint, clusters, weights, points))
    return clusterings


@settings(max_examples=300, deadline=None, derandomize=True, database=None)
@given(clusterings=histories(), tau=st.sampled_from([0.5, 0.6, 0.75]))
def test_lifetimes_are_ordered(clusterings: list[Clustering], tau: float):
    monic = MONIC(tau=tau, tau_split=0.1)
    for n_timepoints, target in enumerate(clusterings, start=1):
        lifetimes = monic.update(target).lifetimes

        assert lifetimes.keys() == target.clusters.keys()
        for lifetime in lifetimes.values():
            assert 1 <= lifetime.strict <= lifetime.internal <= lifetime.absorption
            assert lifetime.absorption <= n_timepoints
