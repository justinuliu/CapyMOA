# MONIC design

Status: design, pre-implementation. Companion to
`docs/MONIC-implementation-notes.md`. Notation follows the paper: `X` is a
cluster of clustering `ζ_i` at `t_i`, `Y` a cluster of `ζ_j` at `t_j`,
`t_j > t_i`.

## 0. Architecture in one paragraph

MONIC is split into two layers. The **core** (`MONIC`) consumes
`ClusteringSnapshot` objects, which are explicit record-level partitions, and
is purely set-theoretic: it never sees a clusterer. The **adapter**
(`SnapshotBuilder`) turns a CapyMOA `Clusterer` plus a buffer of recent
records into a snapshot. Users who re-cluster each timepoint themselves, as
the paper does, skip the adapter and build snapshots directly.

## 1. Membership

**Paper.** Def. 3 defines overlap over `X ∩ Y`, a set of records. Section 6
says that with summary data only, "splits and absorptions cannot be traced".

**Options.** (a) MONIC-side buffer of records, assigned to clusters at each
timepoint. (b) Summary-only overlap.

**Decision: (a).** Option (b) cannot produce three of the five external
transitions and contradicts the non-negotiable in `CLAUDE.md`.

**Buffer.** `SnapshotBuilder` stores, per record: an integer record ID, the
feature vector, and the birth timestamp. IDs are sequential arrival indices
unless the caller supplies them. Records whose weight has reached zero are
dropped (section 2), which bounds memory by the ageing horizon.

**Assignment rule.** At a timepoint, each buffered record with non-zero
weight is assigned to the cluster whose centre is nearest in Euclidean
distance. Ties go to the lower cluster index. **Radii are ignored.**

- Why no radius test: in the recon probe, 183 to 299 of the last 300 points
  lay inside no cluster radius for all four clusterers. A radius rule would
  discard most of the data and leave MONIC comparing near-empty sets.
- Centres come from `get_clustering_result()` when
  `implements_macro_clusters()` is true, otherwise from
  `get_micro_clustering_result()`. A `level` argument (`"auto"`, `"macro"`,
  `"micro"`) overrides this. Micro-cluster monitoring is allowed but will be
  noisy (100 to 335 clusters in the probe).
- Clusters that receive no records are omitted from the snapshot.

**Def. 1 compliance.** Nearest-centre assignment over the live buffer is a
Voronoi partition: clusters are disjoint, and their union is the live
buffer. The dataset `D` at `t` is defined as the live buffer at `t`. There
are no outliers by construction.

**Noise in the adapter.** The adapter never creates a cluster for records
that fit nowhere. With `max_distance` set, a record farther than that from
every centre is noise: alive but unassigned. With the default `None` there
is no noise, and the records left behind by a cluster that has died out are
handed to its nearest neighbour, which is then reported as having absorbed
it. The distance is absolute and supplied by the caller, because the radii
the clusterers report cannot be used: on well-separated blobs
`Denstream_with_dbscan` macro-clusters and `ClusTree` micro-clusters report
a radius of 0.

**Garbage clusters in user-built snapshots.**
`ClusteringSnapshot.without_clusters(ids)` dissolves a garbage cluster into
noise records.

**Noise in user-built snapshots.** A noise record is one that has a weight
in the snapshot but belongs to no cluster. Leaving it out of every cluster
is the preprocessing step Def. 1's commentary assumes. A record of `X` that is alive at `t_j` but
unassigned still counts in the overlap denominator, so the row sum is below
1 in that case. This is exactly the condition in the `CLAUDE.md` invariant
("any X whose surviving records are all assigned").

**If we picked (b):** no buffer and no memory cost, but only disappearance
and a weak form of survival could be reported, and Def. 3 would be replaced
by a geometric proxy the paper does not define.

**Known limitation.** The clusterer ages its own summaries with its own
horizon; MONIC ages the buffer with its ageing function. The two are not
synchronised. The document will say so.

## 2. Ageing

**Paper.** Def. 2: `age(x, t_i) ∈ [0, 1]` for every record and timepoint.

**Interface.**

```python
class AgeingFunction(ABC):
    @abstractmethod
    def weight(self, elapsed: np.ndarray) -> np.ndarray:
        """Weights in [0, 1] for records that arrived `elapsed` time ago."""

    @property
    @abstractmethod
    def horizon(self) -> float:
        """Smallest elapsed time at which the weight is exactly zero (inf if never)."""
```

- `SlidingWindow(size)`: weight 1 when `elapsed < size`, else 0. The paper's
  "sliding window of size 2" is `SlidingWindow(2)`: records from `t_{i-1}`
  and `t_i` are live at `t_i`.
- `ExponentialDecay(rate, min_weight=1e-3)`: weight `2 ** (-rate * elapsed)`,
  set to exactly 0 below `min_weight` so the horizon is finite and the
  buffer bounded.

**Time unit.** `elapsed = t - birth`, in whatever unit the snapshot
timestamps use. The adapter uses the timepoint index (0, 1, 2, ...) and
stamps a record with the timepoint at which it is first clustered, matching
the paper's `D_i`.

**Zero-weight records are not cluster members.** They are dropped on
ingest. For Def. 3 this changes nothing, because a zero weight contributes
zero to both sums. It matters in two places only: a cluster whose records
all have zero weight does not exist, and unweighted counts are never
reported. If we kept them as members, results would be identical except
that empty-weight clusters would appear in `ζ` and immediately disappear.

## 3. Weight convention

**Paper.** Def. 3 weights both sums by `age(·, t_j)`. Table 2 weights `X` by
`age(·, t_i)` and `Y` by `age(·, t_j)`.

**Decision: follow the paper for each purpose.**

| Purpose | Weights for `X` | Weights for `Y` | Attribute | Value |
|---|---|---|---|---|
| Overlap, matching, split test | `t_j` | `t_j` | `overlap_weighting` | `"target"` |
| Size, `μ`, `σ`, `γ` | `t_i` | `t_j` | `statistic_weighting` | `"own"` |

Both are read-only attributes on `MONIC` and are copied into every
`TransitionReport`, so stored results describe themselves. v1 implements
only these values.

- Why `"target"` for overlap: it is the published definition, and it is what
  makes the row-sum invariant hold.
- Why `"own"` for statistics: under a sliding window a stationary cluster
  keeps its size. With `"target"`, `X` would always appear to have lost its
  older half and every survivor would be reported as expanding.

**Consequence for the snapshot type.** A snapshot carries the ageing weights
in force at its own timepoint, for every live record, clustered or not.
`X`'s `t_j` weights are then read from the `t_j` snapshot, and a record
missing from it has weight 0 (aged out). The core never calls an ageing
function; only `SnapshotBuilder` does.

**Zero denominator.** If every record of `X` has zero weight at `t_j`,
`overlap(X, ·) = 0.0` and `X` is reported as disappeared. Never raises,
never NaN.

## 4. Matching and tie-breaking

**Thresholds.** Construction asserts `0 < tau_split < tau <= 1`.

- `tau_split > 0` is an addition: at zero, every `Y` would be a split
  candidate, including those sharing no records with `X`.
- `tau < 0.5` is accepted with a warning. Def. 4 restricts `τ` to
  `[0.5, 1]`, but section 5.2 runs `τ = 0.45`, so rejecting it would make
  the paper's own experiments unreproducible.

**Paper on ties.** "Different tie breakers can be used, choosing e.g. the Y
that has the maximum reverse overlap `overlap(Y, X)` or the Y that is
closest to X in size."

**Decision.** `g(X, Y)` is the lexicographic key

```text
g(X, Y)    = (overlap(X, Y), overlap(Y, X), -index(Y))
g(X, NULL) = (-inf, -inf, -inf)
```

compared with exact equality, no tolerance. `index(Y)` is the cluster's
position in the snapshot, which makes the result deterministic.

- The second and third components fire only when two clusters have equal
  forward overlap of at least `τ`. Overlaps sum to at most 1, so this
  requires `τ <= 0.5`. Inside the paper's range it happens only at exactly
  `τ = 0.5` with a 0.5 / 0.5 division.
- Given equal forward overlap, the larger reverse overlap is the smaller
  `Y`, so the tighter fit wins.
- If we used "closest in size" instead, the winner would differ whenever the
  larger candidate is nearer to `X` in size. Example: `W(X) = 10` with
  candidates of weight 6 and 11; reverse overlap picks the 6, closest size
  picks the 11. Closest size also needs a size convention (section 3),
  which reverse overlap avoids.

**Control flow.** The corrected flow from the notes, unchanged. The split
test sums intersection weights and divides once, to avoid float error at
the `τ` boundary. `splitUnion` is never built.

**Overlap matrix.** Built with an index from record ID to cluster at `t_j`
and one pass over each `X`: `O(|D|)`.

**Emerged clusters.** A `Y` is emerged when it is neither the target of a
survival or absorption nor a member of a successful split. A `Y` that
overlapped an `X` whose split test failed therefore counts as emerged,
following the paper's text. The report carries the overlap matrix so this
case can be inspected.

## 5. Internal transition statistics

All moments are weighted, with the `"own"` convention of section 3.
`w_a` is the record weight and `W = Σ w_a`.

| Symbol | Definition | Type |
|---|---|---|
| `μ(C)` | `Σ w_a x_a / W` | vector |
| `σ(C)` | `sqrt(Σ w_a ‖x_a − μ‖² / W)`, the RMS distance to the centroid | scalar |
| `γ(C)` | per-dimension weighted skewness; 0 for a zero-variance dimension | vector |

- Why this `σ`: it is the square root of the trace of the covariance, so it
  is a single number, rotation-invariant, and cheap. The alternative, the
  mean of per-dimension standard deviations, changes under rotation of the
  feature space and weighs low-variance dimensions equally.
- Why per-dimension `γ`: it is the direct generalisation of the univariate
  statistic the paper names. The alternative, skewness of the
  distance-to-centroid distribution, is a scalar but cannot see a shift
  along one feature.

**Thresholds are relative**, a deviation from the paper, which uses absolute
differences.

| Transition | Paper | This design | Default |
|---|---|---|---|
| shrinks | `W(X) > W(Y) + ε` | `W(Y) < (1 − ε) · W(X)` | `epsilon = 0.1` |
| expands | `W(Y) > W(X) + ε` | `W(Y) > (1 + ε) · W(X)` | |
| more compact | `σ(Y) < σ(X) − δ` | `σ(Y) < (1 − δ) · σ(X)` | `delta = 0.1` |
| more diffuse | `σ(Y) > σ(X) + δ` | `σ(Y) > (1 + δ) · σ(X)` | |
| mean shift (I1) | `|μ(X) − μ(Y)| > τ1` | `‖μ(X) − μ(Y)‖₂ > τ1 · σ(X)` | `tau_mean = 0.5` |
| skew shift (I2) | `|γ(X) − γ(Y)| > τ2` | `‖γ(X) − γ(Y)‖∞ > τ2` | `tau_skew = 0.5` |

- Why relative: absolute `ε` and `δ` depend on stream volume and feature
  scale, so no default could be meaningful. `τ2` stays absolute because
  skewness is already dimensionless.
- **The four default values are not from the paper.** It gives none. They
  are placeholders and the document will say so.
- If we kept absolute thresholds, results would match the paper's formulae
  literally, but every user would have to tune four values per dataset.
- Degenerate case `σ(X) = 0`: more diffuse iff `σ(Y) > 0`, never more
  compact, mean shift iff the centroids differ.

**Scope of internal analysis.**

- Only survived clusters (`X → Y`) are analysed, per section 4.2. Absorbed
  clusters get none.
- `X ↔ Y` (no change) is a survival whose internal set is empty. It is not
  a separate label.
- The three groups (size, compactness, location) are independent and
  combine. Each yields at most one label, so the location group has a single
  label, `SHIFTED`, raised by either indicator I1 or I2.
- `μ`, `σ` and `γ` are undefined across a changed feature space. If the two
  snapshots do not both carry points with the same number of features, the
  whole internal analysis of the pair is skipped, size included, and the
  transition is marked `skipped = True` with an empty label set. A change of
  feature space that keeps the number of features cannot be detected.

## 6. Lifetime

**Paper.** Def. 5 counts timepoints a cluster has survived since it
emerged. Section 4.3 computes the same quantity backward from the final
clustering `ζ_n`.

**Decision: forward, per Def. 5.** A stream has no `ζ_n`, and forward values
are final as soon as they are computed. Every cluster in the current
clustering carries three counters:

| Event producing `Y` | `lifetimeS(Y)` | `lifetimeI(Y)` | `lifetimeA(Y)` |
|---|---|---|---|
| emerged, or child of a split | 1 | 1 | 1 |
| `X → Y`, no internal transition | `S(X) + 1` | `I(X) + 1` | `A(X) + 1` |
| `X → Y`, with internal transition | 1 | `I(X) + 1` | `A(X) + 1` |
| absorption, dominant parent `X*` | 1 | 1 | `A(X*) + 1` |

- `S <= I <= A` holds by induction on this table and is asserted.
- A survival whose internal analysis was skipped is treated as having an
  internal transition, so `lifetimeS` is never overstated.
- For a chain, the forward value on the last cluster equals the paper's
  backward value on the first.

**What the API returns with no final timepoint.** `MONIC.clusters` gives the
lifetime so far of every live cluster. When a lineage ends, its final
lifetimes are recorded on the transition that ended it.

**If we picked backward:** values would attach to past clusters, change
every time a new timepoint arrives, and need the full history. It remains
derivable from the stored reports and is deferred.

**Ratios (Def. 6).** Each report gives `survival_ratio`,
`absorption_ratio` and `passforward_ratio`, divided by `|ζ_i|`, and 0.0
when `ζ_i` is empty. Table 3's absolute counts are not reproduced as
ratios.

## 7. Lineage IDs

**Paper.** Footnote 3: cluster identifiers "do not indicate transitions".
Section 5.4 nevertheless follows chains across years. Propagation is not
defined, so it is defined here.

Lineage IDs are integers allocated in increasing order and never reused.

| Transition | Lineage of `Y` | Lineage of `X` |
|---|---|---|
| survive `X → Y` | inherits `X`'s | continues |
| absorb `{X_1..X_k} ⊂→ Y` | inherits the dominant parent's | others end, recording the absorbing lineage |
| split `X ⊂→ {Y_1..Y_p}` | every child gets a fresh ID, recording `X` as parent | ends |
| disappear | none | ends |
| emerge | fresh ID, no parent | none |

- **Dominant parent:** the absorbed `X` with the largest intersection weight
  with `Y` at `t_j`. Ties go to the larger `lifetimeA`, then the lower
  lineage ID.
- **Why absorption continues a lineage:** `lifetimeA` counts absorptions, so
  some lineage must carry through. If `Y` took a fresh ID, `lifetimeA` would
  always equal `lifetimeI`.
- **Why split children are all fresh:** splits extend no lifetime in the
  paper, and its own example chain ends at the split. If the largest child
  inherited instead, a split would be indistinguishable from a survival
  with shrinkage in the lineage record.
- **A `Y` with several roles:** being a survival or absorption target takes
  precedence over being a split child. A `Y` that is a split child of two
  different parents gets one fresh ID and records both parents.

## 8. Scope

**Ships in v1**

- `ClusteringSnapshot`, `AgeingFunction`, `SlidingWindow`, `ExponentialDecay`.
- External transitions: survive, absorb, split, disappear, emerge.
- Internal transitions for survivors (section 5).
- Forward lifetimes, lineage IDs, Def. 6 ratios.
- `SnapshotBuilder` with nearest-centre assignment.
- Tests: hand-built fixtures with known answers, the row-sum invariant at
  tolerance 1e-9, `S <= I <= A`, threshold assertions, zero-denominator
  behaviour. No randomness.

**Deferred**

- Summary-only overlap (option b).
- Matching across non-consecutive timepoints (`j > i + 1`).
- Backward ("remaining") lifetime.
- Assignment rules other than nearest centre with an absolute distance limit.
- Internal transitions across a changed feature space.
- Internal analysis of absorbed clusters.
- Significance tests in place of `δ`.
- Alternative weighting conventions as constructor options.
- Non-integer record IDs.
- Plotting, a tutorial notebook, and any change to `Clusterer`,
  `ClusteringResult` or `ClusteringEvaluator`.

## 9. Public API sketch

Module: `capymoa.evaluation.monic`. Dependencies: numpy only.

```python
@dataclass(frozen=True)
class ClusteringSnapshot:
    timepoint: float
    clusters: Mapping[int, frozenset[int]]   # cluster id -> record ids
    weights: Mapping[int, float]             # record id -> age(record, timepoint)
    # Feature vectors for sigma/mu/gamma are added with internal transitions.

def overlap(x: Set[int], y: Set[int], weights: Mapping[int, float]) -> float: ...
def build_overlap_matrix(
    source: ClusteringSnapshot, target: ClusteringSnapshot
) -> np.ndarray: ...                         # shape (|zeta_i|, |zeta_j|)

class MONIC:
    overlap_weighting: Literal["target"] = "target"
    statistic_weighting: Literal["own"] = "own"

    def __init__(
        self,
        tau: float = 0.5,
        tau_split: float = 0.1,
        epsilon: float = 0.1,
        delta: float = 0.1,
        tau_mean: float = 0.5,
        tau_skew: float = 0.5,
    ) -> None: ...

    def update(self, snapshot: ClusteringSnapshot) -> TransitionReport: ...
    @property
    def clusters(self) -> list[ClusterState]: ...      # live clusters
    @property
    def history(self) -> list[TransitionReport]: ...

class RecordBuffer:                        # ids count the instances added
    def __init__(self, ageing: AgeingFunction) -> None: ...
    def add(self, point: Sequence[float]) -> int: ...
    def advance(self) -> None: ...
    def records(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def snapshot_from_clusterer(
    clusterer: Clusterer, buffer: RecordBuffer,
    level: str = "auto", max_distance: float | None = None,
) -> ClusteringSnapshot: ...

class MONICMonitor:                        # trains the clusterer, buffers, reports
    def __init__(self, clusterer: Clusterer, snapshot_interval: int = 1000,
                 ageing: AgeingFunction | None = None, level: str = "auto",
                 max_distance: float | None = None, **thresholds) -> None: ...
    def train(self, instance: Instance) -> TransitionReport | None: ...
    def monitor(self, stream: Stream, max_instances: int | None = None
                ) -> Iterator[TransitionReport]: ...

class ExternalKind(Enum):   SURVIVED, ABSORBED, SPLIT, DISAPPEARED, EMERGED
class InternalKind(Enum):   SHRUNK, EXPANDED, COMPACTED, DIFFUSED, SHIFTED

@dataclass(frozen=True)
class ClusterState:
    label: int                 # per-timepoint, meaningless across time
    lineage: int
    lifetime_strict: int
    lifetime_internal: int
    lifetime_absorption: int

@dataclass(frozen=True)
class Transition:
    kind: ExternalKind
    source: ClusterState | None            # None for EMERGED
    targets: tuple[ClusterState, ...]      # empty for DISAPPEARED
    internal: frozenset[InternalKind]      # empty unless SURVIVED
    internal_skipped: bool

@dataclass(frozen=True)
class TransitionReport:
    source_timestamp: float
    target_timestamp: float
    overlap: np.ndarray                    # shape (|zeta_i|, |zeta_j|)
    transitions: tuple[Transition, ...]    # one per X, plus one per emerged Y
    survival_ratio: float
    absorption_ratio: float
    passforward_ratio: float
    overlap_weighting: str
    statistic_weighting: str
```

The first `update` call has no predecessor: it reports every cluster as
emerged with an empty overlap matrix.

**Usage**

```python
from capymoa.cluster import Denstream_with_dbscan
from capymoa.evaluation.monic import MONICMonitor, SlidingWindow
from capymoa.stream.generator import RandomRBFGenerator

stream = RandomRBFGenerator()
clusterer = Denstream_with_dbscan(schema=stream.get_schema())
monitor = MONICMonitor(
    clusterer, snapshot_interval=1000, ageing=SlidingWindow(2), tau=0.5, tau_split=0.1
)

for report in monitor.monitor(stream, max_instances=10_000):
    for t in report.external:
        print(report.target_timepoint, t.kind.name, t.source, t.targets)
```

## 10. Decisions that constrain later options

Each is my recommendation; say which, if any, you want changed.

1. **Core input is a membership snapshot, not a clusterer.** This is what
   keeps the paper's re-clustering workflow and the CapyMOA adapter both
   possible. The reverse choice (MONIC wraps a `Clusterer`) would lock out
   user-supplied partitions.
2. **Ageing depends on elapsed time only.** Per-record importance weights
   are not expressible. Adding them later means a new optional argument on
   `AgeingFunction.weight`.
3. **Record IDs are integers.** String or tuple keys need a caller-side
   mapping. Widening later affects the snapshot type.
4. **`update` compares consecutive snapshots only.** Lifetimes and lineage
   are defined on that basis; arbitrary `j > i` would need a separate
   comparison entry point.
5. **Relative internal thresholds, with invented defaults.** Changing to
   absolute later changes every reported internal transition.
6. **Split children never inherit a lineage.** Changing this later
   renumbers stored lineages.
7. **Docstring style assumed Sphinx**, against the letter of `CLAUDE.md`.

