# MONIC on CapyMOA: implementation report

Oct 13, 2026 · @Jia Liu

MONIC (Spiliopoulou et al., KDD'06) is implemented in `capymoa.evaluation.monic` on branch `monic`, with all five external transitions, internal transitions, lifetimes, ratios and a monitor that wraps any CapyMOA clusterer. Its 150 tests pass and the full repository suite passes (678 passed, 47 skipped). Nothing is committed, and the lineage IDs from the design are not implemented.

## What was built

All of the code is in one new module, with no new runtime dependency beyond numpy. Every file below is untracked except `notebooks/clusterer/index.md`, which is modified.

| File | Lines | Holds |
| --- | --- | --- |
| `src/capymoa/evaluation/monic.py` | 1280 | The whole implementation: snapshot type, overlap, external and internal transitions, lifetimes, ratios, ageing, record buffer, clusterer adapter, monitor, debug checks |
| `tests/test_monic.py` | 1667 | Hand-built fixtures with known answers, no randomness |
| `tests/test_monic_properties.py` | 143 | Two Hypothesis property tests, derandomised, skipped when Hypothesis is absent |
| `notebooks/clusterer/monic.py` | 301 | Tutorial notebook (Jupytext) on a drifting stream, with the threshold sweep |
| `notebooks/clusterer/index.md` | +1 | Adds the notebook to the toctree |
| `docs/MONIC-design.md` | 479 | Design document, sections 0 to 10 |

The public surface, by layer:

- **Core, clusterer-free:** `ClusteringSnapshot`, `overlap`, `build_overlap_matrix`, `detect_external_transitions`, `detect_internal_transitions`, `advance_lifetimes`, `passforward_ratios`, and the `MONIC` class that chains them and returns a `TransitionReport` per timepoint.
- **CapyMOA adapter:** `AgeingFunction` with `SlidingWindow` and `ExponentialDecay`, `RecordBuffer`, `snapshot_from_clusterer`, and `MONICMonitor`.

The module is imported as `capymoa.evaluation.monic`. It is not yet re-exported from `capymoa.evaluation`.

## How it works

CapyMOA clusterers report centres, weights and radii but never which record belongs to which cluster, and MONIC's overlap needs exactly that. The monitor therefore keeps its own records and assigns them itself.

1. `MONICMonitor.train(instance)` trains the wrapped clusterer and adds the instance's feature vector to a `RecordBuffer`, keyed by a monotonic instance counter.
2. Every `snapshot_interval` instances is a timepoint. The buffer advances, and records whose age weight has reached zero are dropped. The default ageing is a sliding window of two timepoints, as in the paper.
3. `snapshot_from_clusterer` reads the clusterer's current centres and assigns each live record to its nearest centre. A record farther than `max_distance` from every centre is left unassigned.
4. The resulting `ClusteringSnapshot` goes to `MONIC.update`, which builds the overlap matrix in one pass over the records using an inverted index, not pairwise.
5. External transitions are detected with the corrected control flow, then internal transitions for the survivors, then lifetimes and ratios.
6. The monitor returns a `TransitionReport` carrying the overlap matrix, both sets of transitions, the lifetimes of the current clusters and the three ratios.

Users who re-cluster at each timepoint themselves, as the paper does, skip steps 1 to 3 and pass snapshots straight to `MONIC`.

**What a transition means here.** The paper re-clusters from scratch at each timepoint, while a CapyMOA clusterer is updated incrementally. A reported transition therefore describes a change in the clusterer's model, seen when the clusterer reacts and not when the data changes. The `MONICMonitor` class docstring states this.

## Project requirements

All six non-negotiables in `CLAUDE.md` are met and tested. One convention is not followed as written: docstrings use the repository's Sphinx `:param` style, not numpy style.

| Requirement | How it is met | Tested by |
| --- | --- | --- |
| Corrected Figure 2 control flow; a cluster with a survival candidate never enters the split branch | A match (best overlap at or above `tau`) is decided first. Split candidates are considered only when there is no match. | `test_match_takes_priority_over_split_candidates`, `test_absorption_takes_priority_over_split_candidates`, `test_failed_split_disappears` |
| Overlap is set-theoretic over persistent record IDs | Snapshots map cluster id to a set of record ids. The buffer's ids are the instance counter and never change. | `test_overlap_matrix_known_values`, `test_overlap_matrix_matches_pairwise_overlap` |
| Overlap rows sum to 1.0 within 1e-9 when all surviving records are assigned | Follows from Def. 3 with both sums weighted at the later timepoint | `test_rows_sum_to_one_when_all_records_survive`, plus the debug check at every timepoint |
| `lifetimeS <= lifetimeI <= lifetimeA` | Holds by construction of the lifetime update | `test_lifetime_invariant_over_five_timepoints`, `test_lifetimes_are_ordered` (300 generated histories) |
| `tau_split < tau` asserted at construction | `ValueError` unless `0 < tau_split < tau <= 1` | `test_invalid_thresholds_are_rejected`, `test_monic_validates_thresholds_at_construction` |
| Zero denominator returns 0.0 and the cluster is disappeared; no exception, no NaN | Division is guarded; an all-zero row has no match and no split candidate | `test_fully_aged_out_cluster_has_zero_overlap`, `test_no_candidates_disappears_when_aged_out` |

On the conventions:

- **Type hints** are on all public signatures.
- **Docstrings** are Sphinx style because that is what the surrounding code uses and what the docs build expects. `CLAUDE.md` asks for both "match surrounding style" and "numpy docstrings", which conflict in this repository.
- **Correctness tests** use no randomness. The two property tests use Hypothesis with `derandomize=True` and no example database, so they generate the same cases on every run.

## Where it departs from or adds to the paper

The paper leaves these points undefined or defines them in a way that does not work on a stream. Each choice is justified in `docs/MONIC-design.md`; that document was written from a plan you did not formally approve.

| Point | Paper | Implementation |
| --- | --- | --- |
| Tie between equally good matches | "Different tie breakers can be used" | Larger reverse overlap wins, then the earlier cluster |
| `tau` below 0.5 | Def. 4 restricts `tau` to 0.5 to 1, yet section 5.2 runs 0.45 | Accepted with a `UserWarning` |
| `tau_split` of zero | Not excluded | Rejected, since every cluster would become a split candidate |
| Internal thresholds | Absolute differences | Relative: size and spread as fractions, mean shift in units of the cluster's spread. Skewness stays absolute. |
| Default internal thresholds | None given | `epsilon = 0.1`, `delta = 0.1`, `tau_mean = 0.5`, `tau_skew = 0.5`. These are placeholders, not values from the paper. |
| Spread of a cluster | "Standard deviation", undefined for several features | Root mean square distance to the weighted mean |
| Location transition | Two indicators, mean and skewness | One label, `SHIFTED`, raised by either indicator |
| Changed feature space | Not addressed | Internal analysis is skipped for that pair and the result says so |
| Lifetime direction | Def. 5 counts forward; section 4.3 computes backward from a final clustering | Forward, since a stream has no final clustering |
| Lifetime after an absorption | Not defined | The absorbing cluster continues the absorption lifetime of the source with the heaviest intersection |
| Table 3 | Headed as ratios, reports absolute counts | Ratios per Def. 6; the docstring notes the discrepancy |
| Noise | Assumed removed beforehand | Explicit `max_distance` policy in the adapter, default off |

The statistics of a cluster use its own timepoint's weights, and overlap uses the later timepoint's weights. Both follow the paper, and every report records which convention produced it.

## Verification

Everything below was re-run on 13 October 2026 except the notebook, which last ran on 12 October.

| Check | Result |
| --- | --- |
| MONIC tests, property tests and module doctests | 150 passed in 9 s |
| Full repository suite (`invoke test.pytest`) | 678 passed, 47 skipped in 107 s |
| `ruff check` and `ruff format --check` on the MONIC files | Clean |
| Notebook under nbmake | Passed in under 5 s |

What the tests cover beyond the requirements table:

- **External transitions:** the ten cases you specified from Table 1, plus empty clusterings, a tie at `tau = 0.5`, and a cluster that is both a match and a split child.
- **Internal transitions:** each label alone, combinations, weighted moments with worked values, scale invariance, zero-spread and constant-feature clusters.
- **Lifetimes:** a five-timepoint history with literal expected values for all three lifetimes.
- **Monitor:** a drifting stream with a split at timepoint 3 and a death at timepoint 5, once with a deterministic test clusterer and once with DenStream.
- **Debug mode:** nine tests corrupt a valid report in one way each and confirm the matching check fires.

After each step I broke the implementation on a scratch copy in targeted ways and confirmed that tests failed, then restored the file and checked it was byte-identical.

Two caveats on the test setup. Hypothesis is installed in the local environment only, so the two property tests are skipped in CI until it is added to the `dev` extra. With Hypothesis installed, pytest prints one warning about the project's `norecursedirs` setting, which is harmless but would fail a run under `-W error`.

## Example and threshold sweep

The sweep only partly matches the paper's Figures 3 and 4: the directions agree, but disappearances do not rise with `tau`, and the split is not confined to small `tau_split`.

The example stream has 12,000 two-dimensional Gaussian instances and 12 timepoints. A cluster emerges at instance 2000, one is replaced by two at 3000 (lopsided, 0.67 / 0.33), one dies at 5000, and one is replaced by three at 7000 (0.48 / 0.32 / 0.19). The clusterer is DenStream and the monitor runs with `tau = 0.5`, `tau_split = 0.1`, `max_distance = 6.0` and `debug=True`.

The log reports the emergence, the death (passforward 0.83) and the three-way split (passforward 0.80) at the right timepoints. Passforward is 1.00 at every other timepoint after the first. Counts below are totals over the 11 transitions between timepoints.

**Varying `tau` at `tau_split = 0.1`**

| `tau` | Survived | Absorbed | Split | Disappeared | Emerged |
| --- | --- | --- | --- | --- | --- |
| 0.45 | 61 | 2 | 0 | 1 | 9 |
| 0.50 to 0.65 | 60 | 2 | 1 | 1 | 7 |
| 0.70 | 59 | 2 | 2 | 1 | 6 |

**Varying `tau_split` at `tau = 0.5`**

| `tau_split` | Survived | Absorbed | Split | Disappeared | Emerged |
| --- | --- | --- | --- | --- | --- |
| 0.10 to 0.15 | 60 | 2 | 1 | 1 | 7 |
| 0.20 to 0.30 | 60 | 2 | 1 | 1 | 8 |
| 0.35 | 60 | 2 | 0 | 2 | 10 |

**Against the paper**

| Paper's trend | Here | Verdict |
| --- | --- | --- |
| Survivals fall as `tau` rises (Fig. 3a) | 61, 60, 59 | Direction matches; the effect is 2 clusters of 61 |
| Disappearances rise as `tau` rises (Fig. 3b) | Flat at 1 | Does not match |
| Splits rise with `tau`, then stop changing above 0.6 (Fig. 3c) | 0, 1, 2, with the last rise at 0.70 | Direction matches; the saturation point does not |
| Splits occur only at small `tau_split` (Fig. 4a) | The split persists to 0.30 and vanishes at 0.35 | Weak match: non-increasing, but not confined to small values |
| Disappearances rise with `tau_split` (Fig. 4b) | 1, then 2 at 0.35 | Direction matches |

Three reasons not to read this as a reproduction:

- **Only two events respond to the thresholds.** Every change in the tables traces to the two replaced clusters. The curves are step functions over two data points.
- **The matching directions are mostly guaranteed by the definitions.** A larger `tau` can never add a match and a larger `tau_split` can never add a split, on any stream.
- **The disappearance mismatch is structural.** Here the records of a replaced cluster all land in some new cluster, so a lost survival becomes a split. In the paper's document data, records scatter.

The paper's curves cannot be checked without its data: ACM H.2.8 documents, re-clustered with bisecting K-means at K = 10.

**Three things to know before relying on the example**

- The lopsided replacement at timepoint 3 is reported as a survival plus an emergence. Two thirds of the old cluster went to one place, which is a match at `tau = 0.5`. It becomes a split only at `tau = 0.7`.
- Timepoints 9 and 10 show an emergence and an absorption because DenStream briefly reported one cluster as two.
- The DenStream settings (`lambda_option=0.5`, `speed=30`) were tuned. Of the 17 settings I tried, across two versions of the stream, this was the only one that tracked every event with a single artefact. The notebook explains the first two points but does not mention the tuning.

## Limitations and what is not done

**Behaviour to be aware of**

- **A dead cluster is reported as absorbed unless `max_distance` is set.** With the default of `None`, its leftover records are assigned to the nearest surviving cluster. A test pins this behaviour.
- **`max_distance` is an absolute distance the user must choose.** The clusterers' own radii could not be used: DenStream's macro-clusters and ClusTree report a radius of 0.
- **Results depend heavily on the clusterer.** Only a clusterer that forgets about as fast as the monitor's window reported a split as a split. Slower settings showed an emergence followed later by a disappearance. CluStream with k-means never reported the split. On richer streams DenStream produced spurious transitions.
- **The clusterer and the buffer age independently.** The clusterer forgets on its own schedule and the buffer on the monitor's ageing function.
- **Only consecutive timepoints are compared.**

**Not done**

- **Lineage IDs** (design section 7) are designed but not implemented. Clusters are identified per timepoint only.
- **No export from `capymoa.evaluation`.** The design said to list the module in `__all__`; `__init__.py` is untouched.
- **`docs/MONIC-design.md` does not mention `debug` mode.**
- **`docs/refs/monic-kdd06.pdf` now exists**, where `CLAUDE.md` points. You added it on 13 October; it is untracked.
- **The decisions log in `docs/MONIC-implementation-notes.md` is still empty.** I did not edit that file.
- **Nothing is committed.**

## Open questions for you

- [ ] Docstring style: keep Sphinx, or convert to numpy as `CLAUDE.md` says?
- [ ] Should `max_distance` be a required argument, so that nobody gets deaths reported as absorptions by default?
- [ ] Add Hypothesis to the `dev` extra so the property tests run in CI?
- [ ] Implement lineage IDs next, or leave them out of the first version?
- [ ] Export the module from `capymoa.evaluation`?
- [ ] `CLAUDE.md` says line 17 of Figure 2 "needs a negation". In the PDF the negation is present; it is lost only in text extraction. Correct the bullet?
- [ ] The paper is now at `docs/refs/monic-kdd06.pdf`: commit it or gitignore it? Check the licence before committing a publisher PDF.
- [ ] Should the notebook say that the DenStream settings were tuned?
