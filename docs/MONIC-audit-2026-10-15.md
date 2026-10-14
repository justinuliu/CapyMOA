# MONIC on CapyMOA: audit note

15 October 2026 · branch `monic` at `80512f6`

This note compares the four MONIC commits with each other and re-runs the checks in the implementation report. It changes none of them. The report, the design document, the implementation notes and `CLAUDE.md` stay as they were committed, as the record of what was decided and claimed at the time. Where they are wrong or out of date, this note says so.

| Commit | Committed | Contents |
| --- | --- | --- |
| `5f8454e` | 13 Oct 2026, 20:25:46 +1300 | `docs/MONIC-implementation-report.md` |
| `42fd6c8` | 13 Oct 2026, 20:26:17 +1300 | `src/capymoa/evaluation/monic.py`, `tests/test_monic.py`, `tests/test_monic_properties.py` |
| `c362b35` | 13 Oct 2026, 20:26:17 +1300 | `notebooks/clusterer/monic.py`, one line in `notebooks/clusterer/index.md` |
| `80512f6` | 13 Oct 2026, 20:26:17 +1300 | `CLAUDE.md`, `docs/MONIC-implementation-notes.md`, `docs/MONIC-design.md` |

## Summary

- The code meets all six requirements in `CLAUDE.md`, and every number in the report that was re-run reproduces.
- The report's "Not done" list is incomplete. Four things the design document specifies were not built and are not listed (section 2).
- One statement in the report is wrong: the Emerged counts include the first clustering (section 3).
- Several statements are out of date, and the decisions log they point to is empty. Where each decision was actually made is listed in section 4.
- The paper is not in the repository, so the claims about it cannot be checked from the record, and two of them contradict each other (section 5).
- The git history does not show the order in which the work was done (section 6).

## 1. What was re-verified

| Check | Report says | Found on 15 Oct 2026 |
| --- | --- | --- |
| MONIC tests, property tests and module doctests | 150 passed | 150 passed with Hypothesis installed. Without it, 148 passed and 1 skipped: the property-test file skips as a whole. Hypothesis is not in `pyproject.toml`. |
| Full repository suite | 678 passed, 47 skipped | 678 passed, 47 skipped (with Hypothesis installed) |
| `ruff check` and `ruff format --check` on the MONIC files | Clean | Clean |
| Tutorial notebook | Passed under nbmake on 12 Oct | Re-run with `runpy` (not nbmake), with `debug=True`. No invariant check fired. The transition log, the overlap rows of the two replaced clusters (0.67 / 0.33 and 0.48 / 0.32 / 0.19) and both threshold tables match the report exactly. |
| Six requirements in `CLAUDE.md` | Met and tested | Confirmed by reading the code and tests. Corrected Figure 2 flow: a match is decided first and splits are tried only without one (`monic.py:353`). Zero denominator gives 0.0 (`monic.py:184`, `monic.py:243`). `0 < tau_split < tau <= 1` raises `ValueError` at construction (`monic.py:322`), not an `assert`. Row-sum test at tolerance 1e-9 (`tests/test_monic.py:34`, `tests/test_monic.py:97`). |
| Docstring style | The repository uses Sphinx, which conflicts with "numpy docstrings" in `CLAUDE.md` | 133 files under `src/capymoa` use `:param`; 2 use numpy-style `Parameters` sections; all of `src/capymoa/evaluation` uses `:param`. |

## 2. Designed but not built, and not in the report's "Not done" list

The report's "Not done" list (report lines 163 to 170) names the lineage IDs, the export from `capymoa.evaluation`, and debug mode missing from the design document. These are also missing:

| Design | Says | Code |
| --- | --- | --- |
| §2, lines 93 to 96 | `AgeingFunction` has an abstract `horizon` property | No `horizon`. `RecordBuffer` drops records from the oldest end while their weight is 0 (`monic.py:1043`), which relies on weights never rising with age. The `AgeingFunction.weight` docstring states that rule. |
| §1, lines 29 to 30 | Record IDs are arrival indices "unless the caller supplies them" | `RecordBuffer.add` (`monic.py:1020`) takes no ID. Caller-chosen IDs are possible only by building a `ClusteringSnapshot` directly. |
| §6, lines 280 to 281 | A lineage's final lifetimes are recorded on the transition that ended it | `ExternalTransition` holds only `kind`, `source` and `targets` (`monic.py:268`). A cluster's last lifetimes are in the previous report's `lifetimes`. |
| §9, line 384 | `MONIC.history` returns the past reports | Not present. Callers keep the reports themselves, as the notebook does. |

`MONIC.clusters` (§6 line 279, §9 line 382) is built under another name: `MONIC.lifetimes` (`monic.py:806`) gives the lifetimes of every live cluster, without the lineage field.

Smaller points:

- **Zero-weight records** (§2, lines 111 to 116) "are not cluster members". The adapter enforces this, because the buffer drops them. `ClusteringSnapshot` accepts zero-weight members in a snapshot the user builds (`monic.py:108` checks only that weights are in `[0, 1]`).
- **Ties for the dominant parent of an absorption** (§7, lines 308 to 310) fall back to the lower lineage ID. With no lineage IDs, the code falls back to the order of the earlier clustering (`monic.py:657`).
- **Scope** (§8): lineage IDs were to ship in v1 (line 329) and did not; the report says so. A tutorial notebook was deferred (line 346), but `c362b35` ships one. "No randomness" in tests (line 333) is met for correctness tests; the two property tests use Hypothesis with `derandomize=True`, as the report says.

### Names that differ between the design and the code

The design's API sketch (§9) was not updated after implementation. Its own usage example (lines 453 to 455) already uses the code's names.

| Design | Code |
| --- | --- |
| `SnapshotBuilder` (§0, §1, §3) | `RecordBuffer`, `snapshot_from_clusterer` and `MONICMonitor` |
| `MONIC.clusters`, a list of `ClusterState` | `MONIC.lifetimes`, a dict of cluster id to `Lifetime` |
| `ClusterState.lifetime_strict`, `lifetime_internal`, `lifetime_absorption` | `Lifetime.strict`, `internal`, `absorption` |
| One `Transition` type with `internal` and `internal_skipped` | `ExternalTransition`, and `InternalTransition` with `kinds` and `skipped` |
| `TransitionReport.transitions` | `TransitionReport.external` and `TransitionReport.internal` |
| `source_timestamp`, `target_timestamp` | `source_timepoint` (`None` on the first report), `target_timepoint` |
| `survival_ratio`, `absorption_ratio`, `passforward_ratio` | `ratios.survival`, `ratios.absorption`, `ratios.passforward` |
| `MONICMonitor(..., **thresholds)` | Explicit keyword arguments |
| Not in the design | `debug` on `MONIC` and `MONICMonitor`; `TransitionReport.lifetimes` |

## 3. A wrong statement in the report

Report line 111: "Counts below are totals over the 11 transitions between timepoints."

The Emerged column also counts the 4 clusters of the first clustering, which `MONIC.update` reports as emerged because there is nothing before them. At `tau = 0.5` and `tau_split = 0.1`, 4 of the 7 emergences come from the first report, and 3 come from transitions (at timepoints 2, 3 and 9). Every Emerged value in both of the report's threshold tables is 4 higher than the number of emergences between timepoints.

The other four columns are correct: they sum to 64, the number of clusters at timepoints 0 to 10. The comparison with the paper (report lines 131 to 137) does not use emergences, so its conclusions stand. The notebook's tables include the same 4 but do not claim to cover only transitions.

## 4. Out-of-date statements, and where the decisions are recorded

| Where | Says | Now |
| --- | --- | --- |
| Report lines 5, 9 and 170 | Nothing is committed; the files are untracked | Committed in `42fd6c8`, `c362b35` and `80512f6` |
| Report line 168, `CLAUDE.md` line 4 | The paper is at `docs/refs/monic-kdd06.pdf` | Not on disk. `*.pdf` is gitignored (`.gitignore:200`). |
| Notes front matter (line 5), design line 3 | Pre-implementation | Implementation committed. Every checkbox in the notes is still open. |
| Notes lines 82 to 92, report line 169 | The decisions log is empty | Still empty. The decisions are in the design document, as below. |

| Decisions log row (notes) | Decided in | Decision | In code |
| --- | --- | --- | --- |
| Ageing weights in overlap denominator | Design §3 | Both `X` and `Y` weighted at `t_j` | `monic.py:235` |
| Zero-denominator convention | Design §3 | Overlap 0.0, `X` disappears | `monic.py:184`, `monic.py:243` |
| Tie-breaker for `g(X, Y)` | Design §4 | Overlap, then reverse overlap, then the earlier cluster | `monic.py:356` |
| `σ` for multivariate data | Design §5 | Root mean square distance to the weighted mean | `monic.py:471` |
| Lifetime direction | Design §6 | Forward, per Def. 5 | `monic.py:589` |
| Lineage ID propagation | Design §7 | Designed | Not implemented |
| Noise cluster policy | Design §1 | `max_distance` in the adapter; `without_clusters` for user-built snapshots | `monic.py:1121`, `monic.py:140` |

## 5. Claims about the paper that cannot be checked here

The paper is not in the repository. Citation: Spiliopoulou, M., Ntoutsi, I., Theodoridis, Y., & Schult, R. (2006). MONIC: modeling and monitoring cluster transitions. In *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 706–711).

Without it:

- `CLAUDE.md` lines 10 to 13 say line 17 of Figure 2 "needs a negation". Report line 179 says the negation is present in the PDF and was lost only in text extraction. The record holds both claims and nothing that settles them.
- The Figure 2 errors in the notes (lines 14 to 16) and the report's comparison with the paper's Figures 3 and 4 rest on the PDF.

## 6. Timeline and authorship

- All four commits were made within 31 seconds on 13 October, in this order: report, code, notebook, then the specifications. The report and the design document both refer to `CLAUDE.md` and the implementation notes as existing requirements, so those two files existed before the code although they were committed last. The commit times record when the files were committed, not when they were written. The only other dates are inside the documents: the report is dated 13 October, and the notebook says "last update on 12/10/2026". The history has not been rewritten.
- The report's byline (line 3) reads "@Jia Liu". The commits are authored by Justin Liu. This note does not resolve the difference.

## How to reproduce

`capymoa` needs a Java runtime, so set `JAVA_HOME` to a JDK first.

```sh
uv run --with hypothesis python -m pytest tests/test_monic.py tests/test_monic_properties.py \
    --doctest-modules src/capymoa/evaluation/monic.py
uv run --with hypothesis python -m pytest
uv run ruff check src/capymoa/evaluation/monic.py tests/test_monic.py \
    tests/test_monic_properties.py notebooks/clusterer/monic.py
```

The Emerged counts in section 3:

```python
import runpy
from collections import Counter

from capymoa.evaluation.monic import MONIC

snapshots = runpy.run_path("notebooks/clusterer/monic.py")["snapshots"]
monic = MONIC(tau=0.5, tau_split=0.1)
reports = [monic.update(snapshot) for snapshot in snapshots]
print(Counter(t.kind.name for t in reports[0].external))  # first clustering
print(Counter(t.kind.name for r in reports[1:] for t in r.external))  # 11 transitions
```
