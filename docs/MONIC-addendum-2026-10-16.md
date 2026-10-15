# MONIC on CapyMOA: addendum on the monitoring interface

16 October 2026 · branch `monic` at `eca2ed8`

This note records the change made after the audit note (`dddfdfd`). Like the audit note, it changes none of the earlier documents. The implementation report, the design document and the audit note describe the code as it was at `80512f6` and `dddfdfd`. Where they no longer match the code at `eca2ed8`, this note says so.

| Commit | Committed | Contents |
| --- | --- | --- |
| `0d1dea7` | 16 Oct 2026, 10:08:28 +1300 | `src/capymoa/core/_instance.py` |
| `eca2ed8` | 16 Oct 2026, 10:08:28 +1300 | `src/capymoa/evaluation/monic.py`, `tests/test_monic.py`, `tests/test_monic_properties.py`, `notebooks/clusterer/monic.py` |

## Summary

- `MONICMonitor` no longer builds clusterings from a clusterer's centres. The clusterer must implement a new interface, `MonitorableClusterer`, and return its own record-level clustering (section 1).
- The adapter has been removed. Record ids, ageing weights and noise are now for the clusterer to decide.
- No clusterer that ships with CapyMOA implements the interface, so none can be monitored. The notebook defines its own.
- Five names changed (section 2). The MONIC core changed in names only, and the six requirements in `CLAUDE.md` are met as before.
- `Instance` now has a unique id, which the notebook's clusterer uses as its record id. That change was not written in the session that made the rest (section 6).
- Several statements in the report and the design document no longer hold (section 3), and several audit findings are resolved or moot (section 5).

## 1. What changed

**The interface.** `MonitorableClusterer` (`monic.py:919`) has one abstract method, `get_clustering() -> Clustering` (`monic.py:941`). Its docstring sets the contract:

- A record keeps its id from one call to the next.
- The weights cover every record the model still holds, including records in no cluster. A record that is left out has aged out.
- Each call's timepoint is later than the previous one.
- Points are optional. Without them no internal transitions are reported.

**The monitor.** `MONICMonitor(clusterer, check_interval=1000, tau, tau_split, epsilon, delta, tau_mean, tau_skew, debug)` (`monic.py:959`). It raises `TypeError` if the clusterer does not implement the interface (`monic.py:1000`) and `ValueError` if `check_interval < 1` (`monic.py:1002`). Every `check_interval` instances it calls `get_clustering()` and passes the result to `MONIC.update`. It holds no records of its own.

**Removed.** `AgeingFunction`, `SlidingWindow`, `ExponentialDecay`, `RecordBuffer` and `snapshot_from_clusterer`, and the monitor's `ageing`, `level` and `max_distance` arguments. Their tests were removed with them, as were the DenStream monitor tests. Four monitor tests (five cases) replace them. They are driven by a clusterer that returns clusterings written in advance with hand-worked answers.

**Unchanged apart from names.** `Clustering` (formerly `ClusteringSnapshot`), `overlap`, `build_overlap_matrix`, the external and internal transition detectors, `advance_lifetimes`, `passforward_ratios`, `MONIC` and the debug checks. Design §10 item 1, that the core takes clusterings and not a clusterer, still holds: only `MONICMonitor` takes a clusterer.

**Instance ids.** Every `Instance` takes the next value of a class-wide `itertools.count` when it is created (`_instance.py:76`, `_instance.py:102`). The ids are therefore unique within a Python process, but the counter is shared by every stream and every `Instance` the process creates. The ids of one stream need not be consecutive, and they differ if the stream is read again in the same process. The attribute is private and undocumented, and nothing in `src/capymoa` reads it.

**The notebook.** DenStream is replaced by `WindowedLeader` (`notebooks/clusterer/monic.py:139`), a clusterer defined in the notebook that implements the interface. It holds the last 2000 instances with weight 1 and finds centres in the most recent 1000 with a running-mean leader algorithm. Each held instance goes to its nearest centre, or to no cluster if that centre is farther than `max_distance`. Its record ids are `Instance._id`. Its `radius` default was raised from 3.0 to 4.0 during development, because at 3.0 one of the clusters that appear at timepoint 7 was given two centres. The notebook does not say so.

## 2. Names

| At `dddfdfd` | At `eca2ed8` |
| --- | --- |
| `ClusteringSnapshot` | `Clustering` |
| `MONIC.snapshot`, `MONICMonitor.snapshot` | `MONIC.clustering`, `MONICMonitor.clustering` |
| `MONIC.update(snapshot)` | `MONIC.update(clustering)` |
| `MONICMonitor(snapshot_interval=...)` | `MONICMonitor(check_interval=...)` |
| `RecordBuffer`, `snapshot_from_clusterer` | None. A clusterer's `get_clustering()` does their work. |
| Not present | `MonitorableClusterer` and its `get_clustering()`, `Instance._id` |

In the tests, `_unit_snapshot` is now `_unit_clustering` and `test_snapshot_rejects_*` is now `test_clustering_rejects_*`. In the notebook, the list `snapshots` is now `clusterings`.

The audit note's table of design names (its section 2) maps the design's `SnapshotBuilder` to `RecordBuffer`, `snapshot_from_clusterer` and `MONICMonitor`. At `eca2ed8` its nearest counterpart is a clusterer's `get_clustering()`.

## 3. Statements in the earlier documents that no longer hold

| Where | Says | At `eca2ed8` |
| --- | --- | --- |
| Report lines 22 to 23 | The core includes `ClusteringSnapshot`. The adapter is `AgeingFunction`, `SlidingWindow`, `ExponentialDecay`, `RecordBuffer`, `snapshot_from_clusterer` and `MONICMonitor` | The core type is `Clustering`. Only `MonitorableClusterer` and `MONICMonitor` remain outside the core. |
| Report lines 29 to 38 | The monitor keeps its own records, ages them and assigns each to the nearest centre | The clusterer does all three. The monitor only asks for the result. |
| Report line 78 | Noise: an explicit `max_distance` policy in the adapter | Noise is any held record the clusterer puts in no cluster. `Clustering.without_clusters` (`monic.py:137`) remains for clusterings built by hand. |
| Report lines 105 to 151 | The DenStream example: overlaps 0.67 / 0.33 and 0.48 / 0.32 / 0.19, both threshold tables, the DenStream artefact at timepoints 9 and 10, the tuning of DenStream | The notebook no longer uses DenStream. Its results now are in section 4. |
| Report lines 157, 158 and 160 | The `max_distance` limitations. The clusterer and the buffer age independently. | There is no `max_distance` and no buffer. Whether a dead cluster's leftover records are reported as absorbed (line 157) now depends on whether the clusterer leaves them unassigned. |
| Report lines 175 and 181 | Open questions on `max_distance` and on the DenStream tuning | Moot |
| Design §0 (lines 8 to 15) | An adapter, `SnapshotBuilder`, turns a clusterer and a record buffer into a snapshot | No adapter. The clusterer returns the clustering. |
| Design §1 (lines 22 to 60, 77 to 79) | Decision (a), a record buffer on the MONIC side; its ids, nearest-centre assignment, `level`, noise in the adapter, Def. 1 compliance, unsynchronised ageing | The records are held by the clusterer, which also assigns them. Whether its clusterings are disjoint and cover its records as Def. 1 asks is for the clusterer to ensure; `Clustering` rejects overlapping clusters. Membership remains set-theoretic over persistent record ids, as `CLAUDE.md` requires. |
| Design §1 (lines 62 to 71) | Garbage clusters and noise in user-built snapshots | Still holds, under the new name, and now for clusterings a clusterer returns as well |
| Design §2 (lines 81 to 117), §3 line 144 | `AgeingFunction` and its two implementations; only `SnapshotBuilder` calls an ageing function | Removed. The rest of §3 holds for the weights a clusterer returns. |
| Design §8, lines 326 and 330 | The first version includes the ageing functions and `SnapshotBuilder` | Not in the code |
| Design §9, lines 355 to 400, and the example at lines 444 to 450 | The API with `ClusteringSnapshot`, `RecordBuffer`, `snapshot_from_clusterer`, `snapshot_interval`, `ageing`, `level` and `max_distance` | See section 2 |
| Design §10, item 2 (lines 466 to 468) | Ageing depends on elapsed time only. Per-record weights would need a new argument on `AgeingFunction.weight`. | No `AgeingFunction`. A clusterer gives each record its own weight. |
| Audit note, every `monic.py:` reference | Line numbers at `80512f6` | The file is shorter at `eca2ed8`, so they no longer match |

## 4. What was re-verified at `eca2ed8`

| Check | Found on 16 Oct 2026 |
| --- | --- |
| MONIC tests, property tests and module doctests | 123 passed with Hypothesis installed: 114 tests and 9 doctests. The audit found 150; the difference is the removed tests. Not re-run without Hypothesis. |
| Full repository suite | 655 passed, 47 skipped, with Hypothesis installed (678 passed at the audit) |
| `ruff check` and `ruff format --check` on the MONIC files | Clean |
| Tutorial notebook | Run with `runpy` (not nbmake), with `debug=True`. No invariant check fired. The output was identical before and after each rename. |

The notebook's transition log, at `tau = 0.5` and `tau_split = 0.1`:

- Timepoint 2: a cluster emerges at (0, −12).
- Timepoint 3: the cluster at (12, 0) is replaced by two. 0.75 of it goes to (12, 2), reported as its survivor, and 0.25 to (12, −4), reported as emerged.
- Timepoint 5: the cluster at (0, 12) disappears (passforward 0.83).
- Timepoint 7: the cluster at (−12, 0) splits into three, with 0.39 to (−10, −1), 0.31 to (−15, −2) and 0.30 to (−12, 3) (passforward 0.80).
- Every other external transition is a survival, and passforward is 1.00 at every other timepoint after the first.

Each of the four events above matches a change built into the stream, so the log has no artefacts.

The threshold tables now count only the 11 transitions between timepoints. The notebook skips the first clustering (`notebooks/clusterer/monic.py:388`), which resolves audit section 3 for the notebook.

**Varying `tau` at `tau_split = 0.1`**

| `tau` | Survived | Absorbed | Split | Disappeared | Emerged |
| --- | --- | --- | --- | --- | --- |
| 0.5 to 0.7 | 61 | 0 | 1 | 1 | 2 |
| 0.8 to 0.9 | 60 | 0 | 2 | 1 | 1 |

**Varying `tau_split` at `tau = 0.5`**

| `tau_split` | Survived | Absorbed | Split | Disappeared | Emerged |
| --- | --- | --- | --- | --- | --- |
| 0.10 to 0.30 | 61 | 0 | 1 | 1 | 2 |
| 0.31 | 61 | 0 | 1 | 1 | 3 |
| 0.35 | 61 | 0 | 0 | 2 | 5 |

Each of the report's five verdicts against the paper (lines 131 to 137) still holds in direction. The survival effect is now one cluster of 61 rather than two, and the second split appears at `tau = 0.8` rather than 0.70. The report's caution that only two events respond to the thresholds (line 141) applies unchanged.

## 5. Audit findings this change affects

| Audit | Finding | At `eca2ed8` |
| --- | --- | --- |
| §1 | 150 MONIC tests, 678 in the suite | 123 and 655 (section 4) |
| §2 | No `AgeingFunction.horizon` | Moot: there is no `AgeingFunction` |
| §2 | The caller cannot supply record ids | The clusterer supplies every record id |
| §2, smaller points | Zero-weight records are never members, because the buffer drops them | Nothing enforces this now. `Clustering` accepts a zero-weight member (`monic.py:106` checks only that weights are in `[0, 1]`), and the interface's contract does not forbid one. |
| §3 | The Emerged counts include the first clustering | The notebook's counts no longer do. The report is unchanged. |
| §4, noise-policy row | `max_distance` in the adapter | The clusterer's choice |
| Reproduction | The Emerged snippet reads `["snapshots"]` from the notebook | Fails at `eca2ed8`. It needs `["clusterings"]`. |

Unaffected: lineage ids, final lifetimes on the ending transition, `MONIC.history`, `MONIC.lifetimes`, the missing paper, the empty decisions log, Hypothesis missing from `pyproject.toml`, and the byline.

## 6. Timeline and authorship

- The change was made in one Claude Code session over 15 and 16 October 2026, and both commits were made at the user's request at 10:08:28 +1300 on 16 October.
- The user chose to remove the adapter rather than keep it beside the interface, and asked for each rename.
- The `_instance.py` change in `0d1dea7` was not written in that session. It appeared in the working tree while the session was running, and who wrote it is not recorded. Its commit names Justin Liu as author, as every commit on this branch does, and carries the same `Co-Authored-By: Claude Opus 5.5` trailer as `eca2ed8`, although Claude did not write it. It was committed on its own, at the user's request, because the notebook depends on it.

## How to reproduce

`capymoa` needs a Java runtime, so set `JAVA_HOME` to a JDK first.

```sh
uv run --with hypothesis python -m pytest tests/test_monic.py tests/test_monic_properties.py \
    --doctest-modules src/capymoa/evaluation/monic.py
uv run --with hypothesis python -m pytest
uv run ruff check src/capymoa/evaluation/monic.py tests/test_monic.py \
    tests/test_monic_properties.py notebooks/clusterer/monic.py
uv run ruff format --check src/capymoa/evaluation/monic.py tests/test_monic.py \
    tests/test_monic_properties.py notebooks/clusterer/monic.py
```

The threshold tables in section 4:

```python
import runpy

import pandas as pd

notebook = runpy.run_path("notebooks/clusterer/monic.py")
count = notebook["count_transitions"]
print(pd.DataFrame([count(tau, 0.1) for tau in (0.5, 0.6, 0.7, 0.8, 0.9)]))
print(pd.DataFrame([count(0.5, ts) for ts in (0.1, 0.2, 0.3, 0.31, 0.35)]))
```
