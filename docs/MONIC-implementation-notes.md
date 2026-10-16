---
title: MONIC implementation notes
paper: "MONIC: Modeling and Monitoring Cluster Transitions (Spiliopoulou, Ntoutsi, Theodoridis, Schult, KDD'06)"
tags: [anomaly-detection, streaming, clustering, implementation-notes]
status: pre-implementation
---

# MONIC: gaps and decisions before implementing

Ambiguities, errors, and undefined details found in the paper, ordered by how much they affect implementation.

## 1. Outright errors in the pseudocode (Figure 2)

- [ ] **Line 15 is wrong.** `IF survivalCandidate == NULL OR splitCandidates == ∅ THEN deadList += X` kills every cluster that has a survival candidate but no split candidates, which is the normal survival case. It must be `AND`.
- [ ] **Line 17 lost a negation** in typesetting: should read `ELSEIF splitCandidates ≠ ∅`.
- [ ] **Branch priority contradicts Table 1.** Table 1 says a cluster with a match survives or is absorbed, full stop. Even with `AND` on line 15, the `ELSEIF` on line 17 routes a cluster that has *both* a survival candidate and split candidates into the split branch.

Corrected control flow:

```text
if survivalCandidate is not None:        # survive or absorbed, decided in pass 2
    absorptions_survivals += (X, survivalCandidate)
elif splitCandidates:                    # split test
    if overlap(X, splitUnion) >= tau:
        splitList += [(X, Y) for Y in splitCandidates]
    else:
        deadList += X
else:
    deadList += X
```

- [ ] **`g(X, Y)` on line 7 is never defined**, and `g(X, NULL)` is called before any candidate exists. Def. 4 says the match is the *argmax* of overlap, but line 7 selects by `g` alone. Implement as lexicographic `(overlap, tiebreak)`, where tiebreak is one of the two the text suggests (max reverse `overlap(Y, X)`, or closest size to `X`), and initialise `g(X, NULL) = -inf`.
- [ ] **Line 34** `ELSEIF absorptionCandidates == X` means `cardinality == 1`.

## 2. The overlap definition (the one that will actually cost time)

Def. 3 weights **both** numerator and denominator by `age(·, t_j)`, including the pivot cluster `X` which was formed at `t_i`.

Consequences:

- [ ] **Zero denominator.** If every record of `X` has aged out by `t_j`, the denominator is 0. With the paper's own setup (sliding window of size 2) this happens for any `j >= i+2`, and also for the part of `X` drawn from `D_{i-1}`. Needs an explicit convention: define `overlap = 0` and classify `X` as disappeared.
- [ ] **Contradicts Table 2.** The size indicator uses `age(x, t_i)` for `X` and `age(y, t_j)` for `Y`, so `X` is measured with `t_i` weights for size but `t_j` weights for overlap. Pick one convention per purpose and document it, because it changes results.
- [ ] **Invariant to assert.** Since `ζ_j` is a partition, `Σ_{Y ∈ ζ_j} overlap(X, Y) = 1` whenever every surviving record of `X` is assigned to some cluster. Build this as a unit test.
- [ ] **Tie-breaking is nearly dead code.** For `τ > 0.5` at most one `Y` can qualify, so the tie-breaker only matters at exactly `τ = 0.5`, or at the `τ = 0.45` used in the experiments despite Def. 4 restricting `τ` to `[0.5, 1]`.
- [ ] **Record identity is assumed and never discussed.** `X ∩ Y` requires persistent record IDs across re-clusterings.
- [ ] **Zero-weight membership.** Are zero-weight records still cluster members? Def. 2 keeps them in the dataset, but in practice they are dropped before clustering, which changes both the intersection and the denominator. Decide once, globally.

## 3. Split semantics

- [ ] Table 1's maximality condition (no `Y` outside `{Y_1..Y_p}` with `overlap >= τ_split`) is satisfied by construction in the algorithm. Do not re-implement it as a separate test.
- [ ] A `Y` with `overlap >= τ` can never join a split set in the algorithm, though the definition would permit it. The definition and the algorithm are not equivalent.
- [ ] **Skip `splitUnion` entirely.** Because `ζ_j` is a partition, `overlap(X, ∪ Y_u) = Σ_u overlap(X, Y_u)`. The paper notes this at the end of 4.1.
- [ ] `p = 1` cannot pass the union test (overlap would have to be both `< τ` and `>= τ`), so no guard is needed. Assert `τ_split < τ`.

## 4. Internal transitions (most underspecified part)

- [ ] **`σ` is never defined for multivariate data.** Standard deviation of what: per-dimension vector, mean of per-dimension stds, or std of distances to the centroid? The paper's data is 30-dimensional TF-IDF. Same problem for skewness `γ` and for `|μ(X) − μ(Y)|`, where the norm is unstated.
- [ ] **`ε`, `δ`, `τ1`, `τ2` have no values and no scale invariance.** `ε` is an absolute weighted-count difference, `δ` an absolute std difference. On a growing collection this makes "expands" trivially true and "compacter" scale-dependent. Prefer relative thresholds and state the deviation.
- [ ] **Feature-space change breaks internal transitions.** The claim that MONIC works under feature-space change holds for set-based overlap but not for `μ`, `σ`, `γ`, which live in different spaces at `t_i` and `t_j`. Either restrict internal indicators to the intersection of the two feature spaces, or gate them on "feature space unchanged".
- [ ] **Which clusters get internal analysis?** Section 4.2 says survived clusters. Absorbed clusters are left undefined. Also decide whether `X ↔ Y` is a separate label or the absence of labels.
- [ ] **Weighted or unweighted moments?** Not stated. Weighted seems intended given the ageing function exists.

## 5. Lifetime and ratios

- [ ] **Two incompatible definitions.** Def. 5 counts lifetime forward from emergence; Section 4.3 computes it backward from `ζ_n` as "how many more timepoints this cluster persists". Different objects attached to different clusters. Decide which one to report.
- [ ] **Recursion only looks at `t_{i+1}`**, although matching is defined for any `j > i`. Implementing the general case requires redefining lifetime.
- [ ] **Which transitions extend which lifetime:** `lifetimeS` uses `X ↔ Y` (survival with no internal transition), `lifetimeI` uses `X → Y`, `lifetimeA` uses `X → Y` or `X ⊂→ Y`. Splits never extend any lifetime. Assert `lifetimeS <= lifetimeI <= lifetimeA`.
- [ ] **Def. 6 vs Table 3.** Def. 6 ratios are per-clustering, divided by `|ζ_i|`, but Table 3 reports absolute counts labelled as passforward ratios. Do not try to reproduce Table 3 as ratios.

## 6. Required but not given by the paper

- [ ] **Lineage IDs.** Footnote 3 says per-timepoint cluster identifiers carry no meaning, yet Section 5.4 shows chains like `C_1998,4 ↗ C_1999,9 ↗ ...`. Nothing defines how a lineage ID propagates through absorption (does the absorbing `Y` inherit from the largest absorbed cluster?) or split (do both children get fresh IDs?). This has to be invented, and it is required to reproduce their qualitative results.
- [ ] **Emerging clusters** are "clusters in `ζ_j` not resulting from an external transition". Edge case: a `Y` that overlapped a cluster `X` whose split test failed counts as emerged even though it is full of old records. Decide whether that is acceptable.
- [ ] **Noise and garbage clusters.** Def. 1 demands a complete partition with no outliers. The K-means "cluster 0" garbage bin distorts matching badly, since it is large and therefore wins matches. Any clusterer with a noise label needs an explicit policy.
- [ ] **Building the overlap matrix.** The `O(K²)` claim assumes `M` is already computed. Build it with an inverted index (`record_id -> cluster index at t_j`), then a single pass over each `X`, giving `O(|D|)` rather than `O(K² · |X|)`.

## 7. Open interface questions

- [ ] **`MonitorableClusterer.get_clustering` may need redefining** (noted 2026-10-16). Instead of the clusterer reporting the instances it holds, the caller would give it a set of instances and the clusterer would predict a cluster for each. Still to decide: who provides the ageing weights (Def. 2) for the given instances, and whether the same set is passed at every timepoint.

## Suggested first steps

1. Implement `overlap` and the transition detector against a hand-built toy example with 4 or 5 clusters where the answer is known by inspection.
2. Assert the row-sum invariant from section 2 at every timepoint. Most of the bugs above surface immediately under these two checks.

## Decisions log

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
|   | Ageing weights in overlap denominator |  |  |
|   | Zero-denominator convention |  |  |
|   | Tie-breaker for `g(X, Y)` |  |  |
|   | `σ` definition for multivariate data |  |  |
|   | Lifetime direction (forward vs backward) |  |  |
|   | Lineage ID propagation |  |  |
|   | Noise cluster policy |  |  |
