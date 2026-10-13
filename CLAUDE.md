# MONIC on CapyMOA

Implementing MONIC (Spiliopoulou et al., KDD'06) as a cluster transition
monitor. Paper PDF: `docs/refs/monic-kdd06.pdf`.
Known paper errors and open decisions: `docs/MONIC-implementation-notes.md`.
Read both before touching MONIC code.

## Non-negotiables

- The Figure 2 pseudocode in the paper is buggy. Use the corrected control
  flow in the notes file, not the paper's. Line 15 needs AND not OR, line 17
  needs a negation, and a cluster with a survival candidate NEVER enters the
  split branch.
- Def. 3 overlap is set-theoretic over persistent record IDs. Any design that
  cannot produce record-level membership cannot implement MONIC as published.
- Invariant, assert it in tests: for a partition at t_j,
  sum over Y in zeta_j of overlap(X, Y) == 1.0 for any X whose surviving
  records are all assigned. Tolerance 1e-9.
- Invariant: lifetimeS <= lifetimeI <= lifetimeA.
- Assert tau_split < tau at construction.
- Zero denominator in overlap (all of X aged out) returns 0.0, X is marked
  disappeared. Never raise, never return NaN.

## Conventions

- Match surrounding CapyMOA style: type hints on all public signatures,
  numpy docstrings, no new runtime dependencies beyond numpy.
- Tests in pytest, hand-built fixtures with known answers, no randomness
  in correctness tests.
  