# Reproducibility Reference Notes

This file stores paraphrased reference notes for the
`reproducibility-contracts` skill.

These notes summarize upstream material instead of copying it verbatim. Use the
linked sources for canonical wording and details.

## Source Map

### 1. Wikipedia: pseudorandom number generator

- URL: https://en.wikipedia.org/wiki/Pseudorandom_number_generator
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - PRNGs are deterministic sequences controlled by seed and internal state.
  - Reproducibility depends on both the generator choice and the captured state.
  - Parallel RNG work benefits from explicit stream partitioning instead of one
    implicit shared generator.

### 2. Wikipedia: floating-point arithmetic

- URL: https://en.wikipedia.org/wiki/Floating-point_arithmetic
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - IEEE 754 improves consistency, but arithmetic is still sensitive to ordering,
    rounding, cancellation, and reassociation.
  - Equivalent algebraic expressions can diverge numerically.
  - Determinism claims should state whether they are same-runtime only or bounded
    across environments.

### 3. MDN: structured clone algorithm

- URL: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm
- Authors: MDN contributors
- License note: MDN content is published under a Creative Commons license.
- Why it matters:
  - Worker handoff is based on structured data, not arbitrary object identity.
  - Functions, DOM nodes, prototypes, and some metadata do not survive cloning.
  - Replay across worker boundaries should use explicit clone-safe payloads.

## Practical Notes

### Seed is not enough

- A seed restarts a generator.
- Mid-run exact replay often needs the current internal RNG state, not only the
  original seed.

### Ordering is observable

- Floating-point reductions and tie-breaks can change results when order changes.
- Worker completion order must not be allowed to leak into semantic output order
  unless the contract says it can.

### Clone-safe state is smaller than live object state

- Structured clone preserves data graphs for supported types.
- It does not preserve everything about a live JavaScript object.
- Deterministic worker replay should rely on plain structured data and versioned
  payloads.

## Working Heuristics For This Repo

- Name the exact determinism rung before implementing.
- Capture RNG state when exact replay is required.
- Treat reduction order as part of the public contract.
- Add telemetry or metadata warnings for lossy, approximate, or runtime-bounded
  replay behavior.