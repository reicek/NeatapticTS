# Interactive Examples + Learning Path Plan

## Purpose

Reduce time-to-first-success by shipping a small set of **runnable examples** that demonstrate:

- building networks (preconfigured + primitives)
- evolving with NEAT
- running in browser

Phase note:

- Phase 3 covers the starter learning path only.
- Standalone export and worker examples are follow-on additions after the corresponding Phase 4 capabilities exist.

## Goals

- G1: Provide a clear learning path: “hello world” → “evolution” → “browser quickstart”, then extend it with “export” and “workers” once those features are shipped.
- G2: Examples are runnable with minimal setup and are kept in sync with the public API.
- G3: Examples serve as regression checks (lightweight smoke checks).

## Non-goals

- A full interactive tutorial platform.
- A complex UI; keep examples simple and code-driven.

## Proposed structure

### Example inventory (initial)

1. **Hello Network**
   - Create a small MLP and run inference.
2. **Train XOR** (if training API exists)
   - Train or fine-tune a small network.
3. **Evolve XOR**
   - Run a tiny NEAT evolution loop.
4. **Sequence (NARX)**
   - Demonstrate state handling + reset.

### Later extensions (after Phase 4 capabilities land)

5. **Standalone export**
   - Export inference module and run it.
6. **Worker evaluation**
   - Evaluate a small batch in workers.

### Location

- Prefer `docs/examples/` (if docs workflow can include it), OR
- `examples/` at repo root with minimal build tooling.

If `bench-browser/` is already used for running code in a browser, reuse it for browser examples rather than creating a second system.

## Example quality bar

- Each example:
  - is short (ideally < 80 lines)
  - has a clear expected outcome (printed fitness, loss curve summary, etc.)
  - avoids external datasets
  - uses deterministic seeds when possible

## Implementation steps

### Step 1 — Pick a canonical examples directory

- Decide whether to use `examples/` or `docs/examples/`.
- Ensure TypeScript build config supports it.

Acceptance:

- `npm run examples:build` works (or equivalent).

### Step 2 — Add minimal Node examples

- Add “Hello Network” and “Evolve XOR”.

Acceptance:

- Examples run in < 30 seconds.

### Step 3 — Add minimal browser examples

- If browser bundles exist, add one browser example that imports from the bundle.

Acceptance:

- Example loads and runs in the browser.

### Step 4 — Document the learning path

- Add a top-level doc that lists examples in recommended order.
- Clearly mark export/worker examples as later-phase extensions so starter examples stay aligned with the roadmap.

Acceptance:

- New users can follow it linearly.

### Step 5 — CI smoke checks

- Add a lightweight CI job that runs 1–2 examples.

Acceptance:

- Prevents broken examples from landing.

## Testing strategy

- Treat examples as smoke tests, not full correctness tests.
- Ensure examples remain deterministic when seeded.

## Risks and mitigations

- Risk: examples drift from API.
  - Mitigation: CI smoke checks + keep examples minimal.

## Success criteria

- A new user can run 2–3 examples and understand the core workflows.
- Browser path is documented and validated.
