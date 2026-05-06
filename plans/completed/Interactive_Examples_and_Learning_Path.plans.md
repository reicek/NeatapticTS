# Interactive Examples + Learning Path Plan

**Status:** [DONE]

## Scope

- Execute the Phase 3 starter-examples lane by adding a small set of runnable examples and the supporting learning-path documentation under `examples/`.
- Reuse the existing browser build, docs copy, and smoke-test surfaces instead of creating a second delivery path.
- Keep `NEATchat` as a separate follow-up lane owned by [plans/NEATchat.plans.md](../NEATchat.plans.md) rather than folding chat-specific work into the starter tranche.

## Current state

- `examples/` is already the canonical home for runnable examples.
- The current example surface is weighted toward the two flagship systems documented in [examples/README.md](../examples/README.md): `flappy_bird/` and `asciiMaze/`.
- The repo already has usable browser and docs plumbing for examples through `build:browser`, `smoke:browser`, `docs:examples`, and the lightweight `bench-browser/` harness.
- `docs/examples/index.html` now publishes all current demo surfaces instead of only the browser-hosted flagships: it groups `helloNetwork` and `evolveXor` under starter examples, keeps `asciiMaze` and `flappy_bird` under flagship demos, and the starter examples now have real browser hosts with rendered result tables rather than source-first placeholder pages.
- The starter browser demos now share the repo's neon retro-arcade direction as well: both pages use TRON-style dark shells, cyan neon double outlines, and consistent framed table presentation instead of separate ad hoc starter styling.
- The starter learning-path examples are now present and browser-hosted: `helloNetwork`, `evolveXor`, and `sequenceReset` all ship lightweight browser entry pages and are published through the shared docs examples index.
- The top-level examples documentation does not yet offer a clear starter-to-flagship reading order.
- `NEATchat` is already tracked separately in [plans/NEATchat.plans.md](../NEATchat.plans.md) and should remain a follow-up after the starter tranche is moving.

## Coverage backlog

### [DONE] Freeze the host and boundary decisions

- Keep `examples/` as the canonical example root.
- Reuse the existing example browser and docs plumbing instead of creating a parallel `docs/examples/` system.
- Keep `NEATchat` out of the starter deliverables and preserve it as a separate Phase 3 follow-up plan.

### [DONE] Step 1 - Starter tranche shape and file layout

- Confirm the folder and README pattern for the starter examples so they fit naturally beside the flagship examples instead of becoming another ad hoc surface.
- The frozen starter pattern is folder-based under `examples/<example-name>/`, with at minimum an `index.ts` entrypoint plus `README.md`; browser-published examples also need `index.html`.
- The current docs publication path is explicit rather than automatic: browser examples must be added to `scripts/copy-examples.ts` if they should appear under `docs/examples/`.
- Decide the smallest stable starter set to implement first:
  - `Hello Network` for fast inference and public API orientation.
  - `Evolve XOR` for the smallest NEAT loop with a predictable outcome.
  - One compact sequence-state example that teaches reset behavior with a recurrent builder.
  - One minimal browser quickstart that proves the browser path without recreating Flappy Bird.
- Keep the first pass small enough that each example can be validated independently.

Acceptance:

- The starter set is frozen into concrete example boundaries before broader implementation begins.

### [DONE] Step 2 - Add `Hello Network`

- Create the smallest Node-first inference example that imports the public API, builds a tiny network, runs one inference pass, and explains the input and output shape.
- Keep the example short enough to teach the API without reintroducing training, evolution, or browser concerns.
- Add a short README or inline doc surface that explains why this is the first stop in the learning path.
- The implemented slice now lives in `examples/helloNetwork/` with a deterministic walkthrough, a focused owner-local test, and the runnable `npm run example:hello-network` entrypoint backed by `tsx`.

Acceptance:

- A new user can run one small script and see a concrete output immediately.

### [DONE] Step 3 - Add `Evolve XOR`

- Create a tiny seeded evolution example that demonstrates the `Neat` loop on XOR with a clear stop condition or generation budget.
- Keep the output understandable: best fitness, solved state, or a short summary rather than noisy logs.
- Treat this example as the first evolution checkpoint in the learning path, not as a benchmark.
- The implemented slice now lives in `examples/evolveXor/` with a seeded feed-forward controller, a focused owner-local Jest test, a runnable `npm run example:evolve-xor` entrypoint, and an educational README.
- While hardening the example, the underlying add-node acyclic split path was repaired so feed-forward NEAT can reach a solved XOR state instead of failing later with duplicate connection innovations during crossover.
- Exact-float replay is still not the right smoke-test contract, so the example test protects bounded solved behavior instead: budget-respecting completion, correct XOR row order, solved score range, and predictions on the expected side of the decision boundary.

Acceptance:

- A user can see a minimal end-to-end neuroevolution loop with deterministic-enough behavior for smoke validation.

### [DONE] Step 4 - Add the sequence reset example

- Created `examples/sequenceReset/` using the same folder-based pattern as `helloNetwork` and `evolveXor`.
- `index.ts`: builds a fixed-weight `Architect.lstm(1, 4, 1)`, runs the same five-step sequence three times — baseline, after `network.clear()`, and without clear — and returns a structured result object.
- `run.ts`: console runner (`npx tsx examples/sequenceReset/run.ts`).
- `sequenceReset.test.ts`: single `it()` / single top-level `expect()` validating that `afterResetMatchesFreshRun: true` and `carryoverDiffersFreshRun: true`.
- `browser-entry.ts`: IIFE bundle exposing `window.sequenceReset` and `window.sequenceResetStart`.
- `index.html` and `README.md`: browser page and educational documentation with Mermaid diagram.
- Registered in `scripts/copy-examples.ts` and `package.json` (`build:sequence-reset` script).
- **Library fix**: revealed that `Node.clear()` was resetting gated connection gains to `0` instead of the neutral default `1`. Fixed in `src/architecture/node/node.ts` so that `clear()` truly restores fresh-state behavior.

Acceptance:

- The example demonstrates state carryover and reset behavior clearly in one short run.

### [DONE] Step 5 - Add the browser quickstart

- The starter browser quickstart is now covered by the lightweight browser-hosted starter examples in `examples/helloNetwork/`, `examples/evolveXor/`, and `examples/sequenceReset/`.
- Each starter example now exposes an `index.html` browser entrypoint and is published through the shared docs examples path so users can open one small browser walkthrough before moving to flagship demos.
- The starter browser surfaces remain intentionally small: no worker control panel, no replay system, and no bespoke visualizer requirements.

Acceptance:

- A new user can load one lightweight browser example without having to start from Flappy Bird or ASCII Maze.

### [DONE] Step 6 - Document the learning path

- Update the examples documentation so the starter examples come first and the flagship demos read as the next step, not the entry point.
- Explain the recommended order: inference first, minimal evolution second, sequence state third, browser quickstart fourth, then flagship examples.
- Keep `NEATchat` listed as an advanced learnability follow-up rather than part of the starter path.
- The docs publication surface is no longer blocked on browser-only inventory: `scripts/copy-examples.ts` now groups starter versus flagship examples, and `helloNetwork` plus `evolveXor` now publish as browser-hosted starter pages with visible result tables on the docs site.
- Rewrote the top-level `examples/README.md` opening so it now acts like a guided syllabus instead of a flat showcase.
- Added a starter-path table that links each example to the concept it teaches and the next recommended stop.
- Kept the browser quickstart tied to the starter pages already published through `docs/examples/` instead of inventing a separate quickstart surface.
- Demoted the flagship example descriptions into the second half of the README so they now read as system-level follow-on study.
- Added an explicit `NEATchat` note that keeps it positioned as an advanced follow-up rather than part of the starter path.
- Validated the edited markdown with workspace diagnostics and a relative-link existence check across the rewritten README.

Acceptance:

- The examples surface reads like a deliberate path instead of a flat list of demos.

### [DONE] Step 7 - Add smoke validation

- Added `examples/starter-examples.smoke.test.ts` with 8 focused smoke checks (export surface + format-title tests) for all three starter examples.
- Added a `starter-examples` Jest project in `jest.config.mjs` that matches only the smoke file so `--selectProjects starter-examples` resolves without PowerShell pipe-regex issues.
- Added `test:smoke:starters` npm script: runs the smoke project in ~9 s without triggering a full evolution pass.
- Excluded the smoke file from the `default` project to prevent double-run during `npm test`; the three behavioral tests remain in `default` as before.

Acceptance:

- The starter examples have a small, maintainable validation path that catches obvious breakage.

## Immediate next steps

All steps complete. The starter-examples Phase 3 lane is done.

## Deferred questions

- If the minimal browser quickstart can cleanly use the public browser bundle, prefer that over another example-specific bundle.
- If a `Train XOR` example still looks necessary after `Hello Network`, `Evolve XOR`, and the sequence reset example exist, add it only as a follow-up and not as a blocker for the starter tranche.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Work on the active Phase 3 starter-examples lane in plans/Interactive_Examples_and_Learning_Path.md. Keep NEATchat separate in plans/NEATchat.plans.md. Reuse the existing examples root plus browser/docs plumbing. `examples/helloNetwork/`, `examples/evolveXor/`, and `examples/sequenceReset/` are done and validated; keep that folder-based pattern. All three starter examples publish as browser-hosted result tables through docs/assets bundles. Step 5 and Step 6 are complete, including the starter-first rewrite of `examples/README.md`. The next active slice is Step 7: add narrow smoke checks that protect the starter examples without turning them into a slow end-to-end lane.
```
