# Preconfigured Architectures (MLP + Sequence Builders) Plan

**Status:** [DONE]

## Purpose

Provide a set of **high-quality, preconfigured architecture builders** that are:

- easy to use (one-liners)
- deterministic
- compatible with evolution and optional gradient fine-tuning
- implemented using the primitives + construct pipeline (or existing internal helpers) so we avoid “two ways to build everything”
- strong enough to upgrade the flagship demos into better reference projects and broader end-to-end test surfaces before Phase 3

## Goals

- G1: Ship ergonomic builders for common architectures.
- G2: Ensure all builders produce networks with explicit I/O roles and stable activation order.
- G3: Provide clear docs with small examples and expected usage patterns.
- G4: Use the builder surface to remove demo-local architecture drift in Flappy Bird and ASCII Maze.
- G5: Treat the demos as reference projects and e2e coverage targets, not as detached showcase apps.

## Non-goals

- Implementing every deep learning layer type.
- Claiming strict equivalence with specialized frameworks.
- Making these architectures ONNX-exportable by default (that’s constrained by ONNX plan).

## Current state

- [DONE] Step 1: MLP builder hardening is closed for the current plan scope; the public `Architect.perceptron(...)` path now produces explicit I/O roles and stable feed-forward intent.
- [DONE] Step 2: `Architect.randomSparse(...)` now supports deterministic seeded replay through the public sparse-builder surface, and downstream architecture matrices exercise `RandomSparse` beyond its owner-local tests.
- [DONE] Step 3: `Architect.narx(...)` is now closed for the current scope with explicit delay-line coverage, public clear-state guidance, and a small deterministic sequence example.
- [DONE] Step 4: `Architect.lstm(...)` and `Architect.gru(...)` are now closed for the current scope with explicit gated-block coverage, GRU shortcut-option parity, direct evolution compatibility, and recurrent training smoke coverage.
- [DONE] Step 5: the public `Architect` builder docs now include runnable examples, recommended starting knobs, and explicit pitfall guidance for `Perceptron`, `RandomSparse`, `NARX`, `GRU`, and `LSTM`.
- [DONE] Step 6: the shared demo architecture-profile contract is landed through one shared example profile registry.
- [DONE] Step 7: Flappy Bird now starts fresh worker-backed runs from multiple shared architecture profiles through a browser selector, explicit worker/HUD profile labels, and browser-local per-profile best-score tracking.
- [DONE] Early Step 8 slice: ASCII Maze now accepts an additive `architectureProfileId` option that threads the shared contract into its evolution-engine and NEAT setup path.
- [DONE] Step 8: ASCII Maze default profile rollout and telemetry/archive metadata. NARX, GRU, LSTM approved for ASCII Maze; MLP is the default seed profile on fresh starts; architectureProfileId threaded through MazeEvolutionRunResult and BrowserEntryCurriculumContext; structuredClone polyfill added to jest-setup for jsdom environment.
- [DONE] Step 9: cross-demo e2e matrix. ASCII Maze NARX/GRU/LSTM profile network shape tests added to architectureProfiles.test.ts; optionsAndSetup.test.ts covers default profile, explicit NARX/GRU/LSTM profiles, and population-override suppression.
- [DONE] Step 10: demo documentation. ASCII Maze README and Flappy Bird README both document the approved architecture profile families, usage patterns, recurrent state guidance, and Mermaid flow diagrams.

## Architecture set (initial)

### [DONE] 1) `Perceptron` / MLP

- `Perceptron(input, ...hidden, output)`
- Fully-connected feed-forward.

### [DONE] 2) `RandomSparse`

- `RandomSparse(input, hidden, output, { connections, backConnections, selfConnections, gates })`
- Focus on evolution-friendly sparse initializations.
- Completion note: the first-class `randomSparse()` builder, validation path, deterministic seeded replay, and broader downstream coverage are landed for the current plan scope.

### [PLANNED] 3) Sequence / time-series family

These are intended for tasks where state matters.

- `NARX(input, hiddenSizes, output, inputMemory, outputMemory)`
- “Remember last N inputs/outputs” via identity memory blocks.
- `LSTM(input, ...blockSizes, output, options)`
- A pedagogical LSTM-like gating structure.
- `GRU(input, ...unitSizes, output)`
- A pedagogical GRU-like structure.

Important note: these are educational implementations intended to interoperate with evolution; they should be clearly documented as such.

## Design principles

- Prefer **small, explicit graphs** rather than clever magic.
- Use **named substructures** (input gate, forget gate, memory cell, etc.) to support visualization and debugging.
- Ensure deterministic initial weights when a seed/initializer is provided.
- When a builder becomes public and stable, demos should consume that builder instead of keeping demo-local seed-network logic.
- Demo architecture selection should always mean **start a fresh run with a new seed profile**, not hot-swap the live population or mutate a running worker in place.
- Cross-demo architecture names should stay concrete and technical (`MLP`, `RandomSparse`, `NARX`, `GRU`, `LSTM`) rather than vague labels such as "advanced" or "beyond NEAT".

## API sketch

```ts
export interface BuildOptions {
 weightInit?: WeightInitializer;
 biasInit?: BiasInitializer;
 activation?: Activation;
 seed?: number;
}

export const Architect = {
 Perceptron(...sizes: number[]): Network,
 RandomSparse(input: number, hidden: number, output: number, options?: RandomSparseOptions): Network,
 NARX(input: number, hidden: number | number[], output: number, inputMemory: number, outputMemory: number, options?: BuildOptions): Network,
 LSTM(...sizesAndMaybeOptions: Array<number | LSTMOptions>): Network,
 GRU(...sizesAndMaybeOptions: Array<number | GRUOptions>): Network,
};
```

Naming should match the repo’s existing surface (if `Architect` already exists, extend it; if not, introduce a new top-level namespace carefully).

## Phase 2 closure boundary

This plan is now the intended **closing lane for Phase 2**.

Reason:

- Phase 2 is no longer only about library-internal builder correctness.
- The repo’s two flagship demos, [examples/flappy_bird/README.md](../examples/flappy_bird/README.md) and [examples/asciiMaze/README.md](../examples/asciiMaze/README.md), are deliberately treated as reference projects and end-to-end test surfaces.
- That means the builder work is not complete when the library can create these architectures in isolation. It is complete when the demos can consume the resulting builder profiles as real seed-network choices, and when those choices become part of the demo-level regression surface.

The guiding rule for this closing lane:

- if a new builder meaningfully improves one of the demos, we should use it there instead of leaving the demo on an older seed path for the sake of convenience.
- because Flappy Bird and ASCII Maze are treated as end-to-end reference demos rather than optional showcases, a builder family that is still not reference-quality in Flappy Bird does not count as Phase 2 complete.
- ASCII Maze may act as the earlier proving ground for memory-oriented profiles, but it is not a substitute for the Flappy Bird end-to-end approval gate at phase close.

## Implementation steps

### [DONE] Step 1 — MLP builder hardening

- Ensure MLP builder:
- uses explicit input/output nodes
- is fully connected
- produces a stable activation order

Acceptance:

- MLP outputs match expectations and works with both evolution and training.

### [DONE] Step 2 — Random sparse builder

- Provide a builder that creates a sparse graph with configurable counts.
- Must avoid invalid connection requests (e.g., more connections than possible) with clear errors.

Completion note:

- The public `randomSparse()` entrypoint, legacy `random()` compatibility wrapper, and constraint-validation path are landed.
- The sparse builder now accepts a public `seed` option and replays deterministic topology and parameter initialization through the owning network RNG.
- Broader downstream coverage now exercises `RandomSparse` in the generic standalone and JSON serialization architecture matrices in addition to the owner-local architect tests.

Acceptance:

- Builds quickly; respects constraints; deterministic under seed.

### [DONE] Step 3 — NARX builder

- Build memory layers using identity/constant nodes.
- Document the “clear state” behavior and when to use it.

Completion note:

- `Architect.narx(...)` continues to route its input and output delay shelves through `Layer.memory(...)`, whose blocks use identity activation, zero bias, and unit carry links.
- Architect coverage now exercises the hydrated NARX delay-line descriptors, the carried-state `clear()` guidance, and a small deterministic running-total sequence example on the public builder surface.

Acceptance:

- Works on a small sequence prediction example.

### [DONE] Step 4 — LSTM/GRU builders (pedagogical)

- Implement as explicit gated graphs using primitives.
- Provide options to toggle extra connections (e.g., input-to-output direct connections).

Completion note:

- `Architect.lstm(...)` continues to build explicit input-gate, forget-gate, memory-cell, output-gate, and output-block groups, with owner-level coverage for the direct input-to-output shortcut toggle.
- `Architect.gru(...)` now exposes the same trailing `inputToOutput` shortcut option pattern as the LSTM builder while preserving GRU's historical default topology when the option is omitted.
- GRU auxiliary nodes now stay on canonical hidden-node roles so the public GRU builder can round-trip through the strict genome boundary and preserve temporal descriptors across crossover.
- Focused coverage now exercises recurrent scheduling stability, direct-connection toggling, GRU crossover preservation, and a tiny manual-epoch recurrent training smoke test.

Acceptance:

- Produces stable graphs; can be evolved and optionally trained.

### [DONE] Step 5 — Docs and examples

- For each builder, include:
- small example
- recommended training/evolution knobs
- pitfalls (state clearing, dataset shuffling)

Completion note:

- The public `Architect` builder docs now provide source-first runnable examples for `perceptron`, `randomSparse`, `narx`, `gru`, and `lstm`.
- Each builder now documents practical starting knobs for training or evolution plus the main misuse patterns, including recurrent state resets and sequence-order handling.

Acceptance:

- Documentation is clear and runnable.

### [DONE] Step 6 — Shared demo architecture profile contract

- Introduce one explicit architecture-profile contract for demos and tests.
- A profile should include:
- stable profile id
- public builder family (`MLP`, `RandomSparse`, `NARX`, `GRU`, `LSTM`)
- size parameters resolved per demo
- whether the profile is feed-forward or recurrent
- whether the profile is currently approved for Flappy Bird, ASCII Maze, or both
- user-facing label and short explanation text
- Keep the contract shared so both demos can talk about the same families even when the exact input/output sizes differ.
- Do not let each demo invent its own architecture vocabulary once this exists.

Acceptance:

- Both demos can request a seed network through the same profile concept, and profile metadata is rich enough for UI labels, logs, and tests.

Completion note:

- A shared example architecture-profile registry now owns stable profile ids, demo-specific resolved size parameters, approval flags, family metadata, and the common builder-backed seed path.
- Flappy Bird's current default MLP seed path now resolves through that shared contract in both the trainer and worker runtime.
- ASCII Maze can now request the same shared profile concept through additive evolution-engine plumbing when a run provides `architectureProfileId`.

### [DONE] Step 7 — Flappy Bird demo integration

- Replace the current hardcoded `Architect.perceptron(...)` seed path in the Node trainer and browser worker with the shared architecture-profile contract.
- Keep architecture-selection buttons as a Flappy Bird-only control surface; do not introduce equivalent architecture buttons into ASCII Maze.
- Start with curated profiles that make sense for Flappy Bird’s current teaching story:
- `MLP` as the classic baseline
- `RandomSparse` as the first evolution-friendly topology alternative when stable
- `NARX` as the first stateful-memory candidate if recurrent/state-reset semantics remain easy to explain in the current worker/playback model
- Treat `GRU` and `LSTM` as gated Flappy deliverables within this phase, not as optional post-phase extras.
- If `GRU` or `LSTM` remain too opaque or unstable for Flappy Bird’s browser label, worker playback, restart semantics, or fresh-run selection flow, then this Phase 2 lane is still open.
- ASCII Maze may validate them earlier, but that earlier success does not close the phase unless Flappy Bird also reaches reference-quality behavior for the same public families.
- Add a small architecture control group in the lower-right stats area only if it behaves as a **new-run selector**:
- clicking a profile stops the current session,
- starts a fresh worker-backed run with the selected seed profile,
- resets population state and telemetry for that run,
- updates the HUD and architecture labels to show the selected family clearly.
- Under each Flappy Bird architecture button, render that architecture's historical best score in a small contrasting font when a record exists; render nothing when that architecture has never been run in the browser.
- Persist those per-architecture historical max scores in browser local storage so repeated runs across sessions can compete against the stored record for that specific family.
- Mark the architecture whose stored historical best is currently highest with a `*` suffix in the button label so the browser UI exposes the leading family at a glance.
- Do **not** implement this as a live switch that mutates the currently running population or swaps one genome family under an active round.
- Keep the labels concrete. Prefer `MLP`, `Sparse`, `NARX`, `GRU`, `LSTM` or similarly exact names over `classic`, `advanced`, or `beyond NEAT` in the final UI.

Completion note:

- The Flappy browser host now exposes shared-profile fresh-run selector buttons for `MLP`, `Sparse`, `NARX`, `GRU`, and `LSTM`, with neon-outline hover treatment and no live population hot-swap path.
- Browser worker initialization and generation-ready payloads now carry the selected `architectureProfileId`, and the runtime HUD keeps the chosen family visible across startup, playback, and restart flows.
- The selector stores per-profile browser-local best pipe scores, renders per-button captions only when a record exists, and marks the current browser leader with a `*` suffix.
- Flappy evaluation and playback now clear carried recurrent network state at rollout and fresh-session boundaries so stateful profiles restart deterministically.

Acceptance:

- Flappy Bird can start a clean new run from multiple curated architecture profiles, the selected profile is visible in the host UI, worker init path, and runtime stats, and the browser UI preserves per-architecture best scores locally with a visible leader marker.

### [DONE] Step 8 — ASCII Maze demo integration

- Add the same architecture-profile contract to the ASCII Maze evolution engine so population seeding can come from approved builders rather than only raw input/output seeding or externally injected networks.
- Keep ASCII Maze focused on library-backed seed selection, telemetry, curriculum, and polish improvements rather than mirroring Flappy Bird's button-driven architecture chooser.
- Support builder-backed profiles in:
- fresh evolution starts
- curriculum warm starts
- optional best-network seeding
- telemetry and archive metadata
- Preserve the current ability to inject explicit external populations or best networks, but make builder-backed profiles the default path when a demo run is not resuming from prior artifacts.
- Use ASCII Maze as the stronger earlier proving ground for memory-oriented profiles while Flappy semantics are still being stabilized, but do not treat that earlier proof as sufficient for Phase 2 closure.

Prerequisite implementation note:

- Additive `architectureProfileId` plumbing already exists in the evolution-engine and NEAT setup path.
- The remaining work is to make builder-backed profiles the default fresh-start path, carry profile identity through telemetry/archive metadata, and thread the same contract through curriculum and warm-start surfaces.

Acceptance:

- ASCII Maze can start from builder-backed profiles without custom demo-only wiring, and run metadata preserves which architecture family seeded the curriculum.

### [DONE] Step 9 — Cross-demo e2e matrix

- Add demo-level regression coverage that treats builder-backed seed profiles as part of the supported public story.
- Minimum matrix for closing this lane:
- `MLP` profile exercised by both Flappy Bird and ASCII Maze
- one non-trivial alternative profile exercised by both demos when feasible, or by the single demo that best fits it when semantics differ materially
- `GRU` and `LSTM` are not counted as Phase 2-complete families unless they are exercised at reference quality in Flappy Bird as well as in any earlier proving surface used during stabilization
- at least one stateful profile exercised by both flagship demos before the lane closes, with Flappy Bird treated as the final approval gate rather than an optional follow-on
- Verify not just boot success, but also:
- explicit I/O role preservation
- deterministic builder labeling in demo telemetry/HUD surfaces
- evolution startup compatibility
- serialization or restart compatibility where the demo already supports it
- Prefer demo tests that fail because the library builder contract changed, not because a one-off demo helper drifted.

Prerequisite implementation note:

- Focused contract coverage already exists for the shared profile registry plus the current Flappy trainer/worker and ASCII Maze consumer seams.
- The remaining work is the real demo-level matrix across approved profiles, startup paths, telemetry labeling, and stateful-family approval gates.

Acceptance:

- Builder regressions surface in demo-level tests quickly enough that the demos meaningfully extend the library’s end-to-end coverage.

### [DONE] Step 10 — Demo documentation and comparison guidance

- Update both demo READMEs to explain which architecture profiles they expose and why.
- Document which architecture families are approved as:
- baseline reference profiles
- advanced but still reference-quality profiles
- deferred or experimental profiles not yet surfaced in the public demo UI
- Keep the teaching story honest:
- Flappy Bird is still primarily a fast control-system demo,
- ASCII Maze is still primarily a compact navigation-and-shaping demo,
- architecture profiles expand those reference surfaces rather than replacing their core lessons.

Acceptance:

- A user can tell which architecture families each demo is intentionally showcasing and why those families belong there.

## Testing strategy

- Snapshot structural tests:
- node/edge counts for known sizes
- roles for I/O nodes
- deterministic output under seed
- Runtime sanity tests:
- XOR for MLP
- tiny sequence task for NARX/GRU/LSTM (1–2 minutes max)
- Demo contract tests:
- Flappy trainer and worker can start from approved builder-backed profiles
- ASCII Maze can start from approved builder-backed profiles
- demo telemetry or HUD surfaces report the selected architecture family consistently
- profile selection in Flappy Bird restarts a fresh run instead of mutating live state
- Flappy Bird persists per-architecture browser best scores in local storage, omits empty score captions for never-run families, and marks the leading stored family with a `*`
- Cross-demo end-to-end checks:
- the same profile family can be resolved into demo-specific network sizes without changing its semantic identity
- at least one builder-backed profile beyond the baseline `MLP` becomes part of the regular demo regression surface

## Risks and mitigations

- Risk: user expects “framework-grade” LSTM/GRU.
- Mitigation: label as pedagogical/evolution-friendly; show when to prefer simpler NARX.
- Risk: recurrent graphs break activation assumptions.
- Mitigation: require recurrent-mode construction semantics and stable ordering.
- Risk: Flappy Bird architecture buttons become demo spectacle instead of a trustworthy reference control.
- Mitigation: only use them as explicit new-run selectors for curated profiles, label the profiles by exact architecture family, and keep the browser score history narrowly scoped to per-architecture bests stored in local storage rather than turning the control into a broader progression system.
- Risk: demo integrations drift into custom seed factories that bypass the public builder surface.
- Mitigation: centralize demo seed profiles on the same `Architect`-backed contract and test them at the demo boundary.
- Risk: memory-oriented architectures muddy Flappy Bird’s current feed-forward teaching story too early.
- Mitigation: use ASCII Maze as the earlier proving ground when needed, but keep Flappy Bird as the mandatory end-to-end closure gate so the phase cannot close with stateful families that still fail the main browser reference demo.

## Success criteria

- Users can create common architectures with one line.
- Builders are deterministic and documented.
- Architectures integrate with evolution and training without special cases.
- Flappy Bird and ASCII Maze both consume builder-backed seed profiles as reference-project features rather than demo-local exceptions.
- `GRU` and `LSTM` do not count as done for this phase unless Flappy Bird can expose them at reference quality with legible labels, worker playback, restart behavior, and fresh-run selection semantics.
- Flappy Bird's architecture selector remains Flappy-specific and shows durable per-architecture local records without leaking that UI pattern into ASCII Maze.
- Demo-level regression coverage meaningfully expands the end-to-end surface for the architecture builders.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Active tracker: plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md.
This plan is [WIP]. Completed coverage: Step 1 MLP builder hardening is done; Step 2 RandomSparse builder hardening is done, including seeded replay and downstream serialize/standalone coverage; Step 3 NARX builder hardening is done, including explicit delay-line coverage, clear-state guidance, and a deterministic sequence example; Step 4 LSTM/GRU builder hardening is done, including GRU shortcut-option parity, GRU crossover compatibility, and recurrent training smoke coverage; Step 5 docs and examples is done, including runnable source-first builder examples plus practical knob and pitfall guidance across the public Architect presets; Step 6 shared example architecture-profile contract is done; Step 7 Flappy integration is done, including browser-side profile selection, explicit worker/HUD profile labels, per-profile local best-score tracking, and recurrent-state reset hardening; ASCII Maze has additive architectureProfileId plumbing into the evolution engine and NEAT setup.
Current active frontier: Step 8 default ASCII profile rollout and telemetry/archive metadata. Next narrow task: make builder-backed profiles the default fresh-start path in ASCII Maze, carry profile identity through telemetry/archive metadata, and thread the same contract through curriculum and warm-start surfaces.
Required validations: targeted Jest for the touched ASCII Maze and shared profile surfaces, then npm run build and npm run docs.
Worktree caution: the repo may contain unrelated ongoing changes; do not revert or normalize files outside this boundary.
```
