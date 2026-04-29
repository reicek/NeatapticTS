# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build (required before running tests)
npm run build               # webpack + tsc
npm run build:ts            # tsc only (faster for type checking)

# Type-check without emitting
npx tsc --noEmit -p tsconfig.json
npx tsc --noEmit -p tsconfig.test.json

# Test
npm test                    # full suite with coverage (runs build first)
npm run test:silent         # same, silent output — preferred for coverage analysis
npm run jest:base -- --testPathPattern="src/neat/mutation" --no-cache   # single file or folder

# Lint
npm run lint
npm run lint:fix

# Format
npm run prettier

# Docs (regenerates all src/**/README.md from JSDoc — run after JSDoc changes)
npm run docs
```

> `npm test` triggers `pretest: npm run build`. For iterating on a focused tranche, run `npx jest --config=jest.config.mjs --no-cache --testPathPattern=<path>` directly to skip the rebuild.

## Architecture

Two cooperating layers, each with deep subfolder hierarchies:

### 1. `src/architecture/` — Network graph primitives

`Network` is the central facade (orchestration-only). Heavy logic lives in subfolders:

- `activate/` — forward-pass policy (object traversal + slab fast path)
- `connect/`, `mutate/`, `remove/`, `prune/`, `topology/` — graph surgery
- `slab/` — typed-array pooling and cache-friendly activation storage
- `serialize/`, `standalone/`, `onnx/` — portability and checkpoint paths
- `training/` — gradient-descent training integration
- `deterministic/` — reproducible state for seeded runs
- `bootstrap/`, `construct/`, `genetic/` — construction and materialization pipeline

`Node`, `Connection`, `Layer`, `Group`, and `Architect` are primitive graph-building blocks at the `src/architecture/` root.

### 2. `src/neat/` — NEAT evolutionary controller

`src/neat.ts` (exported as `Neat`) is the public control desk. `src/neat/` is organized into four lanes:

| Lane | Folders |
|---|---|
| Lifecycle | `init/`, `evaluate/`, `evolve/` |
| Search pressure | `mutation/`, `selection/`, `speciation/` |
| Observability | `telemetry/`, `lineage/`, `diversity/`, `multiobjective/` |
| Reproducibility | `export/`, `rng/`, `cache/`, `maintenance/` |

`src/neat/adaptive/` owns the adaptive mutation controller. `src/neat/topology-intent/` owns feed-forward vs. recurrent policy enforcement.

### 3. `src/methods/` — Stateless algorithm objects

Activation functions, cost functions, crossover operators, selection strategies, gating and rate methods. Pure lookup/dispatch structures with no runtime state.

### 4. `src/multithreading/` — Worker evaluation

Node.js `worker_threads` backend for parallel genome evaluation. Public surface: `src/multithreading/multi.ts`.

### Public API entry point

```ts
// src/neataptic.ts — everything the library exports
export { Neat, Network, Node, Layer, Group, Connection, Architect, methods, config, multi }
```

## Module naming convention

Folder-based layout for every non-trivial module. A module `foo` inside `bar`:

```
bar/foo/
  bar.foo.ts           ← orchestration (public surface)
  bar.foo.utils.ts     ← helpers
  bar.foo.types.ts     ← interfaces and result objects
  bar.foo.errors.ts    ← error classes
  bar.foo.constants.ts ← named constants
```

Sub-modules follow the same pattern: `bar/foo/sub/bar.foo.sub.ts`, etc.

## README files — auto-generated vs manual

`src/**/README.md` files are **auto-generated artifacts** compiled from JSDoc by `npm run docs`. Do not edit them directly. To improve a generated README, improve the JSDoc in the corresponding source files, then re-run `npm run docs`.

A small number of READMEs (e.g. root `README.md`, `plans/README.md`) are manually maintained — these never have a corresponding docs-script source.

## Documentation philosophy

This is a **pedagogist-first, research-and-education library**. Documentation quality is a first-class deliverable.

**Mandatory for every folder's opening README / JSDoc chapter:**

- **Mermaid diagrams**: architecture overviews, data flows, state transitions, decision flows. Every folder-level chapter must have at least one.
- **Formulas**: include mathematical notation (LaTeX or plain Unicode) wherever an algorithm has a formal basis.
- **External references**: link to Wikipedia as the primary source. Always cite academic papers (Stanley & Miikkulainen NEAT paper, etc.) by title and URL.
- **Always credit sources** — never paraphrase a concept without attributing it. When touching existing documentation that contains un-attributed concepts, algorithms, or formulas, **add the missing attribution proactively** — do not leave prior attribution gaps behind.
- **Examples**: short, dependency-light fenced code blocks (`\`\`\`ts`) in JSDoc so the docs generator preserves them.
- **Conceptual depth**: explain the *why* — invariants, tradeoffs, failure modes, historical motivation — not just the *what*.
- **Live sandboxes**: link to runnable browser examples (`docs/` assets) wherever relevant.

Documentation style (from `copilot-instructions.md`): dark-background, blue/cyan structural lines, high-contrast labels, restrained warm neon accents — matches the Astro Bird / neon-retro-arcade aesthetic used in the Mermaid diagram theme.

Public docs should be **atemporal** — never reference plans, tracker phases, PR numbers, or internal roadmap steps. Those belong in `plans/` only.

## Current focus — test coverage to 100%

Active plan: [plans/test-repair-and-coverage.plans.md](plans/test-repair-and-coverage.plans.md)

The goal is 100% statements, branches, functions, and lines across all of `src/`. The pass works file-by-file from the lowest-covered boundary upward. The latest authoritative run is green at **296 passing suites / 2570 passing tests**. The current next target is `src/architecture/network/topology/network.topology.utils.ts` at 96.55% (28/29 lines).

Approach for each tranche:
1. Read the source boundary and its nearest existing test file.
2. Identify uncovered paths from `coverage/lcov.info` (or a focused `--coverage` run).
3. Add the smallest owner-local test that exercises the uncovered path.
4. Validate with a focused Jest slice: `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=<file>`.
5. Run `npm run test:silent` to confirm the repo-wide suite stays green.
6. Update the plan file with the completed tranche.

When a test exposes dead code, remove the dead production branch instead of writing a test to force an unreachable path.

## Skills and agents (`.github/`)

Invoke skills via slash commands — do not re-state their workflow ad hoc:

| Slash command | Use when |
|---|---|
| `/test-fix-workflow` | Repairing multiple **failing** tests (red suite) |
| `/coverage-tranche` | Expanding coverage on **passing** code toward 100% |
| `/solid-split` | SOLID-based module refactor |
| `/tracker-handoff` | Creating, compressing, or closing `.plans.md` / `.logs.md` files |
| `/educational-docs` | JSDoc quality, README tone, Mermaid diagrams, citations |
| `/plan-alignment` | Aligning changes with roadmap intent |
| `/flappy-architecture-polish` | Tuning Flappy Bird architecture profiles |
| `/architecture-builder` | Adding or extending preconfigured builders (MLP/LSTM/GRU/NARX) |
| `/onnx-work` | Extending or hardening ONNX export/import (Phase 6 parallel lane) |
| `/performance-optimization` | Memory/slab/typed-array improvements (Phase 5 parallel lane) |
| `/browser-build` | ESM/IIFE bundle configuration and CDN distribution (Phase 3) |
| `/trace-audit-reporting` | Analyzing Chrome/Perfetto traces and producing performance reports |
| `/trace-analyzer-extension` | Extending `scripts/analyze-trace/analyze-trace.ts` with new rollups |

**Companion agents** — do read-only recon then hand off into the relevant skill:

| Agent | Hands off to |
|---|---|
| `Boundary Mapper` | `solid-split` |
| `Docs Scout` | `educational-docs` |
| `Plan Scout` | `plan-alignment` |
| `Coverage Scout` | `coverage-tranche` |

## Tracker and plans conventions

`plans/` is the active tracker surface. `[PLANNED]`/`[WIP]`/`[DONE]` markers + a `Handoff query` section enable session continuity. `plans/completed/` holds archived closed trackers.

`plans/Roadmap.md` is the sequencing authority across all plans. `plans/README.md` maps triggers to plan files.

Before touching architecture, major refactors, or new subsystems, check `plans/README.md` and `plans/Roadmap.md`.

## Key code style rules

- **ES2023-first**: `toSorted`, `toReversed`, `.at(-1)`, `structuredClone`, `??`, `?.`, numeric separators. No in-place `sort`/`reverse`, no `JSON.parse(JSON.stringify())`, no `Object.assign` for clones.
- **TDD order**: red → implement → green → coverage expansion. No broad suite runs until the active cluster is green.
- **Single expect per test**: each `it()` has exactly one top-level `expect(...)`.
- **Descriptive names**: no single-letter locals except `i`/`j` in trivial loops.
- **Orchestration-first**: top-level exported functions are declarative steps calling small SRP helpers defined below the fold.
- **No `any`/`unknown`** in `src/`, `testing/`, `benchmarks/`, or `examples/` without an eslint-disable comment and short justification.
- **Cognitive complexity**: keep helpers small, pure, and single-responsibility. Declarative `collect → transform → fold` over nested control flow.

## Discovery order for non-trivial tasks

1. Nearest folder `README.md` (generated, compressed overview)
2. Parent folder `README.md` when the task spans sibling areas
3. `plans/README.md` → `plans/Roadmap.md` for roadmap alignment
4. Specific source files

## Certainty and investigation thresholds

- End every user-facing response with `(Certainty: NN%)`
- Below 90%: stop and investigate before proceeding
- Below 95%: investigate further and ask follow-up questions until requirements and environment are clear enough

## Low context window mitigation

When a change requires more context than is currently available:
- Update the relevant source plan document with a `NEXT:` item describing the change and the reason, so future work in that area has more context.
- Provide a handoff prompt in a text-copy box with the relevant context and a clear question, so a companion agent can pick it up and investigate.

## Demo-first library gap policy

Examples in `examples/` are probes that should reveal where the public API, defaults, or runtime contracts fall short. When demo work exposes a mismatch between obvious user intent and library behavior:
- Treat the demo as evidence of a **library DX gap first**.
- Prefer fixing the library, public API, or shared runtime semantics.
- Use demo-local compensation only when the issue is genuinely demo-specific or a library fix would be unsafe for the current task.
- If a temporary demo-local workaround is unavoidable, call it out explicitly as technical debt and note the preferred library-level fix.
