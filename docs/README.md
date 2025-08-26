# NeatapticTS

<div class="badges">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2c3963.svg" alt="License: MIT"/></a>
  <a href="docs/index.html"><img src="https://img.shields.io/badge/docs-generated-2c3963.svg" alt="Docs"/></a>
</div>

<img src="https://cdn-images-1.medium.com/max/800/1*THG2__H9YHxYIt2sulzlTw.png" width="480"/>

> The unofficial modern, typed evolution of **Neataptic** – focused on *clarity you can learn from* + *features you actually tweak*.

**NeatapticTS** is a TypeScript library for *evolving* neural networks (topology + weights) using a modern NEAT core plus opt‑in research extras: multi‑objective Pareto fronts, novelty & diversity pressure, adaptive complexity budgets, operator credit assignment, lineage & performance telemetry, mixed evolution + gradient fine‑tuning, and ONNX round‑trip export.

If you loved the original Neataptic but want type safety, transparent internals, richer telemetry, reproducibility, and current Node (ES2023) ergonomics—this is the forward path.

**Tagline:** *Readable neuro‑evolution you can inspect, extend, and trust.*

Node ≥ 20 required (native ES2023 features used; no polyfill layer). Older runtimes: transpile yourself.

---

## Table of Contents

1. [Why it exists](#why-it-exists)
2. [Core feature snapshot](#core-feature-snapshot)
3. [Install & Quick start](#install--quick-start)
4. [Key concepts & workflow](#key-concepts--workflow)
5. [From original Neataptic → NeatapticTS](#from-original-neataptic--neatapticts)
6. [When to enable advanced levers](#when-to-enable-advanced-levers)
7. [Performance shorthand](#performance-shorthand)
8. [Training (optional polish phase)](#training-optional-polish-phase)
9. [ONNX interop](#onnx-interop)
10. [Documentation map](#documentation-map)
11. [Contributing](#contributing)
12. [Need help?](#need-help)

---

## Why it exists

Most evolutionary NN libs make you pick between speed and pedagogy. NeatapticTS targets *explainable power* with three pillars: (1) readable algorithms, (2) explicit experiment levers, (3) reproducible telemetry.

You get:

- Code that reads like annotated pseudocode (algorithms first, micro‑opts second).
- Every major lever surfaced: mutation dynamics, diversity / novelty, parsimony, multi‑objective trade‑offs.
- Rich telemetry (species, complexity, fronts, operators, lineage, performance) → faster intuition loops.
- Deterministic seeds + RNG snapshot/restore for trustworthy benchmarks & regression tests.
- Progressive opt‑in: begin with plain NEAT; toggle extras one by one without re‑authoring code.

If you are learning NEAT: start plain, enable only `telemetry.performance`, then add a single enhancement (e.g. multi‑objective) once you understand baseline curves.

---

## Core feature snapshot

| Theme | Highlights |
|-------|------------|
| Evolution | NEAT topology + weight evolution, speciation, crossover, structural mutations |
| Diversity | Novelty archive, motif / lineage pressures, adaptive sharing, anti‑inbreeding modes |
| Multi‑Objective | Fast non‑dominated sort (fitness + complexity + optional entropy / custom) with hypervolume telemetry |
| Parsimony | Linear or adaptive complexity budgets, pruning & phased simplify cycles |
| Adaptation | Self‑adaptive mutation rates/amounts, operator bandit / success weighting |
| Telemetry | Per‑gen generation log: performance, species history, fronts, complexity, lineage, operator stats, RNG snapshots |
| Training | Optional gradient fine‑tune: schedulers, advanced optimizers, mixed precision simulation, clipping, dropout / dropconnect / stochastic depth, weight noise |
| Interop | ONNX export/import (layered + experimental recurrent stubs) |
| Reproducibility | Seed + RNG snapshot/export/import, deterministic chain mode (test utility) |

---

## Install & Quick start

Install (Node 20+):

```powershell
npm install @reicek/neataptic-ts
```

Minimal evolve loop:

```ts
import { Neat } from '@reicek/neataptic-ts';

// Goal: output ~2 when input is 1
const fitness = (net: any) => {
  const y = net.activate([1])[0];
  return -(y - 2) ** 2; // higher is better (negative error)
};

const neat = new Neat(1, 1, fitness, { popsize: 30, fastMode: true, seed: 42 });

await neat.evaluate();
await neat.evolve();
console.log('best score', neat.getBest()?.score);
```

Docs (auto‑generated from JSDoc) live in `docs/` (`docs/index.html`) and mirrored summaries in `src/*/README.md`.

Add multi‑objective + novelty quickly:

```ts
import { Neat } from '@reicek/neataptic-ts';

const neatMo = new Neat(2, 1, fitness, {
  popsize: 60,
  multiObjective: { enabled: true, complexityMetric: 'nodes' },
  novelty: { enabled: true, descriptor: g => [g.nodes.length, g.connections.length], k: 8, blendFactor: 0.25 },
  seed: 7,
});
await neatMo.evaluate();
await neatMo.evolve();
console.log(neatMo.getParetoFronts()[0].length, 'front-0 size');
```

Export telemetry for plotting:

```ts
import { Neat } from '@reicek/neataptic-ts';
import { writeFileSync } from 'node:fs';

// ...after evolution
const csv = neat.exportTelemetryCSV();
writeFileSync('telemetry.csv', csv);
```

---

## Key concepts & workflow

| Term | Meaning |
|------|---------|
| Genome | Structural blueprint (nodes + connections + params). |
| Network | Executable phenotype built from a genome. |
| Speciation | Clustering by compatibility distance to protect new structure. |
| Structural mutation | Topology change: add node / connection, disable / modify link. |
| Pareto front | Non‑dominated set across active objectives. |
| Novelty archive | Stores behaviour descriptors to reward exploration. |
| Lineage depth | Max parent depth + 1; used for tracking & diversity pressure. |

**Evolution workflow (baseline):**
1. Initialise minimal population (inputs, outputs, optional seed).
2. Evaluate each genome to assign fitness (+ any objective metrics).
3. Speciate + apply selection and reproduction (crossover + structural / parameter mutation).
4. Apply optional adaptive controllers (mutation rate, complexity budget, novelty blending, operator credit).
5. Record telemetry snapshot (for CSV / JSONL export).
6. Repeat until stopping condition (target score, max generations, stagnation trigger) then optionally fine‑tune best with gradient training.

Tip: Keep a small rolling diff of telemetry CSVs between runs to spot whether a new lever actually improved early‑generation slope instead of just final score variance.

---

## From original Neataptic → NeatapticTS

| Area | Original | This project |
|------|----------|--------------|
| Language | JS (browser + Node) | TypeScript (Node 20+, modern ES) |
| Focus | Quick demos | Readability + experiment discipline |
| Diversity | Basic speciation | Speciation + novelty + lineage/motif pressures |
| Objectives | Mostly scalar fitness | Built‑in multi‑objective (fitness + complexity + custom) |
| Telemetry | Basic stats | Rich per‑gen logs & export (CSV / JSONL) |
| Reproducibility | Basic seeding | Seed + RNG state snapshot/restore |
| Interop | – | ONNX export/import |
| Training | Backprop utilities | Extended schedulers, optimizers, regularizers |

NeatapticTS is *not* a drop‑in API clone; it’s a pedagogical, typed successor. Keep the upstream repo open for historical demos & architectural helpers you may want to port.

Upstream references:

- Original project: https://github.com/wagenaartje/neataptic
- Historical docs / playground: https://wagenaartje.github.io/neataptic/

---

## When to enable advanced levers

Principle: add one lever at a time and measure its *telemetry delta* (front size, meanNodes slope, species count variance) before stacking more. Over‑activating early hides causal impact and can slow convergence.

| Situation | Try |
|-----------|-----|
| Early stagnation | `adaptiveMutation` or light `novelty` (blend 0.2) |
| Bloat (eval cost rising) | `complexityBudget` (start linear) + mild `growth` penalty |
| Need compact & accurate variants | `multiObjective.enabled` (complexityMetric:'nodes') |
| Homogeneous species | Add `novelty` + `diversityPressure` or enable lineage pressure |
| Operator monopoly | `operatorBandit` or `operatorAdaptation` |
| Reproducible benchmark | Set `seed`, export telemetry CSV, snapshot RNG state |

---

## Performance shorthand

Fast iteration recipe:

```ts
const neat = new Neat(inp, out, fit, {
  popsize: 120,
  fastMode: true,
  threads:  (require('os').cpus().length - 1),
  adaptiveMutation: { enabled: true, strategy: 'twoTier' },
  telemetry: { enabled: true, performance: true, complexity: true }
});
```

Watch `getTelemetry().at(-1).perf` for `evalMs` vs `evolveMs`.

Micro‑guide:

| Goal | First lever | Secondary |
|------|-------------|-----------|
| Eval too slow | `threads` or reduce `popsize` | Lower `lamarckianIterations` / switch `activationPrecision:'f32'` |
| Bloat w/o gain | Enable linear `complexityBudget` | Add mild `growth` penalty (e.g. 5e-5) |
| Stagnation | Adaptive mutation (`twoTier`) | Light novelty (blend 0.2) |
| Large front 0 | `adaptiveEpsilon` (if multi‑objective) | Add discriminating complexity objective later |

Advanced perf, memory pooling, deterministic chain mode, and deep tuning: see performance docs.

---

## Training (optional polish phase)

Evolve topology first, then fine‑tune weights:

```ts
const best = neat.getBest();
await best?.train(data, { iterations: 500, rate: 0.01, optimizer: 'adam',
  gradientClip: { mode: 'norm', maxNorm: 1 },
  movingAverageWindow: 5,
});
```

Features available: schedulers (cosine, warm restarts, plateau, warmup+decay), optimizers (adamw / radam / lion / adabelief / lookahead wrapper, etc.), gradient clipping (norm/percentile, layerwise variants), micro‑batch accumulation, mixed precision simulation, dropout / dropconnect / stochastic depth, weight noise, label smoothing, SNIP & magnitude pruning.

Metrics hook example:

```ts
await best?.train(data, {
  iterations: 800,
  rate: 0.01,
  optimizer: 'adamw',
  gradientClip: { mode: 'norm', maxNorm: 1 },
  movingAverageWindow: 7,
  metricsHook: m => console.log(m.iteration, m.error, m.gradNorm)
});
```

Training docs cover scheduler composition (e.g. warmup → plateau) & pruning schedules.

---

## ONNX interop

Export layered (and experimental recurrent) networks for external tooling:

```ts
import { exportToONNX, importFromONNX } from '@reicek/neataptic-ts';
const bytes = exportToONNX(best, { includeMetadata: true });
const roundTrip = importFromONNX(bytes);
```

Current scope: feed‑forward layered perceptrons + heuristic simple recurrence grouping (experimental). Arbitrary cyclic graphs not guaranteed. Constraints:

- Supported activations only (see docs); unsupported → error or fallback.
- Hidden layers must be derivable as a clean sequence (no irregular skip sets) for clean export metadata.
- Recurrent export limited to single‑step diagonal self‑recurrence heuristics.
- Fused LSTM/GRU emission still experimental; verify round‑trip before production use.

See ONNX docs for roadmap (partial connectivity, richer recurrent forms, pruning of unfused subgraphs).

---

## Documentation map

| Topic | Where |
|-------|-------|
| API reference | `docs/index.html` / `src/*/README.md` |
| Performance & parallelism | Docs: Performance section |
| Multi‑objective & fronts | Docs: Evolution Enhancements |
| Novelty / diversity | Docs: Diversity & Novelty |
| Adaptive complexity budgets | Docs: Adaptive Complexity Budget |
| Training extensions | Docs: Added Training Features |
| ONNX import/export | Docs: ONNX Import/Export |
| Telemetry export | `Neat.exportTelemetryCSV()` / JSONL |

Telemetry export helpers: `exportTelemetryCSV()`, `exportTelemetryJSONL()`, `exportParetoFrontJSONL()`, `getParetoArchive()`.

If a README section you expected is missing, it moved into focused docs to keep this front page lean.

---

## Contributing

Pull requests that improve clarity, examples, or diagnostics are welcome. Please:

1. Read `STYLEGUIDE.md` (naming, JSDoc, tests: single expect rule).
2. Add / update tests for behavior changes (`npm test`).
3. Update JSDoc in source; run `npm run docs` to regenerate outputs.
4. Keep patches focused; include brief rationale in PR description.

## License & Attribution

MIT. Builds upon ideas/code from **Neataptic** (Thomas Wagenaar / `wagenaartje`) and **Synaptic** (Juan Cazala). The original Neataptic project appears largely unmaintained; this repo is an educational, modern continuation—not an official fork.

## Need help?

Open an issue describing: (a) your goal, (b) what you tried, (c) which doc section was unclear. We prioritise improvements to learning clarity over adding opaque features.

Enjoy exploring neuro‑evolution — and read the code. :)

---

<sub>Key ES2023 features used: `Array.prototype.findLast`, `Array.prototype.toSorted`, `Object.hasOwn`, top‑level await. No polyfills; use Node ≥ 20 or transpile.</sub>

---

### Appendix

Extra deep‑dive reference material (extended operator tables, advanced tuning heuristics, internal ONNX metadata, large example narratives) lives in the generated docs (`docs/index.html`) to keep this README focused for first‑time users.

If you need something not linked here, open an issue and we can surface it.


### Minimal Experimental Loop

1. Baseline: defaults + `telemetry:{ enabled:true, performance:true, complexity:true }` for ~30 generations.
2. Diagnose: decide if evaluation or evolution phase is dominant (`perf.evalMs` vs `perf.evolveMs`).
3. Adjust ONE lever; re-run short baseline; diff telemetry outputs (CSV/JSONL).
4. After structural growth velocity slows, introduce pruning / complexity caps to consolidate.
5. Lock configuration; use seeds + `rng` snapshot to confirm reproducibility before large runs.

### Hybrid Strategy (Evolve + Train)

Use evolution to discover topology, then fine‑tune weights with gradient training (`net.train`) using advanced schedulers / optimizers. This often reduces total time vs forcing evolution to perform late fine weight adjustments.

> Tip: Export best genome, clone into a fresh `Network`, disable stochastic regularizers, and run a focused training regimen with early stopping to polish weights.

## Performance & Parallelism Tuning

This section summarizes practical knobs to accelerate evolution while preserving solution quality.

### Core Levers

| Lever                                           | Effect                         | Guidance                                                                                                                        |
| ----------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `threads`                                       | Parallel genome evaluation     | Increase until CPU saturation (watch diminishing returns > physical cores). Falls back to single-thread if workers unavailable. |
| `growth`                                        | Structural penalty strength    | Higher discourages bloating (faster eval, may limit innovation). Tune 5e-5..5e-4.                                               |
| `mutationRate` / `mutationAmount`               | Exploration breadth            | For small populations (<=10) library enforces higher defaults. Reduce when convergence noisy.                                   |
| `fastMode`                                      | Lower sampling overhead        | Use for CI or rapid iteration; disables some heavy sampling defaults.                                                           |
| `adaptiveMutation`                              | Dynamic operator pressure      | Stabilizes search; can reduce wasted evaluations.                                                                               |
| `telemetrySelect`                               | Reduce telemetry overhead      | Keep only necessary blocks (e.g. ['performance']).                                                                              |
| `lamarckianIterations` / `lamarckianSampleSize` | Local refinement vs throughput | Lower for diversity, raise for precision on stable plateaus.                                                                    |
| `maxGenerations` (asciiMaze engine)             | Safety cap                     | Prevents runaway long runs during tuning passes.                                                                                |

### Structural Complexity Caching

`network.evolve` caches per-genome complexity metrics (nodes / connections / gates). This avoids recomputing counts each evaluation when unchanged, reducing overhead for large populations or deep architectures.

### Profiling

Enable `telemetry:{ performance:true }` to capture per-generation evaluation & evolve timings. Use these to:

1. Identify evaluation bottlenecks (cost function vs structural overhead).
2. Compare single vs multi-thread scaling (expect near-linear until memory bandwidth limits).
3. Decide when to lower Lamarckian refinement iterations.

### Suggested Workflow

1. Start with `threads = physicalCores - 1` to leave OS headroom.
2. Enable performance telemetry and run a short benchmark (e.g., 30 generations).
3. Inspect mean `evalMs` / `evolveMs`; if `evalMs` dominates and scaling <70% ideal, reduce Lamarckian refinement & complexity penalty if innovation stalls.
4. Prune telemetry (`telemetrySelect`) to omit heavy blocks during large sweeps.
5. Lock seed and record baseline; adjust one lever at a time.

### Determinism Caveat

Parallel evaluation introduces nondeterministic ordering of floating-point accumulation if your cost function is stateful. For strict reproducibility benchmark with `threads=1`.

### Example Minimal Perf Config

```ts
const neat = new Neat(inputs, outputs, fitness, {
  popsize: 120,
  threads: 8,
  telemetry: { enabled: true, performance: true },
  telemetrySelect: ['performance', 'complexity'],
  adaptiveMutation: { enabled: true, strategy: 'twoTier' },
  fastMode: true,
});
```

Monitor `getTelemetry().slice(-1)[0].perf` for current timings.

### Memory Optimizations (Activation Pooling & Precision)

`Network` constructor now accepts:

```ts
new Network(input, output, {
  activationPrecision: 'f32', // default 'f64'; use float32 activations
  reuseActivationArrays: true, // reuse a pooled output buffer each forward pass
  returnTypedActivations: true, // return the pooled Float32Array/Float64Array directly (no Array clone)
});
```

Guidelines:

- Use `activationPrecision:'f32'` for large populations or inference batches where 1e-7 precision loss is acceptable.
- Enable `reuseActivationArrays` to eliminate per-call allocation of output arrays (avoid mutating returned array between passes if reusing!).
- Set `returnTypedActivations:false` (default) if consumer code expects a plain JS array; this will clone when pooling is on.
- For maximum throughput (e.g., evaluation workers), combine all three.

These options are conservative by default to preserve existing test expectations.

### Future Improvements

Planned: micro-batch evaluation, worker task stealing, SIMD/WASM kernels, adaptive Lamarckian schedules.

### Deterministic Chain Mode (Test Utility)

`config.deterministicChainMode` (default: `false`) enables a simplified deterministic variant of the `ADD_NODE` structural mutation used strictly for tests that must guarantee a predictable deep linear chain.

When enabled BEFORE constructing / mutating a `Network`:

- Each `ADD_NODE` splits the terminal connection of the current single input→…→output chain (following the first outgoing edge each step).
- The original connection is replaced by two new ones (from→newHidden, newHidden→to).
- Any alternate outgoing edges encountered along the chain are pruned to preserve strict linearity.
- A direct input→output shortcut is removed once at least one hidden node exists, ensuring depth grows.

Intended Usage:

```ts
import { config } from 'neataptic';
config.deterministicChainMode = true; // enable
const net = new Network(1, 1);
for (let i = 0; i < 5; i++) net.mutate(methods.mutation.ADD_NODE); // guaranteed 5 hidden chain
config.deterministicChainMode = false; // restore (recommended)
```

Rationale:
Standard NEAT-style evolution is stochastic and introduces branching structure; regression tests that assert a precise depth trajectory (e.g. +1 hidden per mutation) therefore need a deterministic surrogate. This flag creates a laboratory setting for examining depth growth and complexity metrics in isolation. Use it only for testing or didactic walkthroughs—real exploratory runs benefit from architectural branching, which this mode deliberately suppresses.

Invariants (enforced after each deterministic `ADD_NODE`):

1. Exactly one outgoing forward edge per non-output node along the primary chain.
2. No direct input→output edge after the first hidden node is inserted.
3. Hidden node count increments by 1 per `ADD_NODE` call (barring impossible edge cases like empty connection sets).

If you need to debug chain growth without enabling global warnings, temporarily set a bespoke flag (e.g. `config.debugDeepPath`) and add localized logging; persistent debug output has been removed to keep test noise low.

## New Evolution Enhancements

Key advanced NEAT features now included:

- Multi-objective Pareto evolution (fast non-dominated sort) with pluggable objectives.
- Runtime objective registration: add/remove custom objectives without rewriting core sorter.
- Hypervolume proxy + Pareto front size telemetry (`getTelemetry()` now includes `fronts`, optional `hv`).
- Structural entropy proxy (degree-distribution entropy) available for custom objectives.
- Adaptive complexity budget (nodes & connections) with slope‑aware growth / contraction and diversity modulation.
- Species extended history (mean complexity, novelty, compatibility, entropy) when `speciesAllocation.extendedHistory=true`.
- Diversity pressure: motif frequency rarity bonus; novelty archive blending & optional sparse pruning.
- Self-adaptive mutation rates & amounts (strategies: twoTier, exploreLow, anneal) plus operator bandit selection.
- Persistent Pareto archive snapshots (first front) retrievable via `getParetoArchive()`.
- Performance profiling (evaluation & evolution durations) opt-in via `telemetry:{ performance:true }`.
- Complexity telemetry block (mean/max nodes/connections, enabled ratio, growth deltas) via `telemetry:{ complexity:true }`.
- Optional hypervolume scalar (`hv`) for first front via `telemetry:{ hypervolume:true }`.
- Lineage tracking: telemetry `lineage` block now includes `{ parents:[id1,id2], depthBest, meanDepth, inbreeding }` when `lineageTracking` (default true). Depth accumulates as max-parent-depth+1; `inbreeding` counts self-matings in last reproduction phase.
- Auto entropy objective: enable `multiObjective:{ enabled:true, autoEntropy:true }` to add a structural entropy maximization objective automatically.
- Adaptive dominance epsilon: `multiObjective:{ adaptiveEpsilon:{ enabled:true, targetFront:~sqrt(pop), adjust:0.002 } }` tunes `dominanceEpsilon` to stabilize first-front size.
- Reference-point hypervolume (optional): supply `multiObjective.refPoint` (array or `'auto'`) for improved HV scalar (overwrites proxy when `telemetry.hypervolume` set).
- Pareto front objective vectors export via `exportParetoFrontJSONL()`.
- Extended species metrics: variance, innovation range, enabled ratio, turnover, delta complexity & score.
- Adaptive novelty archive insertion threshold (`novelty.dynamicThreshold`) to target insertion rate.
- Inactive objective pruning: `multiObjective.pruneInactive` automatically removes stagnant objectives (zero range) after a window.
- Fast mode auto-tuning: `fastMode:true` reduces heavy sampling defaults (diversity, novelty) for faster iteration.
- Adaptive target species: `adaptiveTargetSpecies` maps structural entropy to a dynamic `targetSpecies` value (feeds compatibility controller).
- Auto distance coefficient tuning: `autoDistanceCoeffTuning` adjusts excess/disjoint coefficients based on entropy deviation.
- Adaptive pruning: `adaptivePruning` gently increases sparsity toward target using live complexity metrics.
- Objective importance telemetry (`objImportance`) with per-generation range & variance per objective.
- Objective lifecycle events (`objEvents`) and ages (`objAges`).

### Choosing Which Enhancements To Enable First

| Situation                                     | Minimal Set to Try                                      | Why                                                             |
| --------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------- |
| Baseline task; no stagnation yet              | None (core NEAT only)                                   | Reduce moving parts; establish reference trajectory.            |
| Early stagnation; little structural growth    | `adaptiveMutation.twoTier`                              | Boost exploration on weak genomes while damping top half noise. |
| Unchecked bloat; eval cost rising sharply     | `complexityBudget` (linear)                             | Gentle parsimony without aggressive pruning side‑effects.       |
| Many mediocre similar species                 | `novelty` (low `blendFactor` 0.2) + `diversityPressure` | Inject behaviour + structural dispersion pressure.              |
| Multi-objective learning / parsimony required | `multiObjective.enabled` + complexity metric            | Produces Pareto front; avoids single scalar trade‑off guess.    |
| Operator efficacy unclear                     | `operatorAdaptation` or `operatorBandit`                | Allocates attempts toward empirically productive operators.     |
| Genealogical collapse (high inbreeding)       | `lineagePressure:'antiInbreeding'`                      | Penalises repeated close ancestry matings.                      |

Enable one group at a time; consult telemetry deltas (especially complexity, species count, fronts) before stacking more.

### Multi-Objective Usage

Why: Scalarising (e.g. fitness - λ\*complexity) forces an arbitrary trade‑off. Pareto sorting preserves a frontier of alternatives letting you later pick based on deployment constraints (latency, size) without rerunning evolution.

Dominance rule (simplified): A dominates B if it is >= in every objective and > in at least one (accounting for direction). Sorting groups genomes into ranks (fronts). Within a front crowding distance approximates density so truncation favours spread.

Ranking pseudo-flow:

```
fronts = []
compute dominance counts & dominated sets for each genome
F0 = genomes with dominanceCount=0
for each Fi: decrement dominanceCount of genomes they dominate; those reaching 0 form Fi+1
```

Final ordering = by `(rank asc, crowding desc, fitness desc)` where fitness tie-break is still helpful for readability.

Troubleshooting:
| Symptom | Cause | Adjustment |
| ------- | ----- | ---------- |
| Front 0 size > 60% pop | Objectives weakly discriminative | Add complexity objective later (dynamic.addComplexityAt) or enable `adaptiveEpsilon` |
| Hypervolume flat yet fitness improving | Ref point too loose / not set | Provide tighter `multiObjective.refPoint` |
| Diversity collapse inside front | Low crowding influence (rare) | Increase population or introduce entropy objective |

Default objectives (if `multiObjective.enabled`): maximize fitness (your score) + minimize complexity (`nodes` or `connections`).

Register additional objectives at runtime:

```ts
const neat = new Neat(4, 2, fitnessFn, {
  popsize: 50,
  multiObjective: { enabled: true, complexityMetric: 'nodes' },
});
// Add structural entropy (maximize)
neat.registerObjective('entropy', 'max', (g) =>
  (neat as any)._structuralEntropy(g)
);
await neat.evaluate();
await neat.evolve();
console.log(neat.getObjectives()); // [{key:'fitness',...},{key:'complexity',...},{key:'entropy',...}]
const fronts = neat.getParetoFronts(); // Array<Network[]> for leading fronts
```

Remove custom objectives:

```ts
neat.clearObjectives(); // reverts to default pair (fitness + complexity)
```

Inspect per-genome metrics:

```ts
neat.getMultiObjectiveMetrics(); // [{ rank, crowding, score, nodes, connections }, ...]
```

Telemetry front sizes & hypervolume proxy:

```ts
neat.getTelemetry().slice(-1)[0]; // { gen, best, species, hyper, fronts:[f0,f1,...] }
```

Enable complexity + hypervolume fields:

```ts
const neat = new Neat(4, 2, fit, {
  telemetry: { enabled: true, complexity: true, hypervolume: true },
});
await neat.evolve();
console.log(neat.getTelemetry().slice(-1)[0].complexity); // { meanNodes, meanConns, ... }
```

### Adaptive Complexity Budget

When: Use after confirming baseline unsupervised growth (complexity curve) is significantly super-linear w.r.t fitness gain. Avoid enabling at generation 0 unless you already know the task needs restraint (tiny evaluation budget on large population).

Heuristic settings:
| Population Size | Start / End Node Multiplier (relative to input+output) | increaseFactor | stagnationFactor |
| ---------------- | ------------------------------------------------------ | -------------- | ---------------- |
| ≤50 | 2 → 6 | 1.20 | 0.90 |
| 51–150 | 1.5 → 5 | 1.15 | 0.93 |
| >150 | 1.2 → 4 | 1.10 | 0.95 |

Interpretation:
| Telemetry Pattern | Meaning | Response |
| ----------------- | ------- | -------- |
| Budget hits ceiling frequently + fitness still rising | Budget too tight | Raise `maxNodesEnd` or relax stagnationFactor |
| Fitness flat; complexity oscillates around shrinking budget | Over-constrained | Lower growth penalty; allow wider budget |
| Complexity climbs despite stagnationFactor triggers | Improvement window too short or noise | Increase `improvementWindow` or add smoothing by larger window |

Configure an adaptive schedule that expands limits when improvement slope is positive and contracts during stagnation. This mirrors a common parsimony heuristic: permit complexification only while marginal fitness gain per added structural unit remains appreciable; otherwise bias toward consolidation to reduce evaluation cost and overfitting risk:

```ts
complexityBudget: {
	enabled:true,
	mode:'adaptive',
	maxNodesStart: input+output+2,
	maxNodesEnd: (input+output+2)*6,
	improvementWindow: 8,
	increaseFactor:1.15,
	stagnationFactor:0.93,
	minNodes: input+output+2,
	maxConnsStart: 40,
	maxConnsEnd: 400
}
```

Linear schedule alternative: set `mode:'linear'` and optionally `horizon` (generations) to interpolate from `maxNodesStart` to `maxNodesEnd`.

### Species Extended History

If `speciesAllocation.extendedHistory:true`, each generation stores per-species stats:

```ts
[
  {
    generation,
    stats: [
      {
        id,
        size,
        best,
        lastImproved,
        age,
        meanNodes,
        meanConns,
        meanScore,
        meanNovelty,
        meanCompat,
        meanEntropy,
        varNodes,
        varConns,
        deltaMeanNodes,
        deltaMeanConns,
        deltaBestScore,
        turnoverRate,
        meanInnovation,
        innovationRange,
        enabledRatio,
      },
    ],
  },
];
```

Key new fields:

- `age`: generations since species creation.
- `varNodes/varConns`: intra-species variance of nodes/connections.
- `deltaMean*` & `deltaBestScore`: per-generation change indicators.
- `turnoverRate`: fraction of members new vs previous generation.
- `innovationRange`: spread of innovation IDs inside species.
- `enabledRatio`: enabled / total connections ratio (structural activation health).

### Diversity & Novelty

Use novelty when: (a) deceptive fitness landscape causes premature convergence, (b) behavioural diversity is itself valuable, or (c) you need a reservoir of stepping stones. Keep `blendFactor` modest initially so raw fitness remains influential.

Descriptor design tips:
| Pitfall | Effect | Remedy |
| ------- | ------ | ------ |
| Too low dimensional (e.g., single scalar) | Archive saturates quickly; low discrimination | Concatenate structural + behavioural metrics |
| Highly stochastic descriptor | Noisy novelty ranking | Average descriptor over few evaluations or cache last stable |
| Overly large vector (>50 elements) | kNN cost high | Project to smaller summary (means, counts, entropy) |

Enable novelty search blending:

```ts
novelty:{ enabled:true, descriptor: g => g.nodes.map(n=>n.bias).slice(0,8), k:10, blendFactor:0.3 }
```

Enable motif rarity pressure:

```ts
diversityPressure:{ enabled:true, motifSample:25, penaltyStrength:0.05 }
```

Adaptive novelty threshold targeting an archive insertion rate:

```ts
novelty:{
	enabled:true,
	descriptor:g=>[g.nodes.length, g.connections.length],
	archiveAddThreshold:0.5,
	dynamicThreshold:{ enabled:true, targetRate:0.15, adjust:0.1, min:0.01, max:10 }
}
```

After each evaluation the threshold is nudged up/down so the fraction of inserted descriptors approximates `targetRate`.

### Inactive Objective Pruning

If you experiment with many custom objectives it is common for some to become constant (providing no ranking discrimination). Enable automatic removal of such stagnant objectives:

```ts
multiObjective:{
	enabled:true,
	objectives:[
		{ key:'fitness', direction:'max', accessor: g => g.score },
		{ key:'novelty', direction:'max', accessor: g => (g as any)._novelty }
	],
	pruneInactive:{ enabled:true, window:5, rangeEps:1e-9, protect:['fitness'] }
}
```

Mechanics:

- Each generation the range (max-min) for every objective is computed.
- If the range stays below `rangeEps` for `window` consecutive generations the objective is removed.
- Keys listed in `protect` are never removed (even if stagnant).
- Telemetry will reflect the pruned objective list from the following generation.

Use this to keep the Pareto sorter focused on informative axes.

### Fast Mode Auto-Tuning

When iterating quickly or running inside tests you can set `fastMode:true` to scale down expensive sampling defaults:

```ts
const neat = new Neat(8, 3, fitFn, {
  popsize: 80,
  fastMode: true,
  diversityMetrics: { enabled: true }, // pairSample / graphletSample auto-lowered
  novelty: {
    enabled: true,
    descriptor: (g) => [g.nodes.length, g.connections.length],
  }, // k auto-lowered if not explicitly set
});
```

Auto adjustments (only if corresponding option fields are undefined):

- `diversityMetrics.pairSample`: 40 -> 20
- `diversityMetrics.graphletSample`: 60 -> 30
- `novelty.k`: lowered to 5

The tuning runs once on first diversity stats computation. Override by explicitly supplying your own values.

Lineage depth diversity (auto when lineageTracking + diversityMetrics):
`diversity` object gains:

- `lineageMeanDepth`: mean `_depth` over population.
- `lineageMeanPairDist`: sampled mean absolute depth difference between genome pairs (structure ancestry dispersion).
  Telemetry `lineage` block now also includes:
- `ancestorUniq`: average Jaccard distance (0..1) between ancestor sets of sampled genome pairs within a small depth window (higher = more genealogical diversification).

Use these to detect genealogical stagnation (both remaining near zero) vs broad exploration (rising pair distance).

### Lineage Pressure & Anti-Inbreeding (Optional)

Apply score adjustments based on lineage structure (depth dispersion) or penalize inbreeding (high ancestor overlap):

```ts
lineagePressure:{
	enabled:true,
	mode:'antiInbreeding',   // 'penalizeDeep' | 'rewardShallow' | 'spread' | 'antiInbreeding'
	strength:0.02,           // generic scaling for depth modes
	ancestorWindow:4,        // generations to look back when computing ancestor sets
	inbreedingPenalty:0.04,  // override penalty scaling (defaults to strength*2)
	diversityBonus:0.02      // bonus scaling for very distinct parent lineages
}
```

Modes:

- `penalizeDeep`: subtracts proportional penalty `-(depth-target)*strength` when depth exceeds target.
- `rewardShallow`: adds proportional bonus for depths below target.
- `spread`: encourages dispersion around mean depth (boost far-from-mean, cap excessive depth).
- `antiInbreeding`: computes Jaccard overlap of ancestor sets of both parents (including parents) within `ancestorWindow` generations. Penalizes high overlap (>0.75) and rewards very distinct ancestry (<0.25). Penalty/bonus magnitude scales with overlap distance and the configured penalty/bonus parameters.

Runs after evaluation/speciation before sorting so it integrates with all other mechanisms.

### RNG State Snapshot & Reproducibility

When a numeric `seed` is provided, a fast deterministic PRNG drives stochastic choices. You can snapshot/restore mid-run:

```ts
const neat = new Neat(3, 1, fit, { seed: 1234 });
await neat.evolve();
const snap = neat.snapshotRNGState(); // { state }
// ... later
neat.restoreRNGState(snap);
// Serialize
const json = neat.exportRNGState();
// New instance (even without seed) can resume deterministic sequence
const neat2 = new Neat(3, 1, fit, {});
neat2.importRNGState(json);
```

Restoring installs the internal deterministic generator if it wasn't active.

### Operator Adaptation & Mutation Self-Adaptation

Operator adaptation vs bandit:
| Mechanism | Strength | Weakness | When to Prefer |
| --------- | -------- | -------- | -------------- |
| Simple adaptation (moving success window) | Low overhead; stable | Slow to reallocate after phase shift | Small pops; mild dynamics |
| UCB1 bandit (`operatorBandit`) | Balances exploration/exploitation mathematically | Needs minAttempts; slightly higher overhead | Larger pops; heterogeneous operator payoffs |

Per-genome mutation rate/amount adapt each generation under strategies (`twoTier`, `exploreLow`, `anneal`). Use:

```ts
adaptiveMutation:{ enabled:true, strategy:'twoTier', sigma:0.08, adaptAmount:true }
```

Operator success statistics (bandit + weighting):

```ts
neat.getOperatorStats(); // [{ name, success, attempts }, ...]
```

Telemetry now also records an `ops` array each generation with lightweight success/attempt counts.

### Objective Lifetime & Species Allocation Telemetry

Each telemetry entry now may include:

- `objectives`: array of active objective keys that generation (already used internally for auditing dynamic scheduling).
- `objAges`: object mapping objective key -> consecutive generations active. When an objective is removed (e.g. via pruning or dynamic delay) its age resets to 0 if later reintroduced.
- `speciesAlloc`: array of `{ id, alloc }` giving number of offspring allocated to each species in the reproduction phase that produced the current generation. Useful for diagnosing allocation fairness / starvation.
- `objEvents`: array of `{ type:'add'|'remove', key }` describing objective lifecycle changes that occurred for that generation (emitted when dynamic scheduling or pruning alters the active set).

Example:

```ts
const neat = new Neat(4, 2, fit, {
  popsize: 40,
  multiObjective: {
    enabled: true,
    autoEntropy: true,
    dynamic: { enabled: true, addComplexityAt: 2, addEntropyAt: 5 },
  },
  telemetry: { enabled: true },
});
for (let g = 0; g < 10; g++) {
  await neat.evaluate();
  await neat.evolve();
}
const last = neat.getTelemetry().slice(-1)[0];
console.log(last.objectives); // ['fitness','complexity','entropy']
console.log(last.objAges); // { fitness:10, complexity:9, entropy:5 }
console.log(last.speciesAlloc); // [{ id:1, alloc:7 }, { id:2, alloc:6 }, ...]
```

### CSV Export Columns (Extended)

`exportTelemetryCSV()` now conditionally includes the following columns when present:

- `rng` (deterministic RNG state snapshot if `rngState:true` + seed supplied)
- `ops` (JSON array of operator stats)
- `objectives` (JSON array of active objective keys)
- `objAges` (JSON object of key->age)
- `speciesAlloc` (JSON array of per-species offspring allocations)
- `objEvents` (JSON array of add/remove objective lifecycle events)
  Existing flattened sections remain: `complexity.*`, `perf.*`, `lineage.*`, `diversity.lineageMeanDepth`, `diversity.lineageMeanPairDist`, and `fronts`.

Performance profiling telemetry (optional):

````ts
const neat = new Neat(4,2, fitnessFn, { telemetry:{ enabled:true, performance:true } });
await neat.evaluate(); await neat.evolve();
const last = neat.getTelemetry().slice(-1)[0];
console.log(last.perf); // { evalMs, evolveMs }

Export telemetry:
```ts
neat.exportTelemetryJSONL(); // JSON Lines
neat.exportTelemetryCSV();   // CSV (flattened complexity/perf)
````

Pareto archive snapshots:

```ts
neat.getParetoArchive(); // [{ gen, size, genomes:[{ id, score, nodes, connections }] }, ...]
```

Species history (per-species longitudinal metrics) CSV export:

```ts
// Last ~200 generations (configurable)
const csv = neat.exportSpeciesHistoryCSV();
// Columns include generation plus dynamic keys: id,size,best,lastImproved,age,meanNodes,meanConns,meanScore,meanNovelty,meanCompat,meanEntropy,varNodes,varConns,deltaMeanNodes,deltaMeanConns,deltaBestScore,turnoverRate,meanInnovation,innovationRange,enabledRatio (when extendedHistory enabled)
```

These APIs are evolving; consult source `src/neat.ts` for full option surfaces while docs finalize.

Full option & telemetry reference: [docs/API.md](./docs/API.md)

# Network Constructor Update

The `Network` class constructor now supports an optional third parameter for configuration:

```ts
new Network(input: number, output: number, options?: { minHidden?: number })
```

- `input`: Number of input nodes (required)
- `output`: Number of output nodes (required)
- `options.minHidden`: (optional) If set, enforces a minimum number of hidden nodes. If omitted or 0, no minimum is enforced. This allows true 1-1 (input-output only) networks.

**Example:**

```ts
// Standard 1-1 network (no hidden nodes)
const net = new Network(1, 1);

// Enforce at least 3 hidden nodes
const netWithHidden = new Network(2, 1, { minHidden: 3 });
```

# Neat Evolution minHidden Option

The `minHidden` option can also be passed to the `Neat` class to enforce a minimum number of hidden nodes in all evolved networks:

```ts
import Neat from './src/neat';
const neat = new Neat(2, 1, fitnessFn, { popsize: 50, minHidden: 5 });
```

- All networks created by the evolutionary process will have at least 5 hidden nodes.
- This is useful for ensuring a minimum network complexity during neuro-evolution.

See tests in `test/neat.ts` for usage and verification.

---

# ONNX Import/Export

Interoperability layer for exchanging strictly layered MLP (and experimental recurrent) networks with ONNX tooling. Scope today: feed-forward layered perceptrons plus heuristic detection of simple recurrent gate groupings; arbitrary graphs not guaranteed.

### Basic Usage

```ts
import { exportToONNX, importFromONNX } from './src/architecture/onnx';

// Export
const onnxModel = exportToONNX(network, { includeMetadata: true });
// Persist (pseudo)
writeFileSync('model.onnx', Buffer.from(onnxModel));

// Import (round-trip)
const imported = importFromONNX(readFileSync('model.onnx'));
console.log(imported.activate(sample)[0]);
```

### Round-Trip Checklist

| Requirement                                                    | Why                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------- |
| Layered (no skip / arbitrary cross connections)                | Exporter maps consecutive layers to Gemm ops.         |
| Supported activations only (current core set)                  | Non-supported activations fall back or raise error.   |
| Consistent hidden layer sizes array derivable                  | Needed to emit `layer_sizes` metadata.                |
| No unsupported gating (other than simple recurrence heuristic) | Gate fusion heuristics limited to equal partitions.   |
| If recurrent: single-step self recurrence only                 | Multi-step / full sequence unrolling not emitted yet. |

### Import Failure Diagnostics

| Symptom                                 | Likely Cause                               | Suggested Action                                                           |
| --------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| Missing weights error                   | Non-Gemm node encountered                  | Re-export from a supported tool or simplify model.                         |
| Activation unsupported                  | Activation op not mapped                   | Replace with supported activation before export.                           |
| Metadata absent (layer_sizes undefined) | Model not produced by NeatapticTS exporter | Provide manual layer spec (future option) or re-export using this library. |
| Recurrent fuse skipped                  | Partition grouping heuristic failed        | Ensure hidden size divisible by expected gate count (5 LSTM / 4 GRU).      |

### Additional Capabilities (Phase 2 & 3 roadmap)

- Partial connectivity (enable with `{ allowPartialConnectivity:true }`) inserts zero-weight placeholders for missing edges.
- Mixed activation layers (enable `{ allowMixedActivations:true }`) are decomposed into per-neuron Gemm + Activation + Concat.
- Multi-layer self‑recurrence single-step form (enable `{ allowRecurrent:true, recurrentSingleStep:true }`) exports each recurrent hidden layer with:
  - Previous hidden state input tensor (one per recurrent hidden layer).
  - Feedforward weight matrix `Wk`, bias `Bk`, and diagonal recurrent matrix `Rk` (self-connection weights) per layer.
- Experimental fused recurrent heuristics (enable `{ allowRecurrent:true }`):
  - LSTM heuristic: detects 5-way equal partition of a hidden layer (input, forget, cell, output, output-block) and emits `LSTM_W* / LSTM_R* / LSTM_B*` plus an `LSTM` node (single-step, simplified biases and diagonal recurrence only).
  - GRU heuristic: detects 4-way equal partition (update, reset, candidate, output-block) and emits `GRU_W* / GRU_R* / GRU_B*` plus a `GRU` node.
  - Original Gemm/Activation path is retained (no pruning yet) for transparency; importer reconstructs `Layer.lstm` / `Layer.gru` instances when metadata & tensors are present.

Metadata keys (when `includeMetadata:true`):

| Key                                          | Description                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------- |
| `layer_sizes`                                | JSON array of hidden layer sizes in order                                       |
| `recurrent_single_step`                      | JSON array of 1-based hidden layer indices exported with single-step recurrence |
| `lstm_groups_stub`                           | Detected LSTM grouping stubs (pre-emission heuristic data)                      |
| `lstm_emitted_layers` / `gru_emitted_layers` | Layers where fused nodes were emitted                                           |
| `rnn_pattern_fallback`                       | Near-miss pattern info for diagnostic purposes                                  |

### Limitations / TODO

- Fused LSTM/GRU nodes are experimental: biases currently unsplit (Rb assumed zero) and recurrence limited to diagonal self-connections of memory/candidate groups.
- Gate ordering in heuristics may differ from canonical ONNX spec and will be normalized in a future pass (documented in code comments).
- Redundant unfused Gemm subgraph is not yet pruned when fused nodes are emitted.
- Only self-produced models are guaranteed to round-trip; arbitrary ONNX graphs are out of scope.

### Validation & Tests

See `test/network/onnx.export.test.ts` / `onnx.import.test.ts` for MLP & basic recurrence coverage. Fused LSTM/GRU tests pending finalization of gate ordering normalization.

---

### Further notices

NeatapticTS is based on [Neataptic](https://github.com/wagenaartje/neataptic). Parts of [Synaptic](https://github.com/cazala/synaptic) were used to develop Neataptic.

The neuro-evolution algorithm used is the [Instinct](https://medium.com/@ThomasWagenaar/neuro-evolution-on-steroids-82bd14ddc2f6) algorithm.

##### Original [repository](https://github.com/wagenaartje/neataptic) in now [unmaintained](https://github.com/wagenaartje/neataptic/issues/112)

---

## Added Training Features

- Learning rate schedulers: fixed, step, exp, inv, cosine annealing, cosine annealing w/ warm restarts, linear warmup+decay, reduce-on-plateau.
- Regularization (L1, L2, custom function), dropout, DropConnect.
- Per-iteration `metricsHook` exposing `{ iteration, error, gradNorm }`.
- Checkpointing (`best`, `last`) via `checkpoint.save` callback.
- Advanced optimizers: `sgd`, `rmsprop`, `adagrad`, `adam`, `adamw`, `amsgrad`, `adamax`, `nadam`, `radam`, `lion`, `adabelief`, and `lookahead` wrapper.
- Gradient improvements: per-call gradient clipping (global / layerwise, norm or percentile), micro-batch gradient accumulation (`accumulationSteps`) independent of data `batchSize`, optional mixed precision (loss-scaled with dynamic scaling) training.

### Gradient Improvements

Gradient clipping (optional):

```ts
net.train(data, {
  iterations: 500,
  rate: 0.01,
  optimizer: 'adam',
  gradientClip: { mode: 'norm', maxNorm: 1 },
});
net.train(data, {
  iterations: 500,
  rate: 0.01,
  optimizer: 'adam',
  gradientClip: { mode: 'percentile', percentile: 99 },
});
// Layerwise variants: 'layerwiseNorm' | 'layerwisePercentile'
```

Micro-batch accumulation (simulate larger effective batch without increasing memory):

```ts
// Process 1 sample at a time, accumulate 8 micro-batches, then apply one optimizer step
net.train(data, {
  iterations: 100,
  rate: 0.005,
  batchSize: 1,
  accumulationSteps: 8,
  optimizer: 'adam',
});
```

If `accumulationSteps > 1`, gradients are averaged before the optimizer step so results match a single larger batch (deterministic given same sample order).

Mixed precision (simulated FP16 gradients with FP32 master weights + dynamic loss scaling):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.01,
  optimizer: 'adam',
  mixedPrecision: { lossScale: 1024 },
});
```

Behavior:

- Stores master FP32 copies of weights/biases (`_fp32Weight`, `_fp32Bias`).
- Scales gradients during accumulation; unscales before clipping / optimizer update; adjusts `lossScale` down on overflow (NaN/Inf), attempts periodic doubling after sustained stable steps (configurable via `mixedPrecision.dynamic`).
- Raw gradient norm (pre-optimizer, post-scaling/clipping) exposed via metrics hook as `gradNormRaw` (legacy post-update norm still `gradNorm`).
- Pure JS numbers remain 64-bit; this is a functional simulation for stability and future WASM/GPU backends.

Clipping modes:
| mode | scope | description |
|------|-------|-------------|
| norm | global | Clip global L2 gradient norm to `maxNorm`. |
| percentile | global | Clamp individual gradients above given percentile magnitude. |
| layerwiseNorm | per layer | Apply norm clipping per architectural layer (fallback per node if no layer info). Optionally splits weight vs bias groups via `gradientClip.separateBias`. |
| layerwisePercentile | per layer | Percentile clamp per architectural layer (fallback per node). Supports `separateBias`. |

Notes:

- Provide either `{ mode, maxNorm? , percentile? }` or shorthand `{ maxNorm }` / `{ percentile }`.
- Percentile ranking is magnitude-based.
- Accumulation averages gradients; to sum instead (rare) scale `rate` accordingly.
- Dynamic loss scaling heuristics: halves on detected overflow, doubles after configurable stable steps (default 200) within bounds `[minScale,maxScale]`.
- Config: `mixedPrecision:{ lossScale:1024, dynamic:{ minScale:1, maxScale:131072, increaseEvery:300 } }`.
- Accumulation reduction: default averages gradients; specify `accumulationReduction:'sum'` to sum instead (then adjust learning rate manually, e.g. multiply by 1/accumulationSteps if you want equivalent averaging semantics).
- Layerwise clipping: set `gradientClip:{ mode:'layerwiseNorm', maxNorm:1, separateBias:true }` to treat biases separately from weights.
- Two gradient norms tracked: raw (pre-update) and legacy (post-update deltas). Future APIs may expose both formally.
- Access stats: `net.getTrainingStats()` -> `{ gradNorm, gradNormRaw, lossScale, optimizerStep, mp:{ good, bad, overflowCount, scaleUps, scaleDowns, lastOverflowStep } }`.
- Test hook (not for production): `net.testForceOverflow()` forces the next mixed-precision step to register an overflow (used in unit tests to validate telemetry paths).
- Gradient clip grouping count: `net.getLastGradClipGroupCount()` (useful to verify separateBias effect).
- Rate helper for accumulation: `const adjRate = Network.adjustRateForAccumulation(rate, accumulationSteps, accumulationReduction)`.
- Deterministic seeding: `new Network(4,2,{ seed:123 })` or later `net.setSeed(123)` ensures reproducible initial weights, biases, connection order, and mutation randomness for training (excluding certain static reconstruction paths). For NEAT evolution pass `seed` in `new Neat(...,{ seed:999 })`.
- Overflow telemetry: during mixed precision training, `overflowCount` increments on detected NaN/Inf when unscaling gradients; `scaleUps` / `scaleDowns` count dynamic loss scale adjustments.

### Learning Rate Scheduler Usage

```ts
import methods from './src/methods/methods';
const ratePolicy = methods.Rate.cosineAnnealingWarmRestarts(200, 1e-5, 2);
net.train(data, { iterations: 1000, rate: 0.1, ratePolicy });
```

Reduce-on-plateau automatically receives current error because `train` detects a 3-arg scheduler:

```ts
const rop = methods.Rate.reduceOnPlateau({
  patience: 20,
  factor: 0.5,
  minRate: 1e-5,
});
net.train(data, { iterations: 5000, rate: 0.05, ratePolicy: rop });
```

#### Early Stopping Extensions

You can enable moving-average smoothing and independent early-stop patience separate from any scheduler plateau logic. You can also give the learning-rate scheduler (e.g. reduce-on-plateau) its OWN smoothing configuration if you want it to react differently than early stopping:

```ts
net.train(data, {
	rate: 0.05,
	iterations: 10000,
	error: 0.02,                // target threshold (checked on SMOOTHED early-stop error)
	movingAverageType: 'median',
	movingAverageWindow: 7,     // EARLY STOP smoothing (robust to spikes)
	earlyStopPatience: 25,
	earlyStopMinDelta: 0.0005,

	---
	// Separate plateau smoothing: scheduler sees a faster EMA over shorter horizon
	plateauMovingAverageType: 'ema',
	plateauMovingAverageWindow: 3,
	plateauEmaAlpha: 0.6,       // (otherwise defaults to 2/(N+1))
	ratePolicy: methods.Rate.reduceOnPlateau({ patience: 10, factor: 0.5, minRate: 1e-5 })
});
```

Behavior details:
Smoothing types (set `movingAverageType`):

| Type         | Description                                                                                            | Key Params                                             | When to Use                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| sma          | Simple moving average over window                                                                      | movingAverageWindow                                    | General mild smoothing                                       |
| ema          | Exponential moving average                                                                             | emaAlpha (default 2/(N+1))                             | Faster reaction than SMA                                     |
| adaptive-ema | Dual-track variance-adaptive EMA (takes min of baseline & adaptive for guaranteed non-worse smoothing) | emaAlpha, adaptiveAlphaMin/Max, adaptiveVarianceFactor | Volatile early phase responsiveness with stability guarantee |
| wma          | Linear weighted (recent heavier)                                                                       | movingAverageWindow                                    | Slightly more responsive than SMA                            |
| median       | Moving median                                                                                          | movingAverageWindow                                    | Spike / outlier robustness                                   |
| trimmed      | Trimmed mean (drops extremes)                                                                          | trimmedRatio (0-0.5)                                   | Moderate outliers; keep efficiency                           |
| gaussian     | Gaussian-weighted window (recent emphasized smoothly)                                                  | gaussianSigma (default N/3)                            | Want smooth taper vs linear weights                          |

Additional options:

- trimmedRatio: fraction trimmed from each side for 'trimmed' (default 0.1)
- gaussianSigma: width for 'gaussian' (default window/3)
- trackVariance: true to compute running variance & min (exposed via metricsHook)
- adaptiveAlphaMin / adaptiveAlphaMax / adaptiveVarianceFactor for adaptive-ema control

Metrics hook additions when smoothing active: `rawError`, `runningVariance`, `runningMin` (if trackVariance true) alongside smoothed `error`.

- Target `error` comparison and improvement tracking both use the smoothed value.
- `earlyStopPatience` counts iterations since the last smoothed improvement > `earlyStopMinDelta`.
- Plateau smoothing: provide any of `plateauMovingAverageWindow`, `plateauMovingAverageType`, `plateauEmaAlpha`, `plateauTrimmedRatio`, `plateauGaussianSigma`, `plateauAdaptiveAlphaMin/Max`, `plateauAdaptiveVarianceFactor`.
  - If none are supplied the scheduler reuses the early-stop smoothing.
  - If supplied, metricsHook receives an extra field `plateauError` (the separately smoothed value supplied to the scheduler), while `error` remains the early-stop smoothed value.
  - Plateau adaptive-ema uses the same dual-track min(adaptive, baseline) logic.
- This does not interfere with `Rate.reduceOnPlateau` patience; they are independent.

#### Scheduler Reference

| Scheduler             | Factory Call                                                        | Key Params                                      | Behavior                                                                                    |
| --------------------- | ------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------- |
| fixed                 | `Rate.fixed()`                                                      | –                                               | Constant learning rate.                                                                     |
| step                  | `Rate.step(gamma?, stepSize?)`                                      | gamma (default 0.9), stepSize (default 100)     | Multiplies rate by `gamma` every `stepSize` iterations.                                     |
| exp                   | `Rate.exp(gamma?)`                                                  | gamma (default 0.999)                           | Exponential decay: `rate * gamma^t`.                                                        |
| inv                   | `Rate.inv(gamma?, power?)`                                          | gamma (0.001), power (2)                        | Inverse time decay: `rate / (1 + γ * t^p)`.                                                 |
| cosine annealing      | `Rate.cosineAnnealing(period?, minRate?)`                           | period (1000), minRate (0)                      | Cosine decay from base to `minRate` each period.                                            |
| cosine warm restarts  | `Rate.cosineAnnealingWarmRestarts(initialPeriod, minRate?, tMult?)` | initialPeriod, minRate (0), tMult (2)           | Cosine cycles with period multiplied by `tMult` after each restart.                         |
| linear warmup + decay | `Rate.linearWarmupDecay(totalSteps, warmupSteps?, endRate?)`        | totalSteps, warmupSteps (auto 10%), endRate (0) | Linear ramp to base, then linear decay to `endRate`.                                        |
| reduce on plateau     | `Rate.reduceOnPlateau(opts)`                                        | patience, factor, minRate, threshold            | Monitors error; reduces current rate by `factor` after `patience` non-improving iterations. |

Notes:

- All scheduler factories return a function `(baseRate, iteration)` except `reduceOnPlateau`, which returns `(baseRate, iteration, error)`; `train` auto-detects and supplies `error` if the function arity is 3.
- You can wrap or compose schedulers—see below for a composition pattern.

#### Evolution Warning

`evolve()` now always emits a warning `"Evolution completed without finding a valid best genome"` when no suitable genome is selected (including zero-iteration runs) to aid testability and user feedback.

#### Composing Schedulers (Warmup then Plateau)

You can combine policies by writing a small delegator that switches logic after warmup completes and still passes error when required:

```ts
import methods from './src/methods/methods';

const warmup = methods.Rate.linearWarmupDecay(500, 100, 0.1); // only use warmup phase portion
const plateau = methods.Rate.reduceOnPlateau({
  patience: 15,
  factor: 0.5,
  minRate: 1e-5,
});

// Hybrid policy: first 100 steps use linear ramp; afterward delegate to plateau (needs error)
const hybrid = (base: number, t: number, err?: number) => {
  if (t <= 100) return warmup(base, t); // ignore decay tail by cutting early
  // plateau expects error (3-arg); train will pass it because we define length >= 3 when we use 'err'
  return plateau(base, t - 100, err!); // shift iteration so plateau's patience focuses on post-warmup
};

net.train(data, { iterations: 2000, rate: 0.05, ratePolicy: hybrid });
```

For more elaborate chaining (e.g., staged cosine cycles then plateau), follow the same pattern: evaluate `t`, decide which inner policy to call, adjust `t` relative to that stage, and pass along `error` if the target policy needs it.

### Metrics Hook & Checkpoints

```ts
net.train(data, {
  iterations: 800,
  rate: 0.05,
  metricsHook: ({ iteration, error, gradNorm }) =>
    console.log(iteration, error, gradNorm),
  checkpoint: {
    best: true,
    last: true,
    save: ({ type, iteration, error, network }) => {
      /* persist */
    },
  },
});
```

### DropConnect

```ts
net.enableDropConnect(0.3);
net.train(data, { iterations: 300, rate: 0.05 });
net.disableDropConnect();
```

### Advanced Optimizers

Supply `optimizer` to `train` as a simple string (uses defaults) or a config object.

Basic (defaults):

```ts
net.train(data, { iterations: 200, rate: 0.01, optimizer: 'adam' });
```

Custom AdamW:

```ts
net.train(data, {
  iterations: 500,
  rate: 0.005,
  optimizer: {
    type: 'adamw',
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.01,
  },
});
```

Lion (sign-based update):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.001,
  optimizer: { type: 'lion', beta1: 0.9, beta2: 0.99 },
});
```

Adamax (robust to sparse large gradients):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.002,
  optimizer: { type: 'adamax', beta1: 0.9, beta2: 0.999 },
});
```

NAdam (Nesterov momentum style lookahead on first moment):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.001,
  optimizer: { type: 'nadam', beta1: 0.9, beta2: 0.999 },
});
```

RAdam (more stable early training variance):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.001,
  optimizer: { type: 'radam', beta1: 0.9, beta2: 0.999 },
});
```

AdaBelief (faster convergence / better generalization via surprise-based variance):

```ts
net.train(data, {
  iterations: 300,
  rate: 0.001,
  optimizer: { type: 'adabelief', beta1: 0.9, beta2: 0.999 },
});
```

Lookahead (wraps a base optimizer; performs k fast steps then interpolates):

```ts
net.train(data, {
  iterations: 400,
  rate: 0.01,
  optimizer: { type: 'lookahead', baseType: 'adam', la_k: 5, la_alpha: 0.5 },
});
```

Optimizer reference (choose based on signal quality & overfitting risk):

| Optimizer | Key Params (in object)         | Notes                                                      |
| --------- | ------------------------------ | ---------------------------------------------------------- |
| sgd       | momentum                       | Nesterov momentum internally in propagate when update=true |
| rmsprop   | eps                            | Uses fixed decay 0.9 / 0.1 split for cache                 |
| adagrad   | eps                            | Cache accumulates squared grads (monotonic)                |
| adam      | beta1, beta2, eps              | Standard bias correction                                   |
| adamw     | beta1, beta2, eps, weightDecay | Decoupled weight decay applied after adaptive step         |
| amsgrad   | beta1, beta2, eps              | Maintains max second-moment vhat                           |
| adamax    | beta1, beta2, eps              | Infinity norm (u) instead of v                             |
| nadam     | beta1, beta2, eps              | Nesterov variant of Adam                                   |
| radam     | beta1, beta2, eps              | Rectifies variance early in training                       |
| lion      | beta1, beta2                   | Direction = sign(beta1*m + beta2*m2)                       |
| adabelief | beta1, beta2, eps              | Second moment of (g - m) (gradient surprise)               |
| lookahead | baseType, la_k, la_alpha       | Interpolates toward slow weights every k steps             |

General guidance:

- Start with `adam` (stable) or `adamw` (when you need explicit weight decay to fight drift).
- Use `radam` when very small batch / high variance early phases cause Adam instability.
- Try `adabelief` if convergence stalls yet gradients still fluctuate (captures surprise better).
- `lion` can yield sparse-like updates and sometimes better generalization; may need slightly lower base rate.
- Wrap with `lookahead` to smooth noisy fast optimizers (tune `la_k` up if oscillating, down if too sluggish).
- Step counter resets each `train` call (t=1) for reproducibility; schedule functions see this reset.
- `lookahead.baseType` defaults to `adam` and cannot nest another lookahead.
- Only AdamW applies decoupled `weightDecay`; for others add L2 via existing regularization mechanisms if needed.
- Adamax may help with sparse or bursty gradients (uses infinity norm).\
- NAdam can yield slightly faster early progress due to lookahead on m.\
- RAdam mitigates the need for warmup; behaves like Adam after variance rectification threshold.\
- AdaBelief can reduce over-adaptation to noisy gradients by modeling belief deviation.\
- Lion performs well in some large-scale settings due to sign-based memory efficiency.

### Label Smoothing

Cross-entropy with label smoothing discourages over-confident predictions.

```ts
import methods from './src/methods/methods';
const loss = methods.Cost.labelSmoothing([1, 0, 0], [0.8, 0.1, 0.1], 0.1);
```

### Weight Noise

Adds zero-mean Gaussian noise to weights on each _training_ forward pass (inference unaffected). Original weights are restored immediately after the pass (noise is ephemeral / non-destructive).

Basic usage:

```ts
net.enableWeightNoise(0.05); // global stdDev = 0.05
net.train(data, { iterations: 100, rate: 0.05 });
net.disableWeightNoise();
```

Per-hidden-layer std deviations (layered networks only):

```ts
const layered = Architect.perceptron(4, 32, 16, 8, 2); // input, 3 hidden, output
layered.enableWeightNoise({ perHiddenLayer: [0.05, 0.02, 0.0] }); // third hidden layer noiseless
```

Dynamic schedule (e.g. cosine decay) for global std:

```ts
net.enableWeightNoise(0.1); // initial value (will be overridden by schedule each step)
net.setWeightNoiseSchedule((step) => 0.1 * Math.cos(step / 500));
```

Deterministic seeding / custom RNG (affects dropout, dropconnect, stochastic depth, weight noise sampling):

```ts
net.setSeed(42); // reproducible stochastic regularization
// or provide a custom RNG
net.setRandom(() => myDeterministicGenerator.next());
```

Diagnostics:

- Each connection temporarily stores last noise in `connection._wnLast` (for test / inspection).
- Global training forward pass count: `net.trainingStep`.
- Last skipped layers (stochastic depth): `net.lastSkippedLayers`.
- Regularization statistics (after any forward): `net.getRegularizationStats()` returns
  `{ droppedHiddenNodes, totalHiddenNodes, droppedConnections, totalConnections, skippedLayers, weightNoise: { count, sumAbs, maxAbs, meanAbs } }`.
- To combine with DropConnect / Dropout the sampling is independent (noise applied before masking).

Gotchas:

- Per-layer noise ignores global schedule (schedule currently applies only when using a single global std). If you need both, emulate by updating `enableWeightNoise` each epoch.
- Very large noise (> weight scale) can destabilize gradients; start small (e.g. 1-10% of typical weight magnitude).

### Stochastic Depth (Layer Drop)

Randomly skip (drop) entire hidden layers during training for deeper layered networks to reduce effective depth and encourage resilient representations.

```ts
const deep = Architect.perceptron(8, 16, 16, 16, 4, 2); // input + 4 hidden + output
deep.setSeed(123); // reproducible skipping
deep.setStochasticDepth([0.9, 0.85, 0.8, 0.75]); // survival probabilities per hidden layer
deep.train(data, { iterations: 500, rate: 0.01 });
deep.disableStochasticDepth(); // inference uses full depth
```

Runtime info:

- Recently skipped layer indices available via `(deep as any)._lastSkippedLayers` (for test / debugging).
- Surviving layer outputs are scaled by `1/p` to preserve expectation (like inverted dropout).
- Dynamic scheduling: `deep.setStochasticDepthSchedule((step, current) => current.map((p,i)=> Math.max(0.5, p - 0.0005*step)))` to slowly reduce survival probabilities (example).

Rules:

- Provide exactly one probability per hidden layer (input & output excluded).
- Probabilities must be in (0,1]. Use 1.0 to always keep a layer.
- A layer only skips if its input activation vector size equals its own size (simple identity passthrough). Otherwise it is forced to run to avoid shape mismatch.
- Stochastic depth and node-level dropout can co-exist; skipping occurs before dropout at deeper layers.
- Clear schedule: `deep.clearStochasticDepthSchedule()`.

Example combining schedule + stats capture:

```ts
deep.setSeed(2025);
deep.setStochasticDepth([0.9, 0.85, 0.8]);
deep.setStochasticDepthSchedule((step, probs) =>
  probs.map((p) => Math.max(0.7, p - 0.0001 * step))
);
for (let i = 0; i < 10; i++) {
  deep.activate(sampleInput, true);
  console.log(deep.getRegularizationStats());
}
```

### DropConnect (recap)

Already supported: randomly drops individual connections per training pass.

```ts
net.enableDropConnect(0.2);
net.train(data, { iterations: 200, rate: 0.02 });
net.disableDropConnect();
```

## Architecture & Evolution

```
Structural pruning: magnitude-based & SNIP-style (|w * grad| saliency) prune+optional regrow, scheduled window.
Connection sparsification: progressive schedule toward target sparsity.
Neuroevolution: speciation (compatibility distance: excess, disjoint, weight diff), dynamic threshold (optional targetSpecies), kernel fitness sharing (quadratic within-species), stagnation injection (global refresh after N stagnant generations).
Innovation tracking: per-connection monotonically increasing counter (serialized); fallback Cantor pairing for missing.
Acyclic enforcement: optional; topological order cache for forward pass; cycle prevention via reachability test.
```

This section collects the dials that shape the evolutionary search landscape. In broad strokes you balance:

- Exploration (generate novel structure / escape local optima)
- Exploitation (refine promising topologies / weights)
- Parsimony pressure (avoid bloat that harms generalization & evaluation speed)
- Diversity maintenance (prevent premature species collapse)

The options below let you tune those pressures explicitly rather than relying on hidden heuristics. Start with defaults; introduce one mechanism at a time and watch telemetry (species count, best score, complexity trend) before layering more.

Disabled gene handling:
Connections now carry an `enabled` flag (classic NEAT). Structural mutations or pruning routines may disable a connection instead of deleting it. During crossover:

- Matching genes inherit enabled state; if either parent disabled the gene, it remains disabled unless re-enabled probabilistically.
- Disjoint/excess genes retain their parent's state.
  Re-enable probability is controlled by `reenableProb` (default 0.25). This allows temporarily silenced structure to be revived later, preserving historical innovation without immediate loss.

SNIP usage example:

```ts
net.configurePruning({
  start: 100,
  end: 1000,
  targetSparsity: 0.8,
  frequency: 10,
  regrowFraction: 0.1,
  method: 'snip', // use saliency |w * grad|, falls back to |w| if no grad yet
});
```

Notes:

- Saliency uses last accumulated gradient proxy (totalDeltaWeight or previousDeltaWeight) gathered during training steps.
- If gradients are zero/unused early, ranking gracefully reverts to pure magnitude.
- Regrow explores random new connections (respecting acyclic constraint) to maintain exploration.
- Pruning currently applies during gradient-based training, not inside the NEAT evolutionary loop (future option possible).
- Disabled genes are still serialized (`enabled:false`) and restored on load.

### Evolution Options (selected)

Focused knobs you are most likely to touch early. (Advanced / adaptive variants appear in the next table.)

| Option                             | Description                                                                        | Default       |
| ---------------------------------- | ---------------------------------------------------------------------------------- | ------------- |
| `targetSpecies`                    | Desired number of species used by dynamic compatibility threshold steering         | 8             |
| `sharingSigma`                     | Radius parameter for quadratic kernel fitness sharing                              | 3.0           |
| `globalStagnationGenerations`      | Generations without global improvement before injecting fresh genomes (0 disables) | 30            |
| `reenableProb`                     | Probability a disabled connection gene is re-enabled during crossover              | 0.25          |
| `evolutionPruning.startGeneration` | Generation index to begin structural pruning across genomes                        | –             |
| `evolutionPruning.interval`        | Apply pruning every N generations (default 1)                                      | 1             |
| `evolutionPruning.targetSparsity`  | Final sparsity fraction (e.g. 0.8 keeps 20% connections)                           | –             |
| `evolutionPruning.rampGenerations` | Generations to linearly ramp 0 -> target sparsity                                  | 0 (immediate) |
| `evolutionPruning.method`          | Prune ranking: 'magnitude' or 'snip'                                               | magnitude     |
| `compatAdjust.kp`                  | Proportional gain for threshold adjustment                                         | 0.3           |
| `compatAdjust.ki`                  | Integral gain (slow corrective action)                                             | 0.02          |
| `compatAdjust.smoothingWindow`     | EMA window for species count smoothing                                             | 5             |
| `compatAdjust.minThreshold`        | Lower clamp for compatibility threshold                                            | 0.5           |
| `compatAdjust.maxThreshold`        | Upper clamp for compatibility threshold                                            | 10            |
| `compatAdjust.decay`               | Integral decay factor (anti-windup)                                                | 0.95          |
| `sharingSigma`                     | If > 0 enables kernel fitness sharing (score_i /= sum_j (1-(d_ij/σ)^2))            | 0 (off)       |
| `globalStagnationGenerations`      | If >0 replace worst 20% genomes after N stagnant generations                       | 0 (off)       |

### Advanced Evolution Extensions

These extend the core NEAT loop with adaptive heuristics, diversity scaffolds, dynamic parsimony, and operator credit assignment. Treat them as modular experiments: enable one, observe telemetry trends (complexity trajectory, front size, novelty archive length), then decide whether it helped.

Heuristics taxonomy:

- Adaptive rates (adaptiveMutation, operatorBandit, operatorAdaptation)
- Diversity & dispersion (novelty, adaptiveSharing, diversityPressure, lineagePressure)
- Complexity governance (complexityBudget, phasedComplexity, complexityBudget adaptive)
- Objective management (multiObjective.dynamic / autoEntropy / adaptiveEpsilon)
- Speciation governance (autoCompatTuning, speciesAge\* protection/bonus)

| Option Group                  | Key Fields                                                                                                                                  | Purpose / Behavior                                                                                                                                                                                                                                                       | Default                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ---- |
| `adaptiveMutation`            | `{ enabled, initialRate, sigma, minRate, maxRate, adaptAmount, amountSigma, strategy, adaptEvery }`                                         | Enhanced per-genome adaptive mutation; strategies: twoTier (top half down, bottom up), exploreLow (boost weakest), anneal (global decay); optional simultaneous adaptation of mutation amount.                                                                           | disabled                                                                                                            |
| `novelty`                     | `{ descriptor(g), k, addThreshold, archiveLimit, blendFactor }`                                                                             | Novelty search scaffold blending kNN novelty with fitness; maintains archive                                                                                                                                                                                             | disabled                                                                                                            |
| `speciesAgeBonus`             | `{ youngGenerations, youngMultiplier, oldGenerations, oldPenalty }`                                                                         | Temporary boost for young species, penalty for very old                                                                                                                                                                                                                  | `{ youngGenerations:5, youngMultiplier:1.2, oldGenerations:30, oldPenalty:0.3 }`                                    |
| `speciesAgeProtection`        | `{ grace, oldPenalty }`                                                                                                                     | Protect very young species from early elimination; apply fitness scale to very old                                                                                                                                                                                       | `{ grace:3, oldPenalty:0.5 }`                                                                                       |
| `crossSpeciesMatingProb`      | number                                                                                                                                      | Chance to select second parent from another species (maintains diversity)                                                                                                                                                                                                | 0                                                                                                                   |
| `adaptiveSharing`             | `{ enabled, targetFragmentation, adjustStep, minSigma, maxSigma }`                                                                          | Auto-adjust kernel fitness sharing `sharingSigma` toward target fragmentation (#species/pop)                                                                                                                                                                             | disabled                                                                                                            |
| `minimalCriterion`            | `(network)=>boolean`                                                                                                                        | Filters genomes prior to speciation; failing genomes get score 0 (minimal criterion novelty style)                                                                                                                                                                       | undefined                                                                                                           |
| `operatorAdaptation`          | `{ enabled, window, boost, decay }`                                                                                                         | Tracks mutation operator success; weights selection probability by recent success                                                                                                                                                                                        | disabled                                                                                                            |
| `phasedComplexity`            | `{ enabled, phaseLength, simplifyFraction }`                                                                                                | Alternates complexify vs simplify phases; simplify prunes fraction of weakest connections                                                                                                                                                                                | disabled                                                                                                            |
| `complexityBudget`            | `{ enabled, maxNodesStart, maxNodesEnd, horizon }`                                                                                          | Linear schedule for max allowed nodes; prevents premature bloat                                                                                                                                                                                                          | disabled                                                                                                            |
| `multiObjective`              | `{ enabled, complexityMetric }`                                                                                                             | Pareto non-dominated sorting (maximize fitness, minimize complexity). `complexityMetric`: `'nodes'` or `'connections'`.                                                                                                                                                  | disabled                                                                                                            |
| `reenableProb` (adaptive)     | (base option)                                                                                                                               | Disabled gene reactivation probability; internally adaptively adjusted based on success when adaptive features on                                                                                                                                                        | 0.25                                                                                                                |
| `diversityPressure`           | `{ enabled, motifSample, penaltyStrength }`                                                                                                 | Penalizes over-represented small connection motif signatures to promote structural variety                                                                                                                                                                               | disabled                                                                                                            |
| `autoCompatTuning`            | `{ enabled, target, adjustRate, minCoeff, maxCoeff }`                                                                                       | Automatically scales excess/disjoint coefficients to approach target species count                                                                                                                                                                                       | disabled                                                                                                            |
| `speciesAllocation`           | `{ minOffspring, extendedHistory }`                                                                                                         | Guarantees minimum offspring per species (if capacity) and optionally records extended per-species metrics (compatibility, mean complexity, novelty)                                                                                                                     | `{ minOffspring:1, extendedHistory:true }`                                                                          |
| `minimalCriterionAdaptive`    | `{ enabled, initialThreshold, targetAcceptance, adjustRate, metric }`                                                                       | Dynamically adjusts acceptance threshold to maintain target pass rate before fitness ranking                                                                                                                                                                             | disabled                                                                                                            |
| `complexityBudget` (adaptive) | `{ enabled, mode:'adaptive', maxNodesStart, maxNodesEnd, improvementWindow, increaseFactor, stagnationFactor, maxConnsStart, maxConnsEnd }` | Adaptive complexity caps grow on improvement, shrink on stagnation                                                                                                                                                                                                       | disabled                                                                                                            |
| `operatorBandit`              | `{ enabled, c, minAttempts }`                                                                                                               | UCB1-style multi-armed bandit weighting for mutation operator selection                                                                                                                                                                                                  | disabled                                                                                                            |
| `novelty.pruneStrategy`       | `'fifo'                                                                                                                                     | 'sparse'`                                                                                                                                                                                                                                                                | Archive pruning: fifo drops oldest when over limit; sparse iteratively removes closest pair to maximize dispersion. | fifo |
| `telemetry`                   | `{ enabled, logEvery, performance, complexity, hypervolume, rngState }`                                                                     | Per-generation summary: base {gen,best,species,hyper,fronts}; optional perf timings, complexity block, hv scalar; if `lineageTracking` then lineage:{ parents, depthBest, meanDepth, inbreeding, ancestorUniq }; rngState embeds deterministic PRNG state (seeded runs). | disabled                                                                                                            |
| `lineageTracking`             | `boolean`                                                                                                                                   | Track parent genome ids & depth; adds `_parents`/`_depth` on genomes and enriched telemetry lineage block                                                                                                                                                                | true                                                                                                                |
| `multiObjective.autoEntropy`  | `boolean`                                                                                                                                   | Automatically append entropy (structure diversity proxy) objective (maximize)                                                                                                                                                                                            | false                                                                                                               |
| `multiObjective.dynamic`      | `{ enabled, addComplexityAt, addEntropyAt, dropEntropyOnStagnation, readdEntropyAfter }`                                                    | Dynamic scheduling of complexity/entropy objectives: delay adding complexity & entropy to let pure fitness search early; temporarily drop entropy during stagnation then re-add after cooldown.                                                                          | disabled                                                                                                            |

#### Example: Entropy-guided sharing sigma & ancestor uniqueness adaptive tuning

```ts
const neat = new Neat(inputs, outputs, fitness, {
  seed: 42,
  sharingSigma: 3.0,
  entropySharingTuning: {
    enabled: true,
    targetEntropyVar: 0.25,
    adjustRate: 0.1,
    minSigma: 0.5,
    maxSigma: 6,
  },
  ancestorUniqAdaptive: {
    enabled: true,
    mode: 'epsilon',
    lowThreshold: 0.25,
    highThreshold: 0.6,
    adjust: 0.01,
    cooldown: 4,
  },
  multiObjective: {
    enabled: true,
    autoEntropy: true,
    adaptiveEpsilon: {
      enabled: true,
      targetFront: Math.floor(Math.sqrt(popSize)),
    },
  },
  telemetry: { enabled: true, rngState: true },
  lineageTracking: true,
});
```

`entropySharingTuning` shrinks `sharingSigma` when structural entropy variance is too low (increasing local competition) and widens it when variance is high (reducing over-fragmentation). `entropyCompatTuning` dynamically nudges `compatibilityThreshold` based on mean structural entropy to balance species fragmentation. `ancestorUniqAdaptive` boosts diversity pressure (or relaxes dominance if using mode `epsilon`) when genealogical ancestorUniq metric drops below a threshold, and dials it back when uniqueness is already high. CSV export now includes `ops` operator stats and `objectives` active objective keys per generation when available.

| `multiObjective.adaptiveEpsilon` | `{ enabled,targetFront,adjust,min,max,cooldown }` | Auto-tune dominance epsilon toward target first-front size | disabled |
| `multiObjective.refPoint` | `number[] \| 'auto'` | Reference point for hypervolume (when telemetry.hypervolume true) | auto (slightly >1) |
| `exportParetoFrontJSONL()` | method | Export recent Pareto objective vectors JSONL (for external analysis) | - |
| `lineagePressure` | `{ enabled, mode, targetMeanDepth, strength }` | Post-eval score adjustment using lineage depth (`penalizeDeep` / `rewardShallow` / `spread`) | disabled |

Helper getters:

```ts
neat.getSpeciesHistory(); // rolling snapshots (size, age, best score)
neat.getNoveltyArchiveSize(); // current novelty archive length
neat.getMultiObjectiveMetrics(); // per-genome { rank, crowding, score, nodes, connections }
neat.getOperatorStats(); // operator adaptation/bandit stats
neat.getTelemetry(); // evolution telemetry log (recent)
neat.exportTelemetryCSV(); // flattened telemetry (includes lineage.* & diversity.lineage* if present)
neat.snapshotRNGState(); // { state } deterministic PRNG snapshot (if seeded)
neat.restoreRNGState(snap); // restore snapshot
neat.exportRNGState(); // JSON string of RNG state
neat.importRNGState(json); // restore from JSON
```

#### Example: Enabling Novelty + Multi-Objective

```ts
const neat = new Neat(4, 2, fitness, {
  popsize: 100,
  novelty: {
    descriptor: (net) =>
      net.nodes.filter((n) => n.type === 'H').map((h) => h.index % 2), // toy descriptor
    k: 10,
    addThreshold: 0.9,
    archiveLimit: 200,
    blendFactor: 0.3,
  },
  multiObjective: { enabled: true, complexityMetric: 'nodes' },
  adaptiveMutation: { enabled: true },
});
```

#### Example: Phased Complexity + Operator Adaptation

```ts
const neat = new Neat(3, 1, fitness, {
  phasedComplexity: { enabled: true, phaseLength: 5, simplifyFraction: 0.15 },
  operatorAdaptation: { enabled: true, window: 30, boost: 2.0, decay: 0.9 },
});
```

#### Minimal Criterion

```ts
const neat = new Neat(2, 1, fitness, {
  minimalCriterion: (net) => net.nodes.length >= 5, // require some complexity before scoring
});
```

Use when raw search space contains huge numbers of trivial zero-score genomes (e.g. all-linear tiny nets). The filter prevents them from influencing speciation/dominance ordering; they still evolve structurally until criterion passes.

```

#### Adaptive Sharing

If `adaptiveSharing.enabled` the system adjusts `sharingSigma` each generation:

```

sigma += step \* sign(fragmentation - target)

```

within `[minSigma,maxSigma]`.

#### Multi-Objective Notes & Strategy {#multi-objective-notes--strategy}

Implements a simplified NSGA-II style pass: fast non-dominated sort (O(N^2) current implementation) + crowding distance; final ordering uses (rank asc, crowding desc, fitness desc) before truncation. Practical guidance:

- Start single-objective until baseline performance plateaus, then enable `multiObjective.enabled` with `complexityMetric:'nodes'`.
- If early search stagnates due to premature parsimony, delay complexity with `multiObjective.dynamic.addComplexityAt`.
- Use `autoEntropy` to seed a third diversity proxy objective only when structural collapse is observed (few species, low ancestor uniqueness).
- Monitor front size; if it grows too large relative to population, enable `adaptiveEpsilon` to tighten dominance criteria.

Planned (future): faster dominance (divide-and-conquer), richer motif diversity pressure, automated compatibility coefficient tuning.

## ASCII Maze Example: 6‑Input Long-Range Vision (MazeVision)

The ASCII maze example uses a compact 6‑input perception schema ("MazeVision") with long‑range lookahead via a precomputed distance map. Inputs (order fixed):

1. compassScalar: Encodes the direction of the globally best next step toward the exit as a discrete scalar in {0,0.25,0.5,0.75} corresponding to N,E,S,W. Uses an extended horizon (H_COMPASS=5000) so it can see deeper than openness ratios.
2. openN
3. openE
4. openS
5. openW
6. progressDelta: Normalized recent progress signal around 0.5 ( >0.5 improving, <0.5 regressing ).

### Openness Semantics (openN/E/S/W) {#openness-semantics}

Each openness value describes the quality of the shortest path to the exit if the agent moves first in that direction, using a bounded lookahead horizon H=1000 over the distance map.

Value encoding:

- 1: Direction(s) whose total path length Ldir is minimal among all strictly improving neighbors (ties allowed; multiple 1s possible).
- Ratio 0 < Lmin / Ldir < 1: Direction is a valid strictly improving path but longer than the best (Lmin is the shortest improving path cost; Ldir = 1 + distance of neighbor cell). This supplies graded preference rather than binary pruning.
- 0: Wall, unreachable cell, dead end, or any non‑improving move (neighbor distance >= current distance) – all treated uniformly.
- 0.001: Special back‑only escape marker. When all four openness values would otherwise be 0 but the opposite of the previous successful action is traversable, that single opposite direction is set to 0.001 to indicate a pure retreat (pattern e.g. [0,0,0,0.001]).

Rules / Notes:

- Strict improvement filter: Only neighbors whose distanceMap value is strictly less than the current cell distance are considered for 1 or ratio values.
- Horizon clipping: Paths with Ldir > H are treated as unreachable (value 0) to bound search cost.
- Multiple bests: Corridors that fork into equivalently short routes produce multiple 1s, encouraging neutrality across equally optimal choices.
- Backtrack marker is intentionally very small (0.001) so evolution distinguishes "retreat only" states from true walls without overweighting them.
- Supervised refinement dataset intentionally contains ONLY deterministic single‑path cases (exactly one openness=1, others 0) for clarity; richer ratio/backtrack patterns appear only in the Lamarckian / evolutionary phase.

### progressDelta

Computed from recent distance improvement: delta = prevDistance - currentDistance, clipped to [-2,2], then mapped to [0,1] as 0.5 + delta/4. Values >0.5 mean progress toward exit; <0.5 regression or stalling.

### Debugging

Set ASCII_VISION_DEBUG=1 to emit periodic vision lines: current position, compassScalar, input vector, and per‑direction distance/ratio breakdown for auditing mismatches between maze geometry and distance map.

### Quick Reference

| Signal | Meaning                                             |
| ------ | --------------------------------------------------- |
| 1      | Best strictly improving path(s) (minimal Ldir)      |
| (0,1)  | Longer but improving path (ratio Lmin/Ldir)         |
| 0.001  | Only backtrack available (opposite of prior action) |
| 0      | Wall / dead end / non‑improving / unreachable       |

Implementation: `test/examples/asciiMaze/mazeVision.ts` (function `MazeVision.buildInputs6`).

This design minimizes input size (6 vs earlier large encodings) while preserving directional discrimination and long‑range planning cues, aiding faster evolutionary convergence and avoiding overfitting to local dead‑end noise.

## Roadmap / Backlog

Planned or partially designed enhancements not yet merged:

- Structural motif diversity pressure: penalize over-represented connection patterns (entropy-based sharing) to sustain innovation.
- Automated compatibility coefficient tuning: search or adapt excess/disjoint/weight coefficients to stabilize species counts without manual calibration.
- Faster Pareto sorting: divide-and-conquer or incremental dominance maintenance to reduce O(N^2) overhead for large populations.
- Connection complexity budget (current budget targets nodes only) and dual-objective weighting option.
- Diversity-aware parent selection leveraging motif entropy and archive dispersion.
- Extended novelty descriptors helper utilities (e.g. built-in graph metrics: depth, feedforwardness, clustering).
- Visualization hooks (species lineage graph, archive embedding projection) for diagnostics.
```
