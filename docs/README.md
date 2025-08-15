# NeatapticTS

[![License: MIT](https://img.shields.io/badge/license-MIT-2c3963.svg)](LICENSE) [![Docs](https://img.shields.io/badge/docs-generated-2c3963.svg)](docs/index.html)

<img src="https://cdn-images-1.medium.com/max/800/1*THG2__H9YHxYIt2sulzlTw.png" width="500px"/>

NeatapticTS is an educational TypeScript library for building and evolving neural networks.
It provides a flexible, minimal API for networks, neuro-evolution (NEAT-style operators),
multi-objective selection, and performance telemetry — with defaults suitable for learning,
research prototypes, and teaching.

This repository contains both the library source (in `src/`) and an auto-generated set of
documentation pages (in `docs/`) produced from the project's JSDoc comments. The per-folder
README files inside `src/` are regenerated from source JSDoc so GitHub shows up-to-date
API summaries.

Why use NeatapticTS?
- Small, readable TypeScript codebase aimed at education and research.
- Built-in neuro-evolution primitives (mutation, crossover, speciation).
- Optional multi-objective and Pareto evolution utilities.
- Designed for easy experimentation with runtime telemetry and parallel evaluation.

## Origins & comparison
NeatapticTS is inspired by and based on the original Neataptic project by Thomas Wagenaar (GitHub: `wagenaartje`). The original Neataptic (site: https://wagenaartje.github.io/neataptic/, repo: https://github.com/wagenaartje/neataptic) is a popular JavaScript library (many stars and examples) that offered fast neuro-evolution and backpropagation for browser and Node.js usage.

Notable points about the original Neataptic
- Author / repo: Thomas Wagenaar (`wagenaartje`) — project site and docs at https://wagenaartje.github.io/neataptic/ and source at https://github.com/wagenaartje/neataptic.
- Designed for in-browser demos and learning: playgrounds, neuroevolution examples, and articles accompany the code.
- Uses the Instinct neuro-evolution approach in many examples and claims very fast backpropagation and neural training performance compared with contemporaries.
- Built-in architect helpers (Perceptron, LSTM, GRU, etc.), low-level Node/Group/Layer APIs, and many examples and demos.
- License and status: the original repo ships with a LICENSE (see upstream) and has historically been widely used; at times it has been marked `unmaintained` in issues — consult the upstream repo for current status.

How NeatapticTS compares and what it adds
- TypeScript refactor: strongly-typed sources, clearer contracts, and easier maintenance for educational use.
- Parallel evaluation: worker-based multi-threaded evaluation to speed up population scoring on multi-core machines.
- Multi-objective & Pareto utilities: native support for Pareto-based evolution, hypervolume proxies, and runtime objective management.
- Adaptive complexity & pruning: runtime complexity budgets, adaptive pruning, and structural-entropy utilities to manage bloat during evolution.
- Performance ergonomics: activation array pooling, configurable activation precision, and other throughput options.
- Rich telemetry: extended per-generation telemetry (fronts, operator stats, lineage, complexity) to aid analysis and teaching.
- Documentation workflow: per-folder `README.md` auto-generated from JSDoc, plus rendered HTML site under `docs/` for easy browsing and publishing.
- Educational focus: conservative defaults, clearer examples, and documentation geared toward learners and researchers.

Notes
- This project builds on the ideas and API style of the original Neataptic, but it is a TypeScript-first rework with some API differences. It is not guaranteed to be a drop-in replacement; consult the generated per-folder docs in `src/*/README.md` or `docs/` for exact signatures and behaviors.
- For the canonical original project, demos, and historical context see:

- Neataptic site & docs: https://wagenaartje.github.io/neataptic/
- Neataptic GitHub: https://github.com/wagenaartje/neataptic

Quick links
- Documentation (auto-generated): `docs/` (open `docs/index.html` in a browser)
- Live API summaries mirrored into source: `src/*/README.md` (auto-generated)
- Tests & examples: `test/` and `test/examples/`

## Table of contents
- [Installation](#installation)
- [Minimal example](#minimal-example)
- [Documentation & API](#documentation--api)
- [Performance & tuning](#performance--tuning)
- [Contributing](#contributing)
- [License](#license)
- [Need help?](#need-help)

Installation

This project is primarily intended as a library you develop from source. To install the
published package (when available) or run locally:

1. Clone the repository and install dev dependencies:

```powershell
git clone https://github.com/reicek/NeatapticTS.git
cd NeatapticTS
npm install
```

2. Build and run the docs generator (the repo keeps generated READMEs inside `src/`):

```powershell
npm run docs
```
Minimal example

The example below shows a tiny setup that creates a `Neat` population and runs evaluate + evolve
for a few iterations. See per-folder READMEs in `src/` for detailed API surface.

```ts
import { Neat, Network } from './src/neat';

// simple fitness: maximize negative squared error for x->2x mapping
const fitness = (net: any) => {
	const out = net.activate([1])[0];
	return -Math.pow(out - 2, 2);
};

const neat = new Neat(1, 1, fitness, { popsize: 30, fastMode: true });

async function run() {
	await neat.evaluate();
	await neat.evolve();
	console.log('best', neat.getBest());
}

run();
```

Documentation & API

- The `docs/` folder contains rendered HTML pages (generated from the same READMEs) for
	browsing locally or publishing as static pages. See `docs/index.html`.
- Per-file and per-folder API summaries are kept inside `src/*/README.md` and are regenerated
	using the JSDoc comments in the TypeScript sources. Run `npm run docs` to refresh them.

Performance & tuning

NeatapticTS includes telemetry and knobs to trade throughput vs fidelity (threads, Lamarckian
refinement, mutation scheduling, complexity budgets). See `docs/src/README.md` and the
generated per-folder READMEs for the `Neat`, `Network`, and telemetry option surfaces.

Contributing

- This is a public, educational project. Contributions that improve clarity, add examples,
	or simplify the learning path are especially welcome.
- Please follow the repository's coding conventions and include tests for behavioral changes.
- For docs changes, update JSDoc comments in `src/` and run `npm run docs` to regenerate the
	markdown/html outputs.

License

This project is released under the MIT License — see the `LICENSE` file.

Attribution

- Core ideas and some code are derived from Neataptic (Thomas Wagenaar, `wagenaartje`) and Synaptic (Juan Cazala). See `LICENSE` for details and original copyrights.

Need help?

If something in the API is unclear, open an issue describing what you were trying to do and
which part of the documentation could have helped. We prioritize documentation improvements
and small example additions for educational clarity.

Enjoy learning and experimenting with neuro-evolution!

# NeatapticTS

NeatapticTS offers flexible neural networks; neurons and synapses can be removed with a single line of code. No fixed architecture is required for neural networks to function at all. This flexibility allows networks to be shaped for your dataset through neuro-evolution, which is done using multiple threads.

## Performance & Parallelism Tuning

This section summarizes practical knobs to accelerate evolution while preserving solution quality.

### Core Levers
| Lever | Effect | Guidance |
|-------|--------|----------|
| `threads` | Parallel genome evaluation | Increase until CPU saturation (watch diminishing returns > physical cores). Falls back to single-thread if workers unavailable. |
| `growth` | Structural penalty strength | Higher discourages bloating (faster eval, may limit innovation). Tune 5e-5..5e-4. |
| `mutationRate` / `mutationAmount` | Exploration breadth | For small populations (<=10) library enforces higher defaults. Reduce when convergence noisy. |
| `fastMode` | Lower sampling overhead | Use for CI or rapid iteration; disables some heavy sampling defaults. |
| `adaptiveMutation` | Dynamic operator pressure | Stabilizes search; can reduce wasted evaluations. |
| `telemetrySelect` | Reduce telemetry overhead | Keep only necessary blocks (e.g. ['performance']). |
| `lamarckianIterations` / `lamarckianSampleSize` | Local refinement vs throughput | Lower for diversity, raise for precision on stable plateaus. |
| `maxGenerations` (asciiMaze engine) | Safety cap | Prevents runaway long runs during tuning passes. |

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
	telemetry:{ enabled:true, performance:true },
	telemetrySelect:['performance','complexity'],
	adaptiveMutation:{ enabled:true, strategy:'twoTier' },
	fastMode:true
});
```

Monitor `getTelemetry().slice(-1)[0].perf` for current timings.

### Memory Optimizations (Activation Pooling & Precision)
`Network` constructor now accepts:
```ts
new Network(input, output, {
	activationPrecision: 'f32',          // default 'f64'; use float32 activations
	reuseActivationArrays: true,         // reuse a pooled output buffer each forward pass
	returnTypedActivations: true         // return the pooled Float32Array/Float64Array directly (no Array clone)
});
```
Guidelines:
* Use `activationPrecision:'f32'` for large populations or inference batches where 1e-7 precision loss is acceptable.
* Enable `reuseActivationArrays` to eliminate per-call allocation of output arrays (avoid mutating returned array between passes if reusing!).
* Set `returnTypedActivations:false` (default) if consumer code expects a plain JS array; this will clone when pooling is on.
* For maximum throughput (e.g., evaluation workers), combine all three.

These options are conservative by default to preserve existing test expectations.

### Future Improvements
Planned: micro-batch evaluation, worker task stealing, SIMD/WASM kernels, adaptive Lamarckian schedules.

### Deterministic Chain Mode (Test Utility)
`config.deterministicChainMode` (default: `false`) enables a simplified deterministic variant of the `ADD_NODE` structural mutation used strictly for tests that must guarantee a predictable deep linear chain.

When enabled BEFORE constructing / mutating a `Network`:
* Each `ADD_NODE` splits the terminal connection of the current single input→…→output chain (following the first outgoing edge each step).
* The original connection is replaced by two new ones (from→newHidden, newHidden→to).
* Any alternate outgoing edges encountered along the chain are pruned to preserve strict linearity.
* A direct input→output shortcut is removed once at least one hidden node exists, ensuring depth grows.

Intended Usage:
```ts
import { config } from 'neataptic';
config.deterministicChainMode = true; // enable
const net = new Network(1,1);
for (let i=0;i<5;i++) net.mutate(methods.mutation.ADD_NODE); // guaranteed 5 hidden chain
config.deterministicChainMode = false; // restore (recommended)
```
Rationale:
Normal evolutionary heuristics are stochastic and may add branches; some legacy tests asserted an exact depth progression. The flag isolates that legacy requirement without constraining typical evolutionary runs. Avoid using this mode in production evolution— it suppresses beneficial structural diversity.

Invariants (enforced after each deterministic `ADD_NODE`):
1. Exactly one outgoing forward edge per non-output node along the primary chain.
2. No direct input→output edge after the first hidden node is inserted.
3. Hidden node count increments by 1 per `ADD_NODE` call (barring impossible edge cases like empty connection sets).

If you need to debug chain growth without enabling global warnings, temporarily set a bespoke flag (e.g. `config.debugDeepPath`) and add localized logging; persistent debug output has been removed to keep test noise low.


## New Evolution Enhancements

Key advanced NEAT features now included:
* Multi-objective Pareto evolution (fast non-dominated sort) with pluggable objectives.
* Runtime objective registration: add/remove custom objectives without rewriting core sorter.
* Hypervolume proxy + Pareto front size telemetry (`getTelemetry()` now includes `fronts`, optional `hv`).
* Structural entropy proxy (degree-distribution entropy) available for custom objectives.
* Adaptive complexity budget (nodes & connections) with slope‑aware growth / contraction and diversity modulation.
* Species extended history (mean complexity, novelty, compatibility, entropy) when `speciesAllocation.extendedHistory=true`.
* Diversity pressure: motif frequency rarity bonus; novelty archive blending & optional sparse pruning.
* Self-adaptive mutation rates & amounts (strategies: twoTier, exploreLow, anneal) plus operator bandit selection.
* Persistent Pareto archive snapshots (first front) retrievable via `getParetoArchive()`.
* Performance profiling (evaluation & evolution durations) opt-in via `telemetry:{ performance:true }`.
* Complexity telemetry block (mean/max nodes/connections, enabled ratio, growth deltas) via `telemetry:{ complexity:true }`.
* Optional hypervolume scalar (`hv`) for first front via `telemetry:{ hypervolume:true }`.
* Lineage tracking: telemetry `lineage` block now includes `{ parents:[id1,id2], depthBest, meanDepth, inbreeding }` when `lineageTracking` (default true). Depth accumulates as max-parent-depth+1; `inbreeding` counts self-matings in last reproduction phase.
* Auto entropy objective: enable `multiObjective:{ enabled:true, autoEntropy:true }` to add a structural entropy maximization objective automatically.
* Adaptive dominance epsilon: `multiObjective:{ adaptiveEpsilon:{ enabled:true, targetFront:~sqrt(pop), adjust:0.002 } }` tunes `dominanceEpsilon` to stabilize first-front size.
* Reference-point hypervolume (optional): supply `multiObjective.refPoint` (array or `'auto'`) for improved HV scalar (overwrites proxy when `telemetry.hypervolume` set).
* Pareto front objective vectors export via `exportParetoFrontJSONL()`.
* Extended species metrics: variance, innovation range, enabled ratio, turnover, delta complexity & score.
* Adaptive novelty archive insertion threshold (`novelty.dynamicThreshold`) to target insertion rate.
* Inactive objective pruning: `multiObjective.pruneInactive` automatically removes stagnant objectives (zero range) after a window.
* Fast mode auto-tuning: `fastMode:true` reduces heavy sampling defaults (diversity, novelty) for faster iteration.
* Adaptive target species: `adaptiveTargetSpecies` maps structural entropy to a dynamic `targetSpecies` value (feeds compatibility controller).
* Auto distance coefficient tuning: `autoDistanceCoeffTuning` adjusts excess/disjoint coefficients based on entropy deviation.
* Adaptive pruning: `adaptivePruning` gently increases sparsity toward target using live complexity metrics.
* Objective importance telemetry (`objImportance`) with per-generation range & variance per objective.
* Objective lifecycle events (`objEvents`) and ages (`objAges`).

### Multi-Objective Usage

Default objectives (if `multiObjective.enabled`): maximize fitness (your score) + minimize complexity (`nodes` or `connections`).

Register additional objectives at runtime:
```ts
const neat = new Neat(4,2, fitnessFn, { popsize: 50, multiObjective:{ enabled:true, complexityMetric:'nodes' } });
// Add structural entropy (maximize)
neat.registerObjective('entropy','max', g => (neat as any)._structuralEntropy(g));
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
const neat = new Neat(4,2, fit, { telemetry:{ enabled:true, complexity:true, hypervolume:true } });
await neat.evolve();
console.log(neat.getTelemetry().slice(-1)[0].complexity); // { meanNodes, meanConns, ... }
```

### Adaptive Complexity Budget
Configure an adaptive schedule that expands limits when improvement slope is positive and contracts during stagnation:
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
[{ generation, stats:[ { id, size, best, lastImproved, age, meanNodes, meanConns, meanScore, meanNovelty, meanCompat, meanEntropy, varNodes, varConns, deltaMeanNodes, deltaMeanConns, deltaBestScore, turnoverRate, meanInnovation, innovationRange, enabledRatio } ] }]
```
Key new fields:
- `age`: generations since species creation.
- `varNodes/varConns`: intra-species variance of nodes/connections.
- `deltaMean*` & `deltaBestScore`: per-generation change indicators.
- `turnoverRate`: fraction of members new vs previous generation.
- `innovationRange`: spread of innovation IDs inside species.
- `enabledRatio`: enabled / total connections ratio (structural activation health).

### Diversity & Novelty
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
* Each generation the range (max-min) for every objective is computed.
* If the range stays below `rangeEps` for `window` consecutive generations the objective is removed.
* Keys listed in `protect` are never removed (even if stagnant).
* Telemetry will reflect the pruned objective list from the following generation.

Use this to keep the Pareto sorter focused on informative axes.

### Fast Mode Auto-Tuning

When iterating quickly or running inside tests you can set `fastMode:true` to scale down expensive sampling defaults:

```ts
const neat = new Neat(8,3, fitFn, {
	popsize: 80,
	fastMode: true,
	diversityMetrics:{ enabled:true }, // pairSample / graphletSample auto-lowered
	novelty:{ enabled:true, descriptor: g=>[g.nodes.length,g.connections.length] } // k auto-lowered if not explicitly set
});
```

Auto adjustments (only if corresponding option fields are undefined):
* `diversityMetrics.pairSample`: 40 -> 20
* `diversityMetrics.graphletSample`: 60 -> 30
* `novelty.k`: lowered to 5

The tuning runs once on first diversity stats computation. Override by explicitly supplying your own values.

Lineage depth diversity (auto when lineageTracking + diversityMetrics):
`diversity` object gains:
* `lineageMeanDepth`: mean `_depth` over population.
* `lineageMeanPairDist`: sampled mean absolute depth difference between genome pairs (structure ancestry dispersion).
Telemetry `lineage` block now also includes:
* `ancestorUniq`: average Jaccard distance (0..1) between ancestor sets of sampled genome pairs within a small depth window (higher = more genealogical diversification).

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
* `penalizeDeep`: subtracts proportional penalty `-(depth-target)*strength` when depth exceeds target.
* `rewardShallow`: adds proportional bonus for depths below target.
* `spread`: encourages dispersion around mean depth (boost far-from-mean, cap excessive depth).
* `antiInbreeding`: computes Jaccard overlap of ancestor sets of both parents (including parents) within `ancestorWindow` generations. Penalizes high overlap (>0.75) and rewards very distinct ancestry (<0.25). Penalty/bonus magnitude scales with overlap distance and the configured penalty/bonus parameters.

Runs after evaluation/speciation before sorting so it integrates with all other mechanisms.

### RNG State Snapshot & Reproducibility
When a numeric `seed` is provided, a fast deterministic PRNG drives stochastic choices. You can snapshot/restore mid-run:
```ts
const neat = new Neat(3,1, fit, { seed:1234 });
await neat.evolve();
const snap = neat.snapshotRNGState(); // { state }
// ... later
neat.restoreRNGState(snap);
// Serialize
const json = neat.exportRNGState();
// New instance (even without seed) can resume deterministic sequence
const neat2 = new Neat(3,1, fit, {});
neat2.importRNGState(json);
```
Restoring installs the internal deterministic generator if it wasn't active.

### Operator Adaptation & Mutation Self-Adaptation
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
* `objectives`: array of active objective keys that generation (already used internally for auditing dynamic scheduling).
* `objAges`: object mapping objective key -> consecutive generations active. When an objective is removed (e.g. via pruning or dynamic delay) its age resets to 0 if later reintroduced.
* `speciesAlloc`: array of `{ id, alloc }` giving number of offspring allocated to each species in the reproduction phase that produced the current generation. Useful for diagnosing allocation fairness / starvation.
* `objEvents`: array of `{ type:'add'|'remove', key }` describing objective lifecycle changes that occurred for that generation (emitted when dynamic scheduling or pruning alters the active set).

Example:
```ts
const neat = new Neat(4,2, fit, { popsize:40, multiObjective:{ enabled:true, autoEntropy:true, dynamic:{ enabled:true, addComplexityAt:2, addEntropyAt:5 } }, telemetry:{ enabled:true } });
for (let g=0; g<10; g++) { await neat.evaluate(); await neat.evolve(); }
const last = neat.getTelemetry().slice(-1)[0];
console.log(last.objectives); // ['fitness','complexity','entropy']
console.log(last.objAges);    // { fitness:10, complexity:9, entropy:5 }
console.log(last.speciesAlloc); // [{ id:1, alloc:7 }, { id:2, alloc:6 }, ...]
```

### CSV Export Columns (Extended)
`exportTelemetryCSV()` now conditionally includes the following columns when present:
* `rng` (deterministic RNG state snapshot if `rngState:true` + seed supplied)
* `ops` (JSON array of operator stats)
* `objectives` (JSON array of active objective keys)
* `objAges` (JSON object of key->age)
* `speciesAlloc` (JSON array of per-species offspring allocations)
* `objEvents` (JSON array of add/remove objective lifecycle events)
Existing flattened sections remain: `complexity.*`, `perf.*`, `lineage.*`, `diversity.lineageMeanDepth`, `diversity.lineageMeanPairDist`, and `fronts`.

Performance profiling telemetry (optional):
```ts
const neat = new Neat(4,2, fitnessFn, { telemetry:{ enabled:true, performance:true } });
await neat.evaluate(); await neat.evolve();
const last = neat.getTelemetry().slice(-1)[0];
console.log(last.perf); // { evalMs, evolveMs }

Export telemetry:
```ts
neat.exportTelemetryJSONL(); // JSON Lines
neat.exportTelemetryCSV();   // CSV (flattened complexity/perf)
```

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

NeatapticTS now supports exporting to and importing from ONNX format for strictly layered MLPs. This allows interoperability with other machine learning frameworks.

- Use `exportToONNX(network)` to export a network to ONNX.
- Use `importFromONNX(onnxModel)` to import a compatible ONNX model as a `Network` instance.

See tests in `test/network/onnx.export.test.ts` and `test/network/onnx.import.test.ts` for usage examples.

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
net.train(data, { iterations:500, rate:0.01, optimizer:'adam', gradientClip:{ mode:'norm', maxNorm:1 } });
net.train(data, { iterations:500, rate:0.01, optimizer:'adam', gradientClip:{ mode:'percentile', percentile:99 } });
// Layerwise variants: 'layerwiseNorm' | 'layerwisePercentile'
```

Micro-batch accumulation (simulate larger effective batch without increasing memory):
```ts
// Process 1 sample at a time, accumulate 8 micro-batches, then apply one optimizer step
net.train(data, { iterations:100, rate:0.005, batchSize:1, accumulationSteps:8, optimizer:'adam' });
```
If `accumulationSteps > 1`, gradients are averaged before the optimizer step so results match a single larger batch (deterministic given same sample order).

Mixed precision (simulated FP16 gradients with FP32 master weights + dynamic loss scaling):
```ts
net.train(data, { iterations:300, rate:0.01, optimizer:'adam', mixedPrecision:{ lossScale:1024 } });
```
Behavior:
* Stores master FP32 copies of weights/biases (`_fp32Weight`, `_fp32Bias`).
* Scales gradients during accumulation; unscales before clipping / optimizer update; adjusts `lossScale` down on overflow (NaN/Inf), attempts periodic doubling after sustained stable steps (configurable via `mixedPrecision.dynamic`).
* Raw gradient norm (pre-optimizer, post-scaling/clipping) exposed via metrics hook as `gradNormRaw` (legacy post-update norm still `gradNorm`).
* Pure JS numbers remain 64-bit; this is a functional simulation for stability and future WASM/GPU backends.

Clipping modes:
| mode | scope | description |
|------|-------|-------------|
| norm | global | Clip global L2 gradient norm to `maxNorm`. |
| percentile | global | Clamp individual gradients above given percentile magnitude. |
| layerwiseNorm | per layer | Apply norm clipping per architectural layer (fallback per node if no layer info). Optionally splits weight vs bias groups via `gradientClip.separateBias`. |
| layerwisePercentile | per layer | Percentile clamp per architectural layer (fallback per node). Supports `separateBias`. |

Notes:
* Provide either `{ mode, maxNorm? , percentile? }` or shorthand `{ maxNorm }` / `{ percentile }`.
* Percentile ranking is magnitude-based.
* Accumulation averages gradients; to sum instead (rare) scale `rate` accordingly.
* Dynamic loss scaling heuristics: halves on detected overflow, doubles after configurable stable steps (default 200) within bounds `[minScale,maxScale]`.
* Config: `mixedPrecision:{ lossScale:1024, dynamic:{ minScale:1, maxScale:131072, increaseEvery:300 } }`.
* Accumulation reduction: default averages gradients; specify `accumulationReduction:'sum'` to sum instead (then adjust learning rate manually, e.g. multiply by 1/accumulationSteps if you want equivalent averaging semantics).
* Layerwise clipping: set `gradientClip:{ mode:'layerwiseNorm', maxNorm:1, separateBias:true }` to treat biases separately from weights.
* Two gradient norms tracked: raw (pre-update) and legacy (post-update deltas). Future APIs may expose both formally.
* Access stats: `net.getTrainingStats()` -> `{ gradNorm, gradNormRaw, lossScale, optimizerStep, mp:{ good, bad, overflowCount, scaleUps, scaleDowns, lastOverflowStep } }`.
* Test hook (not for production): `net.testForceOverflow()` forces the next mixed-precision step to register an overflow (used in unit tests to validate telemetry paths).
* Gradient clip grouping count: `net.getLastGradClipGroupCount()` (useful to verify separateBias effect).
* Rate helper for accumulation: `const adjRate = Network.adjustRateForAccumulation(rate, accumulationSteps, accumulationReduction)`.
* Deterministic seeding: `new Network(4,2,{ seed:123 })` or later `net.setSeed(123)` ensures reproducible initial weights, biases, connection order, and mutation randomness for training (excluding certain static reconstruction paths). For NEAT evolution pass `seed` in `new Neat(...,{ seed:999 })`.
* Overflow telemetry: during mixed precision training, `overflowCount` increments on detected NaN/Inf when unscaling gradients; `scaleUps` / `scaleDowns` count dynamic loss scale adjustments.

### Learning Rate Scheduler Usage

```ts
import methods from './src/methods/methods';
const net = new Network(2,1);
const ratePolicy = methods.Rate.cosineAnnealingWarmRestarts(200, 1e-5, 2);
net.train(data, { iterations: 1000, rate: 0.1, ratePolicy });
```

Reduce-on-plateau automatically receives current error because `train` detects a 3-arg scheduler:
```ts
const rop = methods.Rate.reduceOnPlateau({ patience: 20, factor: 0.5, minRate: 1e-5 });
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

| Type | Description | Key Params | When to Use |
|------|-------------|------------|-------------|
| sma | Simple moving average over window | movingAverageWindow | General mild smoothing |
| ema | Exponential moving average | emaAlpha (default 2/(N+1)) | Faster reaction than SMA |
| adaptive-ema | Dual-track variance-adaptive EMA (takes min of baseline & adaptive for guaranteed non-worse smoothing) | emaAlpha, adaptiveAlphaMin/Max, adaptiveVarianceFactor | Volatile early phase responsiveness with stability guarantee |
| wma | Linear weighted (recent heavier) | movingAverageWindow | Slightly more responsive than SMA |
| median | Moving median | movingAverageWindow | Spike / outlier robustness |
| trimmed | Trimmed mean (drops extremes) | trimmedRatio (0-0.5) | Moderate outliers; keep efficiency |
| gaussian | Gaussian-weighted window (recent emphasized smoothly) | gaussianSigma (default N/3) | Want smooth taper vs linear weights |

Additional options:
* trimmedRatio: fraction trimmed from each side for 'trimmed' (default 0.1)
* gaussianSigma: width for 'gaussian' (default window/3)
* trackVariance: true to compute running variance & min (exposed via metricsHook)
* adaptiveAlphaMin / adaptiveAlphaMax / adaptiveVarianceFactor for adaptive-ema control

Metrics hook additions when smoothing active: `rawError`, `runningVariance`, `runningMin` (if trackVariance true) alongside smoothed `error`.
* Target `error` comparison and improvement tracking both use the smoothed value.
* `earlyStopPatience` counts iterations since the last smoothed improvement > `earlyStopMinDelta`.
* Plateau smoothing: provide any of `plateauMovingAverageWindow`, `plateauMovingAverageType`, `plateauEmaAlpha`, `plateauTrimmedRatio`, `plateauGaussianSigma`, `plateauAdaptiveAlphaMin/Max`, `plateauAdaptiveVarianceFactor`.
	- If none are supplied the scheduler reuses the early-stop smoothing.
	- If supplied, metricsHook receives an extra field `plateauError` (the separately smoothed value supplied to the scheduler), while `error` remains the early-stop smoothed value.
	- Plateau adaptive-ema uses the same dual-track min(adaptive, baseline) logic.
* This does not interfere with `Rate.reduceOnPlateau` patience; they are independent.


#### Scheduler Reference

| Scheduler | Factory Call | Key Params | Behavior |
|-----------|--------------|------------|----------|
| fixed | `Rate.fixed()` | – | Constant learning rate. |
| step | `Rate.step(gamma?, stepSize?)` | gamma (default 0.9), stepSize (default 100) | Multiplies rate by `gamma` every `stepSize` iterations. |
| exp | `Rate.exp(gamma?)` | gamma (default 0.999) | Exponential decay: `rate * gamma^t`. |
| inv | `Rate.inv(gamma?, power?)` | gamma (0.001), power (2) | Inverse time decay: `rate / (1 + γ * t^p)`. |
| cosine annealing | `Rate.cosineAnnealing(period?, minRate?)` | period (1000), minRate (0) | Cosine decay from base to `minRate` each period. |
| cosine warm restarts | `Rate.cosineAnnealingWarmRestarts(initialPeriod, minRate?, tMult?)` | initialPeriod, minRate (0), tMult (2) | Cosine cycles with period multiplied by `tMult` after each restart. |
| linear warmup + decay | `Rate.linearWarmupDecay(totalSteps, warmupSteps?, endRate?)` | totalSteps, warmupSteps (auto 10%), endRate (0) | Linear ramp to base, then linear decay to `endRate`. |
| reduce on plateau | `Rate.reduceOnPlateau(opts)` | patience, factor, minRate, threshold | Monitors error; reduces current rate by `factor` after `patience` non-improving iterations. |

Notes:
* All scheduler factories return a function `(baseRate, iteration)` except `reduceOnPlateau`, which returns `(baseRate, iteration, error)`; `train` auto-detects and supplies `error` if the function arity is 3.
* You can wrap or compose schedulers—see below for a composition pattern.

#### Evolution Warning

`evolve()` now always emits a warning `"Evolution completed without finding a valid best genome"` when no suitable genome is selected (including zero-iteration runs) to aid testability and user feedback.

#### Composing Schedulers (Warmup then Plateau)

You can combine policies by writing a small delegator that switches logic after warmup completes and still passes error when required:

```ts
import methods from './src/methods/methods';

const warmup = methods.Rate.linearWarmupDecay(500, 100, 0.1); // only use warmup phase portion
const plateau = methods.Rate.reduceOnPlateau({ patience: 15, factor: 0.5, minRate: 1e-5 });

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
	metricsHook: ({ iteration, error, gradNorm }) => console.log(iteration, error, gradNorm),
	checkpoint: {
		best: true,
		last: true,
		save: ({ type, iteration, error, network }) => {/* persist */}
	}
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
	optimizer: { type: 'adamw', beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 }
});
```

Lion (sign-based update):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'lion', beta1: 0.9, beta2: 0.99 } });
```

Adamax (robust to sparse large gradients):
```ts
net.train(data, { iterations: 300, rate: 0.002, optimizer: { type: 'adamax', beta1: 0.9, beta2: 0.999 } });
```

NAdam (Nesterov momentum style lookahead on first moment):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'nadam', beta1: 0.9, beta2: 0.999 } });
```

RAdam (more stable early training variance):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'radam', beta1: 0.9, beta2: 0.999 } });
```

AdaBelief (faster convergence / better generalization via surprise-based variance):
```ts
net.train(data, { iterations: 300, rate: 0.001, optimizer: { type: 'adabelief', beta1: 0.9, beta2: 0.999 } });
```

Lookahead (wraps a base optimizer; performs k fast steps then interpolates):
```ts
net.train(data, {
	iterations: 400,
	rate: 0.01,
	optimizer: { type: 'lookahead', baseType: 'adam', la_k: 5, la_alpha: 0.5 }
});
```

Optimizer reference:

| Optimizer   | Key Params (in object) | Notes |
|-------------|------------------------|-------|
| sgd         | momentum               | Nesterov momentum internally in propagate when update=true |
| rmsprop     | eps                    | Uses fixed decay 0.9 / 0.1 split for cache |
| adagrad     | eps                    | Cache accumulates squared grads (monotonic) |
| adam        | beta1, beta2, eps      | Standard bias correction |
| adamw       | beta1, beta2, eps, weightDecay | Decoupled weight decay applied after adaptive step |
| amsgrad     | beta1, beta2, eps      | Maintains max second-moment vhat |
| adamax      | beta1, beta2, eps      | Infinity norm (u) instead of v |
| nadam       | beta1, beta2, eps      | Nesterov variant of Adam |
| radam       | beta1, beta2, eps      | Rectifies variance early in training |
| lion        | beta1, beta2           | Direction = sign(beta1*m + beta2*m2) |
| adabelief   | beta1, beta2, eps      | Second moment of (g - m) (gradient surprise) |
| lookahead   | baseType, la_k, la_alpha | Interpolates toward slow weights every k steps |

General notes:
* Step counter resets each `train` call (t starts at 1) for reproducibility.
* `lookahead.baseType` defaults to `adam` and cannot itself be `lookahead`.
* Only AdamW applies decoupled `weightDecay`; for others combine regularization if needed.
* Adamax may help with sparse or bursty gradients (uses infinity norm).\
* NAdam can yield slightly faster early progress due to lookahead on m.\
* RAdam mitigates the need for warmup; behaves like Adam after variance rectification threshold.\
* AdaBelief can reduce over-adaptation to noisy gradients by modeling belief deviation.\
* Lion performs well in some large-scale settings due to sign-based memory efficiency.

### Label Smoothing

Cross-entropy with label smoothing discourages over-confident predictions.

```ts
import methods from './src/methods/methods';
const loss = methods.Cost.labelSmoothing([1,0,0],[0.8,0.1,0.1],0.1);
```

### Weight Noise

Adds zero-mean Gaussian noise to weights on each *training* forward pass (inference unaffected). Original weights are restored immediately after the pass (noise is ephemeral / non-destructive).

Basic usage:
```ts
net.enableWeightNoise(0.05); // global stdDev = 0.05
net.train(data, { iterations: 100, rate:0.05 });
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
net.setWeightNoiseSchedule(step => 0.1 * Math.cos(step / 500));
```

Deterministic seeding / custom RNG (affects dropout, dropconnect, stochastic depth, weight noise sampling):
```ts
net.setSeed(42); // reproducible stochastic regularization
// or provide a custom RNG
net.setRandom(() => myDeterministicGenerator.next());
```

Diagnostics:
* Each connection temporarily stores last noise in `connection._wnLast` (for test / inspection).
* Global training forward pass count: `net.trainingStep`.
* Last skipped layers (stochastic depth): `net.lastSkippedLayers`.
* Regularization statistics (after any forward): `net.getRegularizationStats()` returns
	`{ droppedHiddenNodes, totalHiddenNodes, droppedConnections, totalConnections, skippedLayers, weightNoise: { count, sumAbs, maxAbs, meanAbs } }`.
* To combine with DropConnect / Dropout the sampling is independent (noise applied before masking).

Gotchas:
* Per-layer noise ignores global schedule (schedule currently applies only when using a single global std). If you need both, emulate by updating `enableWeightNoise` each epoch.
* Very large noise (> weight scale) can destabilize gradients; start small (e.g. 1-10% of typical weight magnitude).

### Stochastic Depth (Layer Drop)

Randomly skip (drop) entire hidden layers during training for deeper layered networks to reduce effective depth and encourage resilient representations.

```ts
const deep = Architect.perceptron(8,16,16,16,4,2); // input + 4 hidden + output
deep.setSeed(123); // reproducible skipping
deep.setStochasticDepth([0.9,0.85,0.8,0.75]); // survival probabilities per hidden layer
deep.train(data, { iterations:500, rate:0.01 });
deep.disableStochasticDepth(); // inference uses full depth
```

Runtime info:
* Recently skipped layer indices available via `(deep as any)._lastSkippedLayers` (for test / debugging).
* Surviving layer outputs are scaled by `1/p` to preserve expectation (like inverted dropout).
* Dynamic scheduling: `deep.setStochasticDepthSchedule((step, current) => current.map((p,i)=> Math.max(0.5, p - 0.0005*step)))` to slowly reduce survival probabilities (example).

Rules:
* Provide exactly one probability per hidden layer (input & output excluded).
* Probabilities must be in (0,1]. Use 1.0 to always keep a layer.
* A layer only skips if its input activation vector size equals its own size (simple identity passthrough). Otherwise it is forced to run to avoid shape mismatch.
* Stochastic depth and node-level dropout can co-exist; skipping occurs before dropout at deeper layers.
* Clear schedule: `deep.clearStochasticDepthSchedule()`.

Example combining schedule + stats capture:
```ts
deep.setSeed(2025);
deep.setStochasticDepth([0.9,0.85,0.8]);
deep.setStochasticDepthSchedule((step, probs) => probs.map(p => Math.max(0.7, p - 0.0001*step)));
for (let i=0;i<10;i++) {
	deep.activate(sampleInput, true);
	console.log(deep.getRegularizationStats());
}
```

### DropConnect (recap)
Already supported: randomly drops individual connections per training pass.

```ts
net.enableDropConnect(0.2);
net.train(data, { iterations:200, rate:0.02 });
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

Disabled gene handling:
Connections now carry an `enabled` flag (classic NEAT). Structural mutations or pruning routines may disable a connection instead of deleting it. During crossover:
* Matching genes inherit enabled state; if either parent disabled the gene, it remains disabled unless re-enabled probabilistically.
* Disjoint/excess genes retain their parent's state.
Re-enable probability is controlled by `reenableProb` (default 0.25). This allows temporarily silenced structure to be revived later, preserving historical innovation without immediate loss.

SNIP usage example:
```ts
net.configurePruning({
	start: 100,
	end: 1000,
	targetSparsity: 0.8,
	frequency: 10,
	regrowFraction: 0.1,
	method: 'snip' // use saliency |w * grad|, falls back to |w| if no grad yet
});
```

Notes:
* Saliency uses last accumulated gradient proxy (totalDeltaWeight or previousDeltaWeight) gathered during training steps.
* If gradients are zero/unused early, ranking gracefully reverts to pure magnitude.
* Regrow explores random new connections (respecting acyclic constraint) to maintain exploration.
* Pruning currently applies during gradient-based training, not inside the NEAT evolutionary loop (future option possible).
* Disabled genes are still serialized (`enabled:false`) and restored on load.

### Evolution Options (selected)

| Option | Description | Default |
|--------|-------------|---------|
| `targetSpecies` | Desired number of species used by dynamic compatibility threshold steering | 8 |
| `sharingSigma` | Radius parameter for quadratic kernel fitness sharing | 3.0 |
| `globalStagnationGenerations` | Generations without global improvement before injecting fresh genomes (0 disables) | 30 |
| `reenableProb` | Probability a disabled connection gene is re-enabled during crossover | 0.25 |
| `evolutionPruning.startGeneration` | Generation index to begin structural pruning across genomes | – |
| `evolutionPruning.interval` | Apply pruning every N generations (default 1) | 1 |
| `evolutionPruning.targetSparsity` | Final sparsity fraction (e.g. 0.8 keeps 20% connections) | – |
| `evolutionPruning.rampGenerations` | Generations to linearly ramp 0 -> target sparsity | 0 (immediate) |
| `evolutionPruning.method` | Prune ranking: 'magnitude' or 'snip' | magnitude |
| `targetSpecies` | Desired species count for dynamic compatibility threshold controller | 8 |
| `compatAdjust.kp` | Proportional gain for threshold adjustment | 0.3 |
| `compatAdjust.ki` | Integral gain (slow corrective action) | 0.02 |
| `compatAdjust.smoothingWindow` | EMA window for species count smoothing | 5 |
| `compatAdjust.minThreshold` | Lower clamp for compatibility threshold | 0.5 |
| `compatAdjust.maxThreshold` | Upper clamp for compatibility threshold | 10 |
| `compatAdjust.decay` | Integral decay factor (anti-windup) | 0.95 |
| `sharingSigma` | If > 0 enables kernel fitness sharing (score_i /= sum_j (1-(d_ij/σ)^2)) | 0 (off) |
| `globalStagnationGenerations` | If >0 replace worst 20% genomes after N stagnant generations | 0 (off) |

### Advanced Evolution Extensions

The TypeScript refactor adds several research-grade evolutionary controls. All are optional and off unless noted.

| Option Group | Key Fields | Purpose / Behavior | Default |
|--------------|-----------|--------------------|---------|
| `adaptiveMutation` | `{ enabled, initialRate, sigma, minRate, maxRate, adaptAmount, amountSigma, strategy, adaptEvery }` | Enhanced per-genome adaptive mutation; strategies: twoTier (top half down, bottom up), exploreLow (boost weakest), anneal (global decay); optional simultaneous adaptation of mutation amount. | disabled |
| `novelty` | `{ descriptor(g), k, addThreshold, archiveLimit, blendFactor }` | Novelty search scaffold blending kNN novelty with fitness; maintains archive | disabled |
| `speciesAgeBonus` | `{ youngGenerations, youngMultiplier, oldGenerations, oldPenalty }` | Temporary boost for young species, penalty for very old | `{ youngGenerations:5, youngMultiplier:1.2, oldGenerations:30, oldPenalty:0.3 }` |
| `speciesAgeProtection` | `{ grace, oldPenalty }` | Protect very young species from early elimination; apply fitness scale to very old | `{ grace:3, oldPenalty:0.5 }` |
| `crossSpeciesMatingProb` | number | Chance to select second parent from another species (maintains diversity) | 0 |
| `adaptiveSharing` | `{ enabled, targetFragmentation, adjustStep, minSigma, maxSigma }` | Auto-adjust kernel fitness sharing `sharingSigma` toward target fragmentation (#species/pop) | disabled |
| `minimalCriterion` | `(network)=>boolean` | Filters genomes prior to speciation; failing genomes get score 0 (minimal criterion novelty style) | undefined |
| `operatorAdaptation` | `{ enabled, window, boost, decay }` | Tracks mutation operator success; weights selection probability by recent success | disabled |
| `phasedComplexity` | `{ enabled, phaseLength, simplifyFraction }` | Alternates complexify vs simplify phases; simplify prunes fraction of weakest connections | disabled |
| `complexityBudget` | `{ enabled, maxNodesStart, maxNodesEnd, horizon }` | Linear schedule for max allowed nodes; prevents premature bloat | disabled |
| `multiObjective` | `{ enabled, complexityMetric }` | Pareto non-dominated sorting (fitness maximize, complexity minimize). `complexityMetric`=`'nodes'|'connections'` | disabled |
| `reenableProb` (adaptive) | (base option) | Disabled gene reactivation probability; internally adaptively adjusted based on success when adaptive features on | 0.25 |
| `diversityPressure` | `{ enabled, motifSample, penaltyStrength }` | Penalizes over-represented small connection motif signatures to promote structural variety | disabled |
| `autoCompatTuning` | `{ enabled, target, adjustRate, minCoeff, maxCoeff }` | Automatically scales excess/disjoint coefficients to approach target species count | disabled |
| `speciesAllocation` | `{ minOffspring, extendedHistory }` | Guarantees minimum offspring per species (if capacity) and optionally records extended per-species metrics (compatibility, mean complexity, novelty) | `{ minOffspring:1, extendedHistory:true }` |
| `minimalCriterionAdaptive` | `{ enabled, initialThreshold, targetAcceptance, adjustRate, metric }` | Dynamically adjusts acceptance threshold to maintain target pass rate before fitness ranking | disabled |
| `complexityBudget` (adaptive) | `{ enabled, mode:'adaptive', maxNodesStart, maxNodesEnd, improvementWindow, increaseFactor, stagnationFactor, maxConnsStart, maxConnsEnd }` | Adaptive complexity caps grow on improvement, shrink on stagnation | disabled |
| `operatorBandit` | `{ enabled, c, minAttempts }` | UCB1-style multi-armed bandit weighting for mutation operator selection | disabled |
| `novelty.pruneStrategy` | `'fifo'|'sparse'` | Sparse pruning removes closest archive pair iteratively to keep diversity | fifo |
| `telemetry` | `{ enabled, logEvery, performance, complexity, hypervolume, rngState }` | Per-generation summary: base { gen,best,species,hyper,fronts }; optional perf timings, complexity block, hv scalar; if `lineageTracking` then `lineage:{ parents, depthBest, meanDepth, inbreeding, ancestorUniq }`; `rngState` embeds deterministic PRNG state (seeded runs) | disabled |
| `lineageTracking` | `boolean` | Track parent genome ids & depth; adds `_parents`/`_depth` on genomes and enriched telemetry lineage block | true |
| `multiObjective.autoEntropy` | `boolean` | Automatically append entropy (structure diversity proxy) objective (maximize) | false |
| `multiObjective.dynamic` | `{ enabled, addComplexityAt, addEntropyAt, dropEntropyOnStagnation, readdEntropyAfter }` | Dynamic scheduling of complexity/entropy objectives: delay adding complexity & entropy to let pure fitness search early; temporarily drop entropy during stagnation then re-add after cooldown. | disabled |
#### Example: Entropy-guided sharing sigma & ancestor uniqueness adaptive tuning
```ts
const neat = new Neat(inputs, outputs, fitness, {
	seed: 42,
	sharingSigma: 3.0,
	entropySharingTuning: { enabled:true, targetEntropyVar:0.25, adjustRate:0.1, minSigma:0.5, maxSigma:6 },
	ancestorUniqAdaptive: { enabled:true, mode:'epsilon', lowThreshold:0.25, highThreshold:0.6, adjust:0.01, cooldown:4 },
	multiObjective: { enabled:true, autoEntropy:true, adaptiveEpsilon:{ enabled:true, targetFront: Math.floor(Math.sqrt( popSize )) } },
	telemetry: { enabled:true, rngState:true },
	lineageTracking: true
});
```
`entropySharingTuning` shrinks `sharingSigma` when structural entropy variance is too low (increasing local competition) and widens it when variance is high (reducing over-fragmentation). `entropyCompatTuning` dynamically nudges `compatibilityThreshold` based on mean structural entropy to balance species fragmentation. `ancestorUniqAdaptive` boosts diversity pressure (or relaxes dominance if using mode `epsilon`) when genealogical ancestorUniq metric drops below a threshold, and dials it back when uniqueness is already high. CSV export now includes `ops` operator stats and `objectives` active objective keys per generation when available.

| `multiObjective.adaptiveEpsilon` | `{ enabled,targetFront,adjust,min,max,cooldown }` | Auto-tune dominance epsilon toward target first-front size | disabled |
| `multiObjective.refPoint` | `number[] | 'auto'` | Reference point for hypervolume (when telemetry.hypervolume true) | auto (slightly >1) |
| `exportParetoFrontJSONL()` | method | Export recent Pareto objective vectors JSONL (for external analysis) | - |
| `lineagePressure` | `{ enabled, mode, targetMeanDepth, strength }` | Post-eval score adjustment using lineage depth (`penalizeDeep` / `rewardShallow` / `spread`) | disabled |

Helper getters:
```ts
neat.getSpeciesHistory();       // rolling snapshots (size, age, best score)
neat.getNoveltyArchiveSize();   // current novelty archive length
neat.getMultiObjectiveMetrics(); // per-genome { rank, crowding, score, nodes, connections }
neat.getOperatorStats();        // operator adaptation/bandit stats
neat.getTelemetry();            // evolution telemetry log (recent)
neat.exportTelemetryCSV();      // flattened telemetry (includes lineage.* & diversity.lineage* if present)
neat.snapshotRNGState();        // { state } deterministic PRNG snapshot (if seeded)
neat.restoreRNGState(snap);     // restore snapshot
neat.exportRNGState();          // JSON string of RNG state
neat.importRNGState(json);      // restore from JSON
```
const neat = new Neat(4,2, fit, { telemetry:{ enabled:true, complexity:true, hypervolume:true }, lineageTracking:true, multiObjective:{ enabled:true, autoEntropy:true } });
#### Example: Enabling Novelty + Multi-Objective
```ts
const neat = new Neat(4, 2, fitness, {
	popsize: 100,
	novelty: {
		descriptor: net => net.nodes.filter(n=>n.type==='H').map(h=>h.index%2), // toy descriptor
		k: 10,
		addThreshold: 0.9,
		archiveLimit: 200,
		blendFactor: 0.3
	},
	multiObjective: { enabled: true, complexityMetric: 'nodes' },
	adaptiveMutation: { enabled: true }
});
```

#### Example: Phased Complexity + Operator Adaptation
```ts
const neat = new Neat(3,1, fitness, {
	phasedComplexity: { enabled:true, phaseLength: 5, simplifyFraction: 0.15 },
	operatorAdaptation: { enabled:true, window: 30, boost: 2.0, decay: 0.9 }
});
```

#### Minimal Criterion
```ts
const neat = new Neat(2,1, fitness, {
	minimalCriterion: net => net.nodes.length >= 5 // require some complexity before scoring
});
```

#### Adaptive Sharing
If `adaptiveSharing.enabled` the system adjusts `sharingSigma` each generation:
```
sigma += step * sign(fragmentation - target)
```
within `[minSigma,maxSigma]`.

#### Multi-Objective Notes
Implements a simplified NSGA-II style pass: fast non-dominated sort (O(N^2) current implementation) + crowding distance; final ordering uses (rank asc, crowding desc, fitness desc) before truncation to next generation size.

Planned (future): faster dominance (divide-and-conquer), structural motif diversity pressure, automated compatibility coefficient tuning.


## ASCII Maze Example: 6‑Input Long-Range Vision (MazeVision)

The ASCII maze example uses a compact 6‑input perception schema ("MazeVision") with long‑range lookahead via a precomputed distance map. Inputs (order fixed):

1. compassScalar: Encodes the direction of the globally best next step toward the exit as a discrete scalar in {0,0.25,0.5,0.75} corresponding to N,E,S,W. Uses an extended horizon (H_COMPASS=5000) so it can see deeper than openness ratios.
2. openN
3. openE
4. openS
5. openW
6. progressDelta: Normalized recent progress signal around 0.5 ( >0.5 improving, <0.5 regressing ).

### Openness Semantics (openN/E/S/W)

Each openness value describes the quality of the shortest path to the exit if the agent moves first in that direction, using a bounded lookahead horizon H=1000 over the distance map.

Value encoding:
* 1: Direction(s) whose total path length Ldir is minimal among all strictly improving neighbors (ties allowed; multiple 1s possible).
* Ratio 0 < Lmin / Ldir < 1: Direction is a valid strictly improving path but longer than the best (Lmin is the shortest improving path cost; Ldir = 1 + distance of neighbor cell). This supplies graded preference rather than binary pruning.
* 0: Wall, unreachable cell, dead end, or any non‑improving move (neighbor distance >= current distance) – all treated uniformly.
* 0.001: Special back‑only escape marker. When all four openness values would otherwise be 0 but the opposite of the previous successful action is traversable, that single opposite direction is set to 0.001 to indicate a pure retreat (pattern e.g. [0,0,0,0.001]).

Rules / Notes:
* Strict improvement filter: Only neighbors whose distanceMap value is strictly less than the current cell distance are considered for 1 or ratio values.
* Horizon clipping: Paths with Ldir > H are treated as unreachable (value 0) to bound search cost.
* Multiple bests: Corridors that fork into equivalently short routes produce multiple 1s, encouraging neutrality across equally optimal choices.
* Backtrack marker is intentionally very small (0.001) so evolution distinguishes "retreat only" states from true walls without overweighting them.
* Supervised refinement dataset intentionally contains ONLY deterministic single‑path cases (exactly one openness=1, others 0) for clarity; richer ratio/backtrack patterns appear only in the Lamarckian / evolutionary phase.

### progressDelta
Computed from recent distance improvement: delta = prevDistance - currentDistance, clipped to [-2,2], then mapped to [0,1] as 0.5 + delta/4. Values >0.5 mean progress toward exit; <0.5 regression or stalling.

### Debugging
Set ASCII_VISION_DEBUG=1 to emit periodic vision lines: current position, compassScalar, input vector, and per‑direction distance/ratio breakdown for auditing mismatches between maze geometry and distance map.

### Quick Reference
| Signal | Meaning |
|--------|---------|
| 1 | Best strictly improving path(s) (minimal Ldir) |
| (0,1) | Longer but improving path (ratio Lmin/Ldir) |
| 0.001 | Only backtrack available (opposite of prior action) |
| 0 | Wall / dead end / non‑improving / unreachable |

Implementation: `test/examples/asciiMaze/mazeVision.ts` (function `MazeVision.buildInputs6`).

This design minimizes input size (6 vs earlier large encodings) while preserving directional discrimination and long‑range planning cues, aiding faster evolutionary convergence and avoiding overfitting to local dead‑end noise.

## Roadmap / Backlog
Planned or partially designed enhancements not yet merged:
* Structural motif diversity pressure: penalize over-represented connection patterns (entropy-based sharing) to sustain innovation.
* Automated compatibility coefficient tuning: search or adapt excess/disjoint/weight coefficients to stabilize species counts without manual calibration.
* Faster Pareto sorting: divide-and-conquer or incremental dominance maintenance to reduce O(N^2) overhead for large populations.
* Connection complexity budget (current budget targets nodes only) and dual-objective weighting option.
* Diversity-aware parent selection leveraging motif entropy and archive dispersion.
* Extended novelty descriptors helper utilities (e.g. built-in graph metrics: depth, feedforwardness, clustering).
* Visualization hooks (species lineage graph, archive embedding projection) for diagnostics.
