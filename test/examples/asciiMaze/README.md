# ASCII Maze (Educational Neuroevolution Example)

This folder contains a full **NEAT-driven maze-solving system** built as an educational example for NeatapticTS.

It demonstrates how to combine:

- **Perception** (`MazeVision`) — convert maze state into compact neural inputs.
- **Policy execution** (`MazeMovement`) — run an agent episode from network outputs.
- **Reward shaping** (`FitnessEvaluator`) — score behavior beyond simple win/lose.
- **Evolution orchestration** (`EvolutionEngine`) — evolve populations with telemetry, persistence, and curriculum transfer.
- **Optional supervised fine-tuning** (`NetworkRefinement` / `refineWinner`) — improve winners with backprop.

---

## Why this example is valuable

Unlike toy XOR-style demos, this example includes real-world concerns that appear in production ML systems:

- non-trivial environment state
- shaped rewards and anti-collapse heuristics
- adaptive evolution controls (plateau handling, simplification)
- curriculum transfer (solve small mazes first, then scale)
- browser + terminal visualization
- test integration and deterministic modes

Use this as a reference architecture when building your own task-specific neuroevolution pipelines.

---

## Folder map (what lives where)

### Core runtime

- `evolutionEngine.ts`
  - Public façade entry point: `EvolutionEngine.runMazeEvolution(options)`.
  - Delegates to modular files under `evolutionEngine/`.
- `evolutionEngine/`
  - `optionsAndSetup.ts`: normalize options, prepare maze/distance map, create NEAT instance.
  - `evolutionLoop.ts`: generation loop, cancellation, stop conditions, telemetry flow.
  - `populationDynamics.ts`: pruning, simplify phase, anti-collapse, dynamic population.
  - `trainingWarmStart.ts`: warm-start + Lamarckian/Baldwinian helpers.
  - `telemetryMetrics.ts`: generation metrics collection/log formatting.
  - `rngAndTiming.ts`, `scratchPools.ts`, `sampling.ts`, etc.: performance + deterministic helpers.

### Maze simulation pipeline

- `mazeVision.ts`
  - Builds 6D inputs for the policy network.
- `mazeMovement.ts`
  - Simulates one episode with action selection, movement, penalties/rewards, and final run result.
- `fitness.ts`
  - Converts simulation outcomes into scalar fitness.
- `mazeUtils.ts`
  - Encoding, BFS distance, progress calculations, coordinate utilities.

### Visualization and UX

- `dashboardManager.ts`
  - Rich per-generation dashboard with telemetry history, archive of solved mazes, and trend snapshots.
- `mazeVisualization.ts`
  - Colorized maze rendering and summary stats.
- `networkVisualization.ts`
  - Network topology formatting/inspection.
- `terminalUtility.ts` / `browserTerminalUtility.ts`
  - Rendering primitives for Node terminal and browser DOM.
- `browserLogger.ts`
  - Browser-target logging utility.

### Scenario definitions and adapters

- `mazes.ts`
  - Static mazes (`tiny`, `small`, `medium`, `large`, `minotaur`) plus procedural `MazeGenerator`.
- `interfaces.ts`
  - Canonical contracts: run options, telemetry contracts, network/visualization types.
- `index.ts` and `asciiMaze.ts`
  - Re-exports for easier imports.

### Demo and test entry points

- `browser-entry.ts`
  - Public `start(...)` API for browser demo lifecycle.
- `index.html`
  - Simple host page loading `docs/assets/ascii-maze.bundle.js`.
- `asciiMaze.e2e.test.ts`
  - Curriculum-style end-to-end evolution in test form.
- `networkRefinement.ts` and `refineWinner.ts`
  - Two refinement helpers (class-based and functional style).

---

## Conceptual architecture

```text
ASCII Maze + Agent State
				|
				v
MazeVision.buildInputs6
				|
				v
Network.activate (4 outputs)
				|
				v
MazeMovement.simulateAgent
				|
				v
FitnessEvaluator.evaluateNetworkFitness
				|
				v
EvolutionEngine generation loop
				|
				+--> Dashboard/Telemetry
				+--> Persistence (optional)
				+--> Curriculum seed for next maze
```

---

## Input and output encoding (important)

### Policy input vector (6 values)

`MazeVision` produces:

1. `compassScalar` — compressed compass direction to exit (`0, 0.25, 0.5, 0.75` for N/E/S/W)
2. `openN`
3. `openE`
4. `openS`
5. `openW`
6. `progressDelta` — clipped/scaled progress signal between steps

This keeps the network small and learnable while preserving directional context.

### Policy outputs (4 values)

The network outputs action logits/probabilities for:

1. North
2. East
3. South
4. West

`MazeMovement` applies softmax-style action interpretation and tracks entropy/saturation diagnostics.

---

## Episode scoring (fitness intuition)

`FitnessEvaluator` composes several signals:

- base movement/progress score from simulation
- exploration bonus for uniquely visited cells
- proximity weighting (exploring near promising regions helps more)
- success bonus for reaching the exit
- efficiency bonus for short successful paths vs optimal BFS baseline

Why this is educationally useful:

- It shows how **reward shaping** can guide sparse-goal tasks.
- It highlights tradeoffs between exploration and path efficiency.

---

## Evolution flow and stopping behavior

At a high level, `EvolutionEngine.runMazeEvolution` performs:

1. Option normalization and defaults (`optionsAndSetup.ts`)
2. Maze preparation (encoding, start/exit detection, distance map)
3. NEAT creation and optional warm-start seeding
4. Generation loop:
   - evaluate population
   - apply adaptive/population dynamics
   - apply optional refinement phases
   - log/update dashboard/telemetry
5. Stop when one of these occurs:
   - solved threshold reached
   - stagnation cap hit
   - max generations reached
   - cancellation/abort requested

The engine supports deterministic mode, telemetry toggles, persistence intervals, and dynamic population controls.

---

## Browser demo API

`browser-entry.ts` exports:

- `start(container?, opts?) => Promise<AsciiMazeRunHandle>`

The handle provides:

- `stop()`
- `isRunning()`
- `done` promise
- telemetry subscribe/unsubscribe
- `getTelemetry()` snapshot access

The browser demo runs a curriculum that scales procedural mazes from small to larger dimensions while optionally carrying forward best networks.

---

## Run the example

From repository root:

### 1) Type-check (quick sanity)

```bash
npx tsc --noEmit -p tsconfig.json
```

### 2) Run e2e curriculum test with logs

```bash
npm run test:e2e:logs
```

### 3) Build browser bundle for docs/assets

```bash
npm run build:ascii-maze
```

### 4) Build docs (includes example asset copy + docs rendering)

```bash
npm run docs
```

Node engine requirement in this repo is `>=22`.

---

## Minimal integration example (programmatic)

```ts
import { EvolutionEngine } from './evolutionEngine';
import { MazeGenerator } from './mazes';
import { DashboardManager } from './dashboardManager';
import { TerminalUtility } from './terminalUtility';

const dashboard = new DashboardManager(
  TerminalUtility.createTerminalClearer(),
  (...args) => console.log(...args),
);

const result = await EvolutionEngine.runMazeEvolution({
  mazeConfig: { maze: new MazeGenerator(24, 24).generate() },
  agentSimConfig: { maxSteps: 2000 },
  evolutionAlgorithmConfig: {
    popSize: 40,
    maxGenerations: 100,
    maxStagnantGenerations: 50,
    minProgressToPass: 95,
    allowRecurrent: true,
  },
  reportingConfig: {
    dashboardManager: dashboard,
    logEvery: 1,
    label: 'demo-24x24',
  },
});

console.log(result.exitReason, result.bestResult?.progress);
```

---

## Customization guide (safe starting points)

If you’re teaching or experimenting, these knobs are typically most impactful first:

1. `agentSimConfig.maxSteps`
   - increase for larger/harder mazes
2. `evolutionAlgorithmConfig.popSize`
   - larger population improves search breadth but costs compute
3. `evolutionAlgorithmConfig.maxGenerations`
   - higher cap for difficult layouts
4. `evolutionAlgorithmConfig.minProgressToPass`
   - solved threshold sensitivity
5. `lamarckianIterations` and `lamarckianSampleSize`
   - adjust local supervised-style refinement pressure
6. `deterministic` + `randomSeed`
   - reproducibility for educational comparisons

Then, for advanced learners:

- experiment with `fitnessEvaluator` override
- study `mazeMovement.ts` penalty/reward constants
- enable/inspect telemetry trends in `DashboardManager`

---

## Educational exercises

Try these in order:

1. **Perception ablation**
   - Remove one input channel from `MazeVision` and observe learning degradation.
2. **Reward shaping experiment**
   - Reduce exploration bonus and track effects on dead-end behavior.
3. **Curriculum comparison**
   - Train directly on big mazes vs phased growth with transfer.
4. **Determinism study**
   - Fix seed and compare run-to-run variance when toggling certain heuristics.
5. **Refinement impact**
   - Compare before/after `NetworkRefinement.refineWinnerWithBackprop` on transfer tasks.

---

## Common pitfalls

- **No `S` or `E` in maze**: position lookup throws during setup.
- **Inconsistent row lengths**: can break assumptions in encoding/path code.
- **Overly harsh penalties**: can collapse exploration early.
- **Tiny `maxSteps` on large mazes**: policy never gets enough trajectory to improve.
- **Expecting instant convergence**: evolution is stochastic even with strong shaping.

---

## Where to start reading the code

Recommended order:

1. `interfaces.ts` (contracts)
2. `mazeVision.ts` (inputs)
3. `mazeMovement.ts` (episode simulation)
4. `fitness.ts` (score)
5. `evolutionEngine.ts` + `evolutionEngine/` modules (training loop)
6. `asciiMaze.e2e.test.ts` (real usage pattern)
7. `browser-entry.ts` (embedding API + telemetry consumption)

This progression mirrors the actual runtime path and makes the design easier to internalize.

---

## Notes on design philosophy

This example intentionally favors:

- explicit interfaces and documentation-heavy types
- modular orchestration over monolithic loops
- optimization where it helps educational performance (scratch buffers, typed arrays)
- practical instrumentation (telemetry, trend summaries, archive views)

It is both a **teaching artifact** and a **stress test** for integrating evolution, simulation, and visualization in one cohesive workflow.
