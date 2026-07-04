# Public library demo-agnostic audit

## Question

Does the NeatapticTS public library (`src/`) contain any naming, JSDoc, or design references that tie it to specific demos (`flappy_bird`, `racing_curriculum`, `neatChat`, NGE benchmarks, etc.) rather than expressing the feature as a general library capability? If so, where are the violations, how severe are they, and what is the safest remediation?

## Evidence

### Discovery methodology

1. Ran Cortex MCP `freshness_check` and `search_corpus` for demo-specific terms inside `src/`.
2. Cross-referenced `examples/` and `docs/browser-tests/` to establish the repo's demo vocabulary.
3. Used `git grep` with fixed-string patterns to enumerate every `src/` occurrence of the demo-derived terms.
4. Read representative production `.ts` files to confirm export status, exact line content, and whether the term was a general ML/concurrency concept or a demo reference.

Cortex index was stale at audit time (`cortex-index` gate `pass: false`, `fixHint: node rag-index/build-index.mjs`), so native `git grep` and `view` were used as fallback for exact line numbers.

### Demo vocabulary in this repo

| Demo / application | Location | Key demo-derived terms |
|---|---|---|
| **Flappy Bird** | `examples/flappy_bird/` | flappy, bird, flap, pipe, generation, network-view, shared-inference worker |
| **Racing Curriculum** | `examples/racing_curriculum/` | racing, race, car, vehicle, track, circuit, race-pack, race-step, team-radio, opponent snapshot |
| **NEATchat** | `examples/neatChat/` | NEATchat, neatchat, chat, tokenizer, corpus, session, memory-bank |
| **ASCII Maze** | `examples/asciiMaze/` | asciiMaze, maze, cell, wall, agent, mazeVision |
| **NGE browser benchmarks** | `docs/browser-tests/webgpu-nge-tier-*.html`, `docs/browser-tests/scenarios/*` | NGE tier benchmark, tier, ladder |
| **Predator/Prey (planned)** | `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` | predator/prey, ant-hive, angel |
| **Ant Hive (planned)** | `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md` | ant-hive, ant hive, pheromone, EVA/Angel |
| **Starter demos** | `examples/helloNetwork/`, `examples/evolveXor/`, `examples/sequenceReset/` | helloNetwork, evolveXor, sequenceReset |

The NGE (`Neuro-evolutionary Genesis Engine` / `Neuro-Genesis Engine`) subsystem in `src/neat/nge-*` is exposed as a public experimental namespace. The subsystem name itself is general; however, its source JSDoc repeatedly uses `racing`, `ant-hive`, and `predator` as concrete examples, which makes the documentation read as if the subsystem is a scaffold for those demos rather than a demo-agnostic library primitive.

### Violations by severity

#### CRITICAL — file names and public API names

| File | Line | Offending content | Why demo-specific | Recommended fix |
|---|---|---|---|---|
| `src/architecture/network/gpu/network.gpu.racing.ts` | filename | `network.gpu.racing.ts` | File is named after the racing demo. | Rename to `network.gpu.batch-evaluation.ts` (or `network.gpu.concurrent.ts`). |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 10 | `export interface RacingBatchOptions` | Public symbol named after racing. | Rename to `BatchEvaluationOptions` / `ConcurrentBatchOptions`. |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 141 | `export async function evaluateRacingGeneration` | Public function named after racing. | Rename to `evaluateBatchGeneration` / `evaluateGeneration`. |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 160 | `export interface RacingAgentRequest` | Public symbol named after racing. | Rename to `AgentEvaluationRequest` / `ConcurrentAgentRequest`. |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 180 | `export async function evaluateConcurrentRacingAgents` | Public function named after racing. | Rename to `evaluateConcurrentAgents`. |
| `src/architecture/network/gpu/network.gpu.racing.test.ts` | filename | `network.gpu.racing.test.ts` | Test file named after the racing demo. | Rename alongside the module it tests. |

These are breaking public API changes. Any consumer importing from `network.gpu.racing` or using the `Racing*` symbols will break until they migrate.

#### HIGH — JSDoc / public examples that name specific demos

| File | Line | Offending content | Why demo-specific | Recommended fix |
|---|---|---|---|---|
| `src/architecture/network/gpu/network.gpu.racing.ts` | 8 | `Options controlling batch evaluation in the racing-curriculum worker seam.` | References racing-curriculum worker. | `Options controlling batch evaluation for multi-agent GPU seams.` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 23-25 | `The racing-curriculum worker can override this through RacingBatchOptions.gpuBatchThreshold...` | Caller described as racing-curriculum worker. | `Callers can override this through BatchEvaluationOptions.gpuBatchThreshold...` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 32 | `Reused from the kernel contract so the racing eligibility check stays in sync...` | `racing eligibility check`. | `batch eligibility check`. |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 61 | `Decide whether the racing generation can use the batched GPU path.` | `racing generation`. | `Decide whether the batched GPU path should be used for this generation.` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 126 | `Racing-curriculum generation evaluation seam.` | `Racing-curriculum`. | `Batch generation evaluation seam.` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 133 | `@param networks — One network per car / genome in the generation.` | `car` is demo-specific. | `One network per agent / genome in the generation.` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 158 | `Single concurrent racing evaluation request.` | `racing`. | `Single concurrent agent evaluation request.` |
| `src/architecture/network/gpu/network.gpu.racing.ts` | 166 | `Evaluate many racing agents in parallel on the GPU.` | `racing agents`. | `Evaluate many agents in parallel on the GPU.` |
| `src/architecture/network/gpu/network.gpu.batched.ts` | 5 | `...when the racing-curriculum worker or another demo needs to score a whole generation at once.` | Names racing-curriculum worker and `demo`. | `...when a caller needs to score a whole generation at once.` |
| `src/architecture/network/gpu/network.gpu.batched.ts` | 12 | `...through evaluateRacingGeneration or a caller-local fallback.` | References racing-specific function. | Update after `evaluateRacingGeneration` is renamed. |
| `src/architecture/network/gpu/network.gpu.batched.ts` | 38 | `...such as the racing-curriculum worker controller...` | Names racing-curriculum worker. | `...such as a multi-agent worker controller...` |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 9 | `...benchmarks such as racing, predator/prey, and ant-hive all need to ship initial state...` | Lists specific demos as examples. | `...benchmarks and multi-agent evaluations all need to ship initial state...` |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 24-26 | `It does not know about racing frames, opponent snapshots, or track physics...` | Racing-specific frame/state names. | `It does not know about benchmark frames, opponent snapshots, or domain physics...` |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 42-43 | `Wrap --> Frame[benchmark frame<br/>RacingRenderFrame]` | Mermaid names `RacingRenderFrame`. | Use generic `BenchmarkRenderFrame` or remove the concrete type from the diagram. |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 115-117 | `Benchmark-specific inputs (opponent snapshots, track physics, etc.) are injected...` | `track physics` is racing-specific. | `domain physics` / `benchmark-specific physics`. |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 216-217 | `This is the generic version of the racing-specific assertRacingSchemaVersion...` | References a racing-specific function by name. | `This is the generic version of the original demo-specific schema-version guard...` |
| `src/architecture/network/visualization/network.visualization.ts` | 18-19 | `In-browser canvas renderers (e.g. the Flappy Bird network panel). Terminal ASCII renderers (e.g. ASCII Maze).` | Names Flappy Bird and ASCII Maze in public JSDoc. | `In-browser canvas renderers (e.g. a live dashboard network panel). Terminal ASCII renderers (e.g. a grid-world monitor).` |
| `src/architecture/network/worker-payload/network.worker-payload.browser-url.ts` | 31 | `'flappy-shared-inference.worker.bundle.js'` | Example asset named after the flappy demo. | `'myapp-shared-inference.worker.bundle.js'` or a generic placeholder. |
| `src/neat/export/neat.export.ts` | 137 | `...consumers such as NEATchat can attach namespaced descriptors...` | Names NEATchat in public JSDoc. | `...downstream consumers can attach namespaced descriptors...` |
| `src/neat/export/neat.export.ts` | 401, 443 | `neatchat: { memoryBankId / branchId }` | Examples use the demo namespace key. | Use a generic namespace key such as `myApp` or `consumer`. |
| `src/neat.ts` | 1286, 1317 | `neatchat: { memoryBankId / branchId }` | Examples on public `exportState`/`exportLightState` use demo namespace. | Use generic `myApp` / `consumer`. |
| `src/neat/neat.nge-lifecycle.ts` | 13 | `...that a racing curriculum or ant hive would use at runtime.` | Names racing and ant hive. | `...that a multi-agent benchmark would use at runtime.` |
| `src/neat/nge-collective/neat.nge-collective.shared-field.ts` | 14 | `...contract required by ant-hive pheromone trails and racing team-radio channels.` | Names ant-hive and racing. | `...contract required by stigmergy fields such as pheromone trails or team-communication channels.` |
| `src/neat/nge-collective/neat.nge-collective.team-fitness.ts` | 9 | `...racing, ant-hive, and future collective benchmarks share one evaluator contract...` | Lists specific demos. | `...collective benchmarks share one evaluator contract...` |
| `src/neat/nge-collective/neat.nge-collective.team-fitness.ts` | 20-29 | Mermaid nodes named `Racing`, `Ant Hive`, plus `Racing is the first proven consumer...` | Public JSDoc diagram and prose name specific demos. | Rename nodes to `Consumer A`, `Consumer B`; replace prose with generic consumer description. |
| `src/neat/nge-collective/neat.nge-collective.team-fitness.ts` | 38 | `Deterministic race-pack transport` | `race-pack` is racing-specific. | `Deterministic episode-pack transport` / `deterministic transport shell`. |
| `src/neat/nge-collective/neat.nge-collective.team-fitness.ts` | 56 | `...racing, ant-hive, and future consumers on one shared evaluator contract...` | Lists specific demos. | `...consumers on one shared evaluator contract...` |
| `src/neat/nge-collective/neat.nge-collective.ts` | 24 | Mermaid `Consumers["Racing · Ant Hive\nbenchmark consumers"]` | Names specific demos in a public diagram. | `Consumers["Collective benchmark consumers"]` |
| `src/neat/nge-collective/neat.nge-collective.ts` | 33-34 | `...stigmergy contract required by the ant-hive and racing benchmarks.` | Names ant-hive and racing. | `...stigmergy contract required by collective benchmarks.` |
| `src/neat/nge-collective/neat.nge-collective.ts` | 116 | `// Racing and Ant Hive each supply their own policy...` | Inline example names specific demos. | `// Consumers each supply their own policy...` |
| `src/neat/nge-collective/neat.nge-collective.types.ts` | 146-148 | `Consumers such as racing or ant-hive own the scoring policy...` | Names specific demos. | `Consumers such as multi-agent benchmarks own the scoring policy...` |
| `src/neat/nge-dna/neat.nge-dna.ts` | 6 | `...expands racing-worker shorthand values into the canonical envelope shape...` | `racing-worker shorthand`. | `...expands consumer shorthand values into the canonical envelope shape...` |
| `src/neat/nge-dna/neat.nge-dna.ts` | 19 | `...core accepts racing values at input...` | `racing values`. | `...core accepts shorthand values at input...` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 11 | `...accepts a small set of racing-worker shorthand values at input time...` | `racing-worker shorthand`. | `...accepts a small set of consumer shorthand values at input time...` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 18 | `...racing-worker reference name for the deterministic single-drone-per-region assignment...` | `racing-worker reference name`. | `...consumer shorthand for the deterministic single-drone-per-region assignment...` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 24 | `...core accepts racing values at input...` | `racing values`. | `...core accepts shorthand values at input...` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 70 | `## Racing-worker compatibility` | Heading names racing worker. | `## Consumer shorthand compatibility` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 72 | `...racing-worker reference name for the deterministic single-drone-per-region assignment...` | `racing-worker reference name`. | `...consumer shorthand...` |
| `src/neat/nge-dna/neat.nge-dna.types.ts` | 131 | `...core accepts racing values at input...` | `racing values`. | `...core accepts shorthand values at input...` |
| `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts` | 8 | `## Racing-worker compatibility` | Heading names racing worker. | `## Consumer shorthand compatibility` |
| `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts` | 10 | `...racing-worker reference name for the deterministic single-drone-per-region assignment...` | `racing-worker reference name`. | `...consumer shorthand...` |
| `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts` | 15 | `...preserves the racing-worker shorthand...` | `racing-worker shorthand`. | `...preserves the consumer shorthand...` |
| `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts` | 782 | `Expand racing-worker seed-governance shorthand into the canonical policy shape.` | `racing-worker`. | `Expand consumer seed-governance shorthand into the canonical policy shape.` |
| `src/neat/nge-juvenile/neat.nge-juvenile.ts` | 16-17 | `...lets the same engine run inside a racing curriculum, an ant hive, a predator simulation...` | Lists specific demos. | `...lets the same engine run inside any multi-agent benchmark...` |
| `src/visualization/network-view/network-view.types.ts` | 5-7 | `Demo-specific overlays (Flappy input bands, ASCII Maze labels, etc.)...` | Names Flappy and ASCII Maze. | `Consumer-specific overlays (input bands, layer labels, etc.)...` |
| `src/visualization/network-view/network-view.types.ts` | 81-82 | `Flappy Bird injects input-group label bands and per-input descriptions. ASCII Maze could inject custom layer labels...` | Names specific demos. | `A consumer can inject input-group label bands and per-input descriptions; another can inject custom layer labels...` |
| `src/visualization/network-view/network-view.types.ts` | 87 | `...for example, Flappy Bird uses this hook to add input-group label bands.` | Names Flappy Bird. | `...for example, a dashboard can use this hook to add input-group label bands.` |

#### MEDIUM — internal comments / parameter names with demo coupling

| File | Line | Offending content | Why demo-specific | Recommended fix |
|---|---|---|---|---|
| `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` | 89-91 | `Benchmark-specific episode state such as opponentSnapshot, trackId, or featureFlags...` | `trackId` is racing-specific. | `episodeId` / `domainId` / `featureFlags`. |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 13-18 | Constants and comments use `cars`, `teammate-radio channels per car`. | `cars` and `radio` tied to racing. | Use `agents`, `slots`, `communication channels per agent`. |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 51 | `Shared scaffold for the smallest honest 2v2 race-pack harness.` | `race-pack`. | `2v2 multi-agent evaluation harness.` |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 77 | `...matches the race-pack contract used by the example worker seam...` | `race-pack` / `example worker seam`. | `...matches the episode-pack contract used by consumer worker seams...` |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 152 | `Produces the Phase 3 evaluation scaffold for one 2v2 race tick.` | `race tick`. | `Produces the Phase 3 evaluation scaffold for one 2v2 episode tick.` |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 155 | `...so the surrounding racing example can exercise the two-population seam...` | `surrounding racing example`. | `...so a consumer benchmark can exercise the two-population seam...` |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 161, 174, 187, 195 | Parameter/field named `raceState` and propagated `raceState`. | `raceState`. | Rename to `episodeState` / `tickState` / `sharedState`. |
| `src/neat/nge-collective/neat.nge-collective.two-population.ts` | 223 | `...packed race-step frames, transfer lists, or worker topology.` | `race-step frames`. | `episode-step frames` / `tick frames`. |
| `src/neat/nge-collective/neat.nge-collective.ts` | 167 | `// --- Two-population racing scaffold ---` | `racing scaffold`. | `// --- Two-population benchmark scaffold ---` |

#### LOW — generated READMEs, tests, and docs metadata

| File | Reason |
|---|---|
| `src/**/README.md` (generated) | Mirror the JSDoc violations above. Must be regenerated with `npm run docs` after source JSDoc is fixed, not hand-edited. |
| `src/architecture/network/gpu/README.md` | Contains `network.gpu.racing.ts` heading and all `Racing*` symbol docs. |
| `src/architecture/network/gpu/docs.order.json:11` | Lists `network.gpu.racing.ts`; update when renaming. |
| `src/architecture/network/evaluation-pack/README.md` | Mirrors `network.evaluation-pack.ts` demo references. |
| `src/architecture/network/visualization/README.md` | Mirrors Flappy Bird / ASCII Maze renderer examples. |
| `src/architecture/network/worker-payload/README.md` | Mirrors `flappy-shared-inference.worker.bundle.js` example. |
| `src/neat/README.md`, `src/neat/export/README.md`, `src/neat/nge-*/README.md`, `src/visualization/README.md`, `src/visualization/network-view/README.md` | Mirror the NGE / NEATchat / Flappy / ASCII Maze references. |
| `src/architecture/network/gpu/network.gpu.parity.test.ts:50` | `describe('racing-browser scale network ...)'` uses racing-browser demo framing. Rename to a scale descriptor, e.g. `large-scale dense network`. |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts:26` | `/** Minimal transport-neutral inputs (no racing-specific fields). */` — test comment names racing. Rewrite to `/** Minimal transport-neutral inputs (no benchmark-specific fields). */`. |
| `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts:125` | `describe('resolveTransferList — race-step transport', ...)` — `race-step`. Use `episode-step transport`. |
| `src/neat/export/neat.export.test.ts:1258, 2104` | Test fixtures use `neatchat:` namespace key. Use generic `consumer:` key. |
| `src/neat/nge-collective/neat.nge-collective.two-population.test.ts` | Uses `createRaceStateFixture()` helper and `raceState: unknown` fields. Rename to `createEpisodeStateFixture()` / `episodeState`. |
| `src/visualization/network-view/network-view.test.ts:4` | Comment: `Full browser rendering tests are integrated via Flappy Bird and ASCII Maze demos.` Use generic language about consumer integration tests. |

## Decision

The public library (`src/`) is **not demo-agnostic**. The most severe violation is a demo-named module and exported symbols in `src/architecture/network/gpu/network.gpu.racing.ts`. Additional clusters of demo-specific JSDoc and examples appear in the GPU batched seam, the evaluation-pack contract, NGE collective/evaluation/DNA/evolution modules, the export/checkpoint extension docs, and the visualization types.

**Recommended remediation path:**

1. **Rename the critical surface first.** Rename `src/architecture/network/gpu/network.gpu.racing.ts` and its test to a generic multi-agent/batch GPU name, and rename the exported `Racing*` / `evaluateRacing*` symbols to generic equivalents (`BatchEvaluationOptions`, `evaluateBatchGeneration`, `AgentEvaluationRequest`, `evaluateConcurrentAgents`).
2. **Rewrite JSDoc and public examples.** Remove every concrete demo name from public JSDoc, `@example` blocks, and mermaid diagrams. Replace them with generic consumers, benchmarks, or placeholder namespaces (`myApp`, `consumer`, `benchmark`).
3. **Rename internal identifiers where they carry demo semantics.** In `neat.nge-collective.two-population.ts`, replace `car`, `race-pack`, `race tick`, and `raceState` with `agent`, `episode-pack`, `episode tick`, and `episodeState`.
4. **Update tests and fixtures.** Rename test files, helper functions, and fixture keys that embed demo names. Update import paths and assertions.
5. **Regenerate generated docs.** Run `npm run docs` so that all `src/**/README.md` files reflect the sanitized source. Do not hand-edit generated READMEs.
6. **Update docs metadata.** Update `src/architecture/network/gpu/docs.order.json` to reference the renamed file.
7. **Verify no regressions.** Run targeted tests for the affected folders and lint/type checks.

This is a breaking API refactor for any external consumer of the racing-named GPU functions, but it is necessary to keep the public library demo-agnostic.

## Risks

| Risk | Owner / mitigation |
|---|---|
| **Breaking public API change.** Renaming `network.gpu.racing.ts` and the `Racing*` exports will break any external consumer or downstream demo that imports them. A migration note must accompany the release. | Implementation phase / release notes. |
| **Generated README drift.** If `npm run docs` is not run after JSDoc edits, the generated `src/**/README.md` files will continue to display demo references. | 06-documenting / build step. |
| **Test fixture coupling.** Tests inside `src/` use the same demo names and import paths; co-renaming is required or tests will fail to compile. | 04-implementing / 05-green-testing. |
| **NGE subsystem perception.** Although NGE is a general experimental namespace, its documentation currently reads as racing/ant-hive/predator scaffolding. Sanitizing the docs is essential or the public API will still feel demo-derived even after file renames. | 04-implementing. |
| **Scope creep into examples/.** The audit deliberately excludes `examples/` and `docs/browser-tests/`, which are allowed to be demo-specific. Implementation must not refactor demo directories; only `src/` should change. | 04-implementing / boundary review. |
| **Cortex index stale during research.** The semantic index was stale; the audit relied on `git grep` and direct file reads, which is sufficient for exact line-level evidence but should be refreshed before broader semantic searches. | `00-helping` / `repo-cortex-workflow` (run `node rag-index/build-index.mjs`). |
