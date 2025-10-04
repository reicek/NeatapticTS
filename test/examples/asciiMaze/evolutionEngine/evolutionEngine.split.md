# EvolutionEngine Refactor Checklist

> Goal: dissolve the monolithic `evolutionEngine.ts` into focused ES2023 modules while reducing the original file to a thin, well-documented façade that simply wires the subsystems together.
> Progress log: update this checklist immediately after each step so the team can pause or roll back safely.

## Phase 0 · Safety Nets

- [x] Capture current behaviour with smoke tests or snapshots (maze solve happy path, telemetry log sample). _(Baseline commit recorded; proceeding without new tests per instructions.)_
- [x] Freeze public entry points (`runMazeEvolution`, `setDeterministic`, `clearDeterministic`, `printNetworkStructure`) as the only exports that remain attached to the façade.

## Phase 1 · `engineState.ts`

- [x] Introduce `EngineState` shape exporting pooled buffers and constants (initial scratch set extracted; expand with remaining fields in follow-up steps).
- [x] Move buffer initialisers (`#SCRATCH_EXPS` → `initialScratchState` helper, etc.) into pure factory functions.
		- [x] Progress: shared state now documents every pooled buffer/toggle, centralizes sample/string pools, small explore table, visited-hash state (table + load factor), node index pool, and shared/non-shared logits buffers, and now exposes `createScratchState`/`createToggleState` helpers.
					- [x] Extract telemetry stats initializer helper (`initialiseTelemetryScratch`) from the setup paths in `#logOutputBiasStats` / entropy helpers.
					- [x] Extract visited-hash growth helper (`ensureVisitedHashCapacity`) from inline sizing inside `#countDistinctCoordinatesHashed`.
					- [x] Extract RNG cache initializer + reseed helper from the `#fastRandom` deterministic-path setup.
					- [x] Update façade wiring to invoke the new helpers during setup.
- [x] Document `engineState` responsibilities once helpers are extracted (top-of-file summary).
- [x] Add typed getters/setters for toggles (`#REDUCED_TELEMETRY`, `#DISABLE_BALDWIN`, `#TELEMETRY_MINIMAL`).
- [x] Update façade to instantiate one shared `EngineState` and pass it down.

## Phase 2 · `rngAndTiming.ts`

- [x] Move `#PROFILE_T0`, `#PROFILE_ADD`, `#fastRandom`, `#now` into pure functions that accept `EngineState`.
- [x] Rehome `setDeterministic` / `clearDeterministic` to re-export thin wrappers that drive the shared state.
- [x] Replace direct static field access in callers with imports from this module (façade now delegates RNG/profiling to helpers).

## Phase 3 · `scratchPools.ts`

- [x] Extract `#allocateLogitsRing`, `#initSharedLogitsRing`, `#ensureLogitsRingCapacity` (now provided by `scratchPools.ensureLogitsRingCapacity`).
- [x] Migrate `#ensureScratchCapacity`, `#maybeShrinkScratch`, `#ensureConnFlagsCapacity` into shared helpers backed by `EngineState`.
- [x] Update façade call sites to consume the new helpers (logits ring upkeep, deterministic pooling, and recurrent-flag detection now flow through the shared-state module).

## Phase 4 · `sampling.ts`

- [x] Relocate `#sampleArray`, `#sampleIntoScratch`, `#sampleSegmentIntoScratch`.
- [x] Move history helpers `#getTail`, `#pushHistory`.
- [x] Adjust telemetry/population modules to consume the new exports.
- [x] Describe pooled sampling responsibilities at the top of `sampling.ts`.

## Phase 5 · `telemetryMetrics.ts`

- [ ] Move orchestration method `#logGenerationTelemetry`.
- [ ] Sequentially extract helpers:
	- [ ] `#logActionEntropy`
	- [ ] `#computeActionEntropy`
	- [ ] `#logOutputBiasStats`
	- [ ] `#computeOutputBiasStats`
	- [ ] `#logLogitsAndCollapse`
	- [ ] `#computeLogitStats` (+ `#resetLogitScratch`, `#accumulate*`, `#finalize*`)
	- [ ] `#computeDecisionStability`
	- [ ] `#softmaxEntropyFromVector`
	- [ ] `#logExploration`
	- [ ] `#computeExplorationStats`, `#countDistinctCoordinatesTiny`, `#countDistinctCoordinatesHashed`
	- [ ] `#logDiversity`
	- [ ] `#computeDiversityMetrics`
	- [ ] `#collectTelemetryTail`
	- [ ] `#joinNumberArray`
- [ ] Relocate collapse heuristics (`#antiCollapseRecovery`, thresholds) here or to population module as needed.

## Phase 6 · `populationPruning.ts`

- [ ] Extract `#applySimplifyPruningToPopulation`.
- [ ] Move pruning helpers in order:
	- [ ] `#pruneWeakConnectionsForGenome`
	- [ ] `#collectEnabledConnections`
	- [ ] `#collectHiddenToOutputConns`
	- [ ] `#sortCandidatesByStrategy`
	- [ ] `#insertionSortByAbsWeight`
	- [ ] `#disableSmallestEnabledConnections`
- [ ] Port compass warm-start & bias recentre helpers (`#applyCompassWarmStart`, `#centerOutputBiases`).

## Phase 7 · `populationDynamics.ts`

- [ ] Move generation-level helpers:
	- [ ] `#updatePlateauState`
	- [ ] `#handleSimplifyState`
	- [ ] `#runSimplifyCycle`
- [ ] Transfer population changes:
	- [ ] `#expandPopulation`
	- [ ] `#prepareExpansion`
	- [ ] `#determineMutateCount`
	- [ ] `#applyMutationsToClone`
	- [ ] `#registerClone`
	- [ ] `#createChildFromParent`
	- [ ] `#getSortedIndicesByScore`
	- [ ] `#insertionSortIndices`
	- [ ] `#medianOfThreePivot`
	- [ ] `#qsPushRange`
	- [ ] `#getMutationOps`
	- [ ] `#ensureOutputIdentity`
	- [ ] `#handleSpeciesHistory`
	- [ ] `#maybeExpandPopulation`
	- [ ] `#pruneSaturatedHiddenOutputs`
	- [ ] `#compactGenomeConnections`
	- [ ] `#compactPopulation`
- [ ] Rehome collapse recovery methods (`#antiCollapseRecovery`, `#reinitializeGenomeOutputsAndWeights`).

## Phase 8 · `trainingWarmStart.ts`

- [ ] Migrate Lamarckian pipeline:
	- [ ] `#warmStartPopulationIfNeeded`
	- [ ] `#buildLamarckianTrainingSet`
	- [ ] `#pretrainPopulationWarmStart`
	- [ ] `#applyLamarckianTraining`
	- [ ] `#adjustOutputBiasesAfterTraining`
	- [ ] `#applyCompassWarmStart` (if not already moved)
- [ ] Ensure helpers expose pure functions with explicit parameters (no hidden static usage).

## Phase 9 · `optionsAndSetup.ts`

- [ ] Relocate environment/setup methods:
	- [ ] `#makeFlushToFrame`
	- [ ] `#initPersistence`
	- [ ] `#makeSafeWriter`
	- [ ] `#createNeat`
	- [ ] `#seedInitialPopulation`
	- [ ] `#createAndSeedNeat`
	- [ ] `#prepareEnvironmentForRun`
	- [ ] `#normalizeRunOptions`
- [ ] Rewrite to accept configuration + `EngineState`, returning POJOs (no use of class statics).

## Phase 10 · `evolutionLoop.ts`

- [ ] Move orchestration logic:
	- [ ] `#makeFlushToFrame` (if not in setup), `#prepareLoopHelpers`
	- [ ] `#runGeneration`
	- [ ] `#simulateAndPostprocess`
	- [ ] `#checkCancellation`
	- [ ] `#checkStopConditions`
	- [ ] `#runEvolutionLoop`
- [ ] Remove direct static references; interact solely via imported helpers and `EngineState`.

## Phase 11 · `networkInspection.ts`

- [ ] Migrate developer tooling:
	- [ ] `printNetworkStructure`
	- [ ] `#classifyNodes`
	- [ ] `#normalizeNodesArray`
	- [ ] `#classifyNodesFromArray`
	- [ ] `#gatherActivationNames`
	- [ ] `#detectRecurrentOrGated`
	- [ ] `#ensureConnFlagsCapacity` (share via state module)
	- [ ] `#swallowError`
- [ ] Modernize logging (structured console output, optional pretty printing).

## Phase 12 · Façade Slim-Down (`evolutionEngine.ts`)

- [ ] Replace static class with exported functions that delegate to modules (or keep class with 1–3 line wrappers if API stability demands).
- [ ] Ensure each public export has fresh JSDoc referencing the new module functions.
- [ ] Re-export deterministic controls, run loop, and inspection via thin wrappers referencing shared state.
- [ ] Delete redundant private fields and confirm file length < 200 LOC.

## Phase 13 · Tidy & Docs

- [ ] Update tests to target new modules with single-expect cases.
- [ ] Refresh README/examples to reference the façade API only.
- [ ] Document module responsibilities inline (top-of-file comments referencing this checklist).
- [ ] Run lint + build + tests; capture results in PR description.

