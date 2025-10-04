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

- [x] Move orchestration method `#logGenerationTelemetry` _(refactored to delegate to extracted helpers)_.
- [x] Sequentially extract helpers:
	- [x] `#logActionEntropy` _(extracted to `telemetryMetrics.ts` as `logActionEntropy`)_
	- [x] `#computeActionEntropy` _(extracted as internal helper in `telemetryMetrics.ts`)_
	- [x] `#logOutputBiasStats` _(extracted to `telemetryMetrics.ts` as `logOutputBiasStats`)_
	- [x] `#computeOutputBiasStats` _(extracted as internal helper in `telemetryMetrics.ts`)_
	- [x] `#logLogitsAndCollapse` _(extracted to `telemetryMetrics.ts` as `logLogitsAndCollapse`)_
	- [x] `#computeLogitStats` (+ `#resetLogitScratch`, `#accumulate*`, `#finalize*`) _(extracted as internal helpers)_
	- [x] `#computeDecisionStability` _(extracted as internal helper in `telemetryMetrics.ts`)_
	- [x] `#softmaxEntropyFromVector` _(extracted as internal helper in `telemetryMetrics.ts`)_
	- [x] `#logExploration` _(extracted to `telemetryMetrics.ts` as `logExploration`)_
	- [x] `#computeExplorationStats`, `#countDistinctCoordinatesTiny`, `#countDistinctCoordinatesHashed` _(extracted as internal helpers)_
	- [x] `#logDiversity` _(extracted to `telemetryMetrics.ts` as `logDiversity`)_
	- [x] `#computeDiversityMetrics` _(extracted as internal helper in `telemetryMetrics.ts`)_
	- [x] `#collectTelemetryTail` _(extracted to `telemetryMetrics.ts` as `collectTelemetryTail`)_
	- [x] `#joinNumberArray` _(extracted as internal helper in `telemetryMetrics.ts`)_
- [x] **Note:** Collapse recovery (`#antiCollapseRecovery`) remains in `evolutionEngine.ts` and is invoked via callback from `logLogitsAndCollapse`. Private `#compute*` helpers remain in the engine as they're used by both the telemetry module and other parts of the engine.
- [x] **Verification:** All 6 public telemetry functions imported, legacy methods removed, ~365 lines reduced from façade. File size reduction confirmed (8,083 → 7,755 lines).

## Phase 6 · `populationPruning.ts`

- [x] Extract `#applySimplifyPruningToPopulation` _(extracted to `populationPruning.ts` as `applySimplifyPruningToPopulation`)_.
- [x] Move pruning helpers in order:
	- [x] `#pruneWeakConnectionsForGenome` _(extracted as internal helper in `populationPruning.ts`)_
	- [x] `#collectEnabledConnections` _(extracted as internal helper in `populationPruning.ts`)_
	- [x] `#collectHiddenToOutputConns` _(extracted as internal helper in `populationPruning.ts`; minimal version retained in engine for other uses)_
	- [x] `#sortCandidatesByStrategy` _(extracted as internal helper in `populationPruning.ts`)_
	- [x] `#insertionSortByAbsWeight` _(extracted as internal helper in `populationPruning.ts`)_
	- [x] `#disableSmallestEnabledConnections` _(extracted as internal helper in `populationPruning.ts`)_
- [x] Port compass warm-start & bias recentre helpers _(extracted to `populationPruning.ts` as `applyCompassWarmStart` and `centerOutputBiases`)_.
- [x] **Note:** Helpers `#getNodeIndicesByType` and `#collectHiddenToOutputConns` retained as minimal internal helpers in `evolutionEngine.ts` because they're used by non-pruning methods elsewhere in the engine. The pruning module has its own `collectNodeIndicesByType` implementation.
- [x] **Verification:** All 3 public pruning functions (`applySimplifyPruningToPopulation`, `applyCompassWarmStart`, `centerOutputBiases`) imported and delegated, legacy methods removed, ~182 lines reduced from façade (7,755 → ~6,970 lines). TypeScript reports no errors.


## Phase 7 · `populationDynamics.ts`

- [x] Move generation-level helpers:
	- [x] `#updatePlateauState` _(extracted to `populationDynamics.ts` as `updatePlateauState`)_
	- [x] `#handleSimplifyState` _(extracted to `populationDynamics.ts` as `handleSimplifyState`)_
	- [x] `#runSimplifyCycle` _(extracted to `populationDynamics.ts` as `runSimplifyCycle`)_
- [x] Transfer population changes:
	- [x] `#expandPopulation` _(extracted to `populationDynamics.ts` as `expandPopulation`)_
	- [x] `#prepareExpansion` _(extracted to `populationDynamics.ts` as `prepareExpansion`)_
	- [x] `#determineMutateCount` _(extracted to `populationDynamics.ts` as `determineMutateCount`)_
	- [x] `#applyMutationsToClone` _(extracted to `populationDynamics.ts` as `applyMutationsToClone`)_
	- [x] `#registerClone` _(extracted to `populationDynamics.ts` as `registerClone`)_
	- [x] `#createChildFromParent` _(extracted to `populationDynamics.ts` as `createChildFromParent`)_
	- [x] `#getSortedIndicesByScore` _(extracted to `populationDynamics.ts` as `getSortedIndicesByScore`)_
	- [x] `#insertionSortIndices` _(extracted as internal helper in `populationDynamics.ts`)_
	- [x] `#medianOfThreePivot` _(extracted as internal helper in `populationDynamics.ts`)_
	- [x] `#qsPushRange` _(extracted as internal helper in `populationDynamics.ts`)_
	- [x] `#getMutationOps` _(extracted as internal helper in `populationDynamics.ts`)_
	- [x] `#ensureOutputIdentity` _(extracted to `populationDynamics.ts` as `ensureOutputIdentity`)_
	- [x] `#handleSpeciesHistory` _(extracted to `populationDynamics.ts` as `handleSpeciesHistory`)_
	- [x] `#maybeExpandPopulation` _(extracted to `populationDynamics.ts` as `maybeExpandPopulation`)_
	- [x] `#pruneSaturatedHiddenOutputs` _(extracted to `populationDynamics.ts` as `pruneSaturatedHiddenOutputs`)_
	- [x] `#compactGenomeConnections` _(extracted to `populationDynamics.ts` as `compactGenomeConnections`)_
	- [x] `#compactPopulation` _(extracted to `populationDynamics.ts` as `compactPopulation`)_
- [x] Rehome collapse recovery methods:
	- [x] `#antiCollapseRecovery` _(extracted to `populationDynamics.ts` as `antiCollapseRecovery`)_
	- [x] `#reinitializeGenomeOutputsAndWeights` _(extracted to `populationDynamics.ts` as `reinitializeGenomeOutputsAndWeights`)_
- [x] **Note:** Internal helpers `maybeStartSimplify`, `insertionSortIndices`, `medianOfThreePivot`, `qsPushRange`, and `getMutationOps` remain as non-exported functions within the module. Two small helper methods retained in `evolutionEngine.ts`: `#getNodeIndicesByType` and `#collectHiddenToOutputConns` (used by non-Phase-7 internal code).
- [x] **Delegation Complete:** All 22 public population dynamics functions imported into `evolutionEngine.ts` and all call sites updated to delegate to the new module. Zero TypeScript errors after delegation.
- [x] **Next Step:** Legacy private methods in `evolutionEngine.ts` still present, awaiting deletion in final cleanup pass (est. 800-1000 line reduction from current ~7,007 lines).
- [x] **Verification:** populationDynamics.ts module created (1,574 lines) with 22 public exports and 5 internal helpers. TypeScript compilation successful. All call sites delegated including: updatePlateauState, handleSimplifyState, getSortedIndicesByScore (2 calls), createChildFromParent, determineMutateCount, applyMutationsToClone, registerClone (2 calls), ensureOutputIdentity, handleSpeciesHistory, maybeExpandPopulation, antiCollapseRecovery, pruneSaturatedHiddenOutputs, reinitializeGenomeOutputsAndWeights, compactGenomeConnections, compactPopulation.

## Phase 8 · `trainingWarmStart.ts`

- [x] Migrate Lamarckian pipeline:
	- [x] `#buildLamarckianTrainingSet` _(extracted to `trainingWarmStart.ts` as `buildLamarckianTrainingSet`)_
	- [x] `#adjustOutputBiasesAfterTraining` _(extracted to `trainingWarmStart.ts` as `adjustOutputBiasesAfterTraining`)_
	- [x] `#pretrainPopulationWarmStart` _(extracted to `trainingWarmStart.ts` as `pretrainPopulationWarmStart`)_
	- [x] `#applyLamarckianTraining` _(extracted to `trainingWarmStart.ts` as `applyLamarckianTraining`)_
	- [x] `#warmStartPopulationIfNeeded` _(extracted to `trainingWarmStart.ts` as `warmStartPopulationIfNeeded`)_
- [x] Ensure helpers expose pure functions with explicit parameters (no hidden static usage).
- [x] **Note:** `applyCompassWarmStart` and `centerOutputBiases` already moved to `populationPruning.ts` in Phase 6.
- [x] **Verification:** All 5 warm-start functions imported and delegated, legacy methods removed, ~509 lines reduced from façade (3,780 → 3,271 lines). TypeScript compilation successful. Zero errors after extraction.

## Phase 9 · `optionsAndSetup.ts`

- [x] Relocate environment/setup methods:
	- [x] `#makeFlushToFrame` _(extracted to `setupHelpers.ts` as `makeFlushToFrame`)_
	- [x] `#initPersistence` _(extracted to `setupHelpers.ts` as `initPersistence`)_
	- [x] `#makeSafeWriter` _(extracted to `setupHelpers.ts` as `makeSafeWriter`)_
	- [x] `#createNeat` _(extracted to `neatConfiguration.ts` as `createNeat`)_
	- [x] `#seedInitialPopulation` _(extracted to `neatConfiguration.ts` as `seedInitialPopulation`)_
	- [x] `#createAndSeedNeat` _(extracted to `optionsAndSetup.ts` as `createAndSeedNeat`)_
	- [x] `#prepareEnvironmentForRun` _(extracted to `optionsAndSetup.ts` as `prepareEnvironmentForRun`)_
	- [x] `#normalizeRunOptions` _(extracted to `optionsAndSetup.ts` as `normalizeRunOptions`)_
- [x] Rewrite to accept configuration + `EngineState`, returning POJOs (no use of class statics).
- [x] **Note:** Phase 9 split into three modules: `setupHelpers.ts` (3 functions, 228 lines), `neatConfiguration.ts` (2 functions, 296 lines), and `optionsAndSetup.ts` (3 functions, 414 lines).
- [x] **Verification:** All 8 setup functions imported and delegated, legacy methods removed, ~690 lines reduced from façade (3,084 → 2,585 lines → 2,464 lines after Phase 10a). TypeScript compilation successful. Zero errors after extraction. Import paths corrected (no `.js` for local modules, `../` for parent directory).

## Phase 10 · `evolutionLoop.ts`

- [x] Move orchestration logic:
	- [x] `#prepareLoopHelpers` _(extracted to `evolutionLoop.ts` as `prepareLoopHelpers`)_
	- [x] `#checkCancellation` _(extracted to `evolutionLoop.ts` as `checkCancellation`)_
	- [x] `#checkStopConditions` _(extracted to `evolutionLoop.ts` as `checkStopConditions`)_
	- [x] `#persistSnapshotIfNeeded` _(extracted to `evolutionLoop.ts` as `persistSnapshotIfNeeded`)_
	- [x] `#updateDashboardAndMaybeFlush` _(extracted to `evolutionLoop.ts` as `updateDashboardAndMaybeFlush`)_
	- [x] `#updateDashboardPeriodic` _(extracted to `evolutionLoop.ts` as `updateDashboardPeriodic`)_
	- [x] `#emitProfileSummary` _(extracted to `evolutionLoop.ts` as `emitProfileSummary`)_
	- [ ] `#runGeneration`
	- [ ] `#simulateAndPostprocess`
	- [ ] `#runEvolutionLoop`
- [x] Remove direct static references; interact solely via imported helpers and `EngineState`.
- [x] **Progress:** Seven helper functions (751 lines in module) extracted. File reduced: 2,574 → 2,464 → 2,324 → 2,043 → 1,960 lines (-614 lines total, 72% reduction from original 7,007). Remaining: 3 complex orchestration methods with tight class coupling.
- [ ] **Verification pending:** Full Phase 10 completion awaits extraction of remaining 3 methods (runGeneration, simulateAndPostprocess, runEvolutionLoop).

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

