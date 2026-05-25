/**
 * Metric-building helpers for the telemetry recorder pipeline.
 *
 * This chapter is where raw controller state becomes interpretable telemetry.
 * The recorder owns the question, "build one entry for this generation," but
 * the metrics subtree owns the harder follow-up question: "what evidence should
 * go into that entry so a human can understand how the run is behaving?"
 *
 * Read the metrics family as six teaching-oriented layers:
 * - diversity and entropy helpers estimate how structurally varied the current
 *   population still is
 * - lineage helpers summarize ancestry depth, inbreeding, and ancestor
 *   uniqueness so genealogical pressure is visible
 * - objective helpers explain what the multi-objective controller is tracking,
 *   which objectives changed, and how the Pareto frontier is evolving
 * - complexity helpers turn node and connection growth into explicit telemetry
 * - RNG helpers expose reproducibility state when a run needs deterministic replay
 * - performance helpers attach evaluation and evolution timings so telemetry can
 *   explain computational cost as well as search behavior
 *
 * The key pedagogical boundary is that these helpers compute and attach values,
 * but they do not decide when an entry is recorded or how it is stored. That is
 * why they live below `recorder/` and beside `runtime/`: the recorder orchestrates,
 * runtime persists safely, and metrics explains the generation.
 *
 * A good way to read this chapter is:
 * 1. start with the diversity and lineage helpers to understand population health
 * 2. continue to objective and complexity helpers to understand search pressure
 * 3. finish with RNG and performance helpers to understand reproducibility and cost
 *
 * ```mermaid
 * flowchart LR
 *   Snapshot["Recorder starts one generation snapshot"] --> Diversity["diversity + entropy<br/>How varied is the population?"]
 *   Snapshot --> Lineage["lineage<br/>How related are the current genomes?"]
 *   Snapshot --> Objectives["objectives + Pareto<br/>What tradeoffs are active?"]
 *   Snapshot --> Complexity["complexity<br/>How large are genomes becoming?"]
 *   Snapshot --> RNG["RNG state<br/>Can this run be replayed?"]
 *   Snapshot --> Performance["performance<br/>What did this generation cost?"]
 *   Diversity --> Entry["Telemetry entry"]
 *   Lineage --> Entry
 *   Objectives --> Entry
 *   Complexity --> Entry
 *   RNG --> Entry
 *   Performance --> Entry
 * ```
 *
 * If the recorder chapter explains telemetry as a pipeline, this chapter explains
 * telemetry as evidence. It is the place to read when the question is not
 * "how was the entry recorded?" but "why do these numbers exist, and what do
 * they reveal about the search?"
 */
/** Re-export entropy-cache reader with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { getCachedEntropy } from './telemetry.metrics.entropy';
/** Re-export degree counter helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeDegreeCounts } from './telemetry.metrics.entropy';
/** Re-export degree histogram helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { buildDegreeHistogram } from './telemetry.metrics.entropy';
/** Re-export entropy-from-histogram helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeEntropyFromHistogram } from './telemetry.metrics.entropy';
/** Re-export entropy-cache writer with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { setCachedEntropy } from './telemetry.metrics.entropy';

/** Re-export telemetry core snapshot helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { getTelemetryCoreSnapshot } from './telemetry.metrics.selection';
/** Re-export telemetry key stripping helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { stripUnselectedTelemetryKeys } from './telemetry.metrics.selection';
/** Re-export telemetry core merge helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { mergeTelemetryCoreFields } from './telemetry.metrics.selection';
/** Re-export safe telemetry selection helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { safelyApplyTelemetrySelect } from './telemetry.metrics.selection';

/** Re-export fast-mode defaults helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyFastModeDefaults } from './telemetry.metrics.diversity';
/** Re-export compatibility statistics helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeCompatibilityStats } from './telemetry.metrics.diversity';
/** Re-export entropy statistics helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeEntropyStats } from './telemetry.metrics.diversity';
/** Re-export graphlet entropy helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeGraphletEntropy } from './telemetry.metrics.diversity';
/** Re-export distinct-index sampler with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { pickDistinctIndices } from './telemetry.metrics.diversity';
/** Re-export enabled-edge counter with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { countEnabledEdges } from './telemetry.metrics.diversity';

/** Re-export operator stats snapshot helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeOperatorStatsSnapshot } from './telemetry.metrics.operator';
/** Re-export operator stats reader with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { readOperatorStats } from './telemetry.metrics.operator';

/** Re-export hypervolume proxy helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeHyperVolumeProxy } from './telemetry.metrics.objectives';
/** Re-export Pareto front sizing helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeParetoFrontSizes } from './telemetry.metrics.objectives';
/** Re-export objective-importance applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyObjectiveImportance } from './telemetry.metrics.objectives';
/** Re-export objective-age applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyObjectiveAges } from './telemetry.metrics.objectives';
/** Re-export objective-event applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyObjectiveEvents } from './telemetry.metrics.objectives';
/** Re-export species allocation applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applySpeciesAllocation } from './telemetry.metrics.objectives';
/** Re-export objective snapshot applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyObjectivesSnapshot } from './telemetry.metrics.objectives';
/** Re-export hypervolume telemetry applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyHypervolumeTelemetry } from './telemetry.metrics.objectives';

/** Re-export RNG state applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyRngState } from './telemetry.metrics.rng';

/** Re-export lineage statistics helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeLineageStats } from './telemetry.metrics.lineage';
/** Re-export multi-objective lineage applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyLineageStatsMultiObjective } from './telemetry.metrics.lineage';
/** Re-export mono-objective lineage applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyLineageStatsMonoObjective } from './telemetry.metrics.lineage';
/** Re-export lineage eligibility guard with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { isLineageEligible } from './telemetry.metrics.lineage';
/** Re-export lineage depth collector with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { collectDepths } from './telemetry.metrics.lineage';
/** Re-export lineage mean-depth helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeMeanDepth } from './telemetry.metrics.lineage';
/** Re-export sampled ancestor-uniqueness helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeAncestorUniquenessSampled } from './telemetry.metrics.lineage';
/** Re-export distinct-pair sampler with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { pickDistinctPairIndices } from './telemetry.metrics.lineage';
/** Re-export pairwise Jaccard distance helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computePairJaccardDistance } from './telemetry.metrics.lineage';
/** Re-export lineage context builder with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { buildLineageContext } from './telemetry.metrics.lineage';
/** Re-export ancestor intersection counter with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { countAncestorIntersection } from './telemetry.metrics.lineage';
/** Re-export lineage entry builder with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { buildLineageEntry } from './telemetry.metrics.lineage';

/** Re-export population count collector with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { collectPopulationCounts } from './telemetry.metrics.complexity';
/** Re-export mean-count helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeMeanCounts } from './telemetry.metrics.complexity';
/** Re-export max-count helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeMaxCounts } from './telemetry.metrics.complexity';
/** Re-export enabled-ratio helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeEnabledRatios } from './telemetry.metrics.complexity';
/** Re-export mean enabled-ratio helper with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeMeanEnabledRatio } from './telemetry.metrics.complexity';
/** Re-export growth-value updater with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { computeAndStoreGrowthValues } from './telemetry.metrics.complexity';
/** Re-export complexity entry builder with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { buildComplexityEntry } from './telemetry.metrics.complexity';
/** Re-export multi-objective complexity applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyComplexityStatsMultiObjective } from './telemetry.metrics.complexity';
/** Re-export mono-objective complexity applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyComplexityStatsMonoObjective } from './telemetry.metrics.complexity';

/** Re-export performance telemetry applier with recorder-facing semantics and durable documentation intent for generated guides, runtime diagnostics, and evidence interpretation during telemetry entry assembly. */
export { applyPerformanceStats } from './telemetry.metrics.performance';
