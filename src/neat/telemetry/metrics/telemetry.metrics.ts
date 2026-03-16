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
export * from './telemetry.metrics.entropy';
export * from './telemetry.metrics.selection';
export * from './telemetry.metrics.diversity';
export * from './telemetry.metrics.operator';
export * from './telemetry.metrics.objectives';
export * from './telemetry.metrics.rng';
export * from './telemetry.metrics.lineage';
export * from './telemetry.metrics.complexity';
export * from './telemetry.metrics.performance';
