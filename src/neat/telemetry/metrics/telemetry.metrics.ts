/**
 * Metric-building helpers for the telemetry recorder pipeline.
 *
 * This chapter groups the computation-only slices of telemetry that enrich a
 * generation snapshot after the runtime buffer has been prepared. Entropy,
 * diversity, lineage, complexity, objective, RNG, and performance helpers all
 * live together here so `src/neat/telemetry/recorder/telemetry.recorder.ts`
 * can stay orchestration-first while the metrics subtree owns the actual
 * calculations and payload shaping rules.
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
