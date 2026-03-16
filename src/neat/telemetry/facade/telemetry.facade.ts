/**
 * Public read-heavy facade helpers for Neat telemetry, objectives, and archive inspection.
 *
 * This chapter is the user-facing answer to a practical question: once a NEAT
 * run has produced telemetry, where should a caller look first?
 * The write-side telemetry chapters explain how evidence is recorded; this
 * facade explains how that evidence comes back out as compact snapshots,
 * species summaries, objective views, diversity reads, and export-friendly logs.
 *
 * A useful way to read the module is by inspection workflow:
 * - buffer views answer "what happened recently?" via `getTelemetry()` and telemetry exports
 * - objective and Pareto helpers answer "what tradeoffs are active right now?"
 * - species and lineage helpers answer "which families are growing, stalling, or inheriting together?"
 * - diversity and performance helpers answer "is search still broad, and how expensive was the last step?"
 * - clearing helpers reset evidence buffers when you want a fresh observation window
 *
 * ```mermaid
 * flowchart LR
 *   Run["Neat run"] --> Buffer["telemetry buffer<br/>recent generation entries"]
 *   Run --> Species["species history<br/>and live species registry"]
 *   Run --> Objectives["objective registry<br/>and Pareto archive"]
 *   Run --> Runtime["cached diversity<br/>and timing snapshots"]
 *   Buffer --> Inspect["inspect recent trend<br/>getTelemetry()"]
 *   Species --> Compare["compare families<br/>getSpeciesStats() / getSpeciesHistory()"]
 *   Objectives --> Tradeoffs["inspect tradeoffs<br/>getObjectives() / getParetoFronts()"]
 *   Runtime --> Health["check search health<br/>getDiversityStats() / getPerformanceStats()"]
 *   Buffer --> Export["share or archive<br/>CSV / JSONL exports"]
 * ```
 *
 * Read this chapter after `recorder/` when you care less about how a telemetry
 * entry is built and more about how an experimenter, dashboard, notebook, or
 * test can inspect the recorded state without reaching into controller internals.
 */
import type Network from '../../../architecture/network';
import type { DiversityStats } from '../../diversity/diversity';
import type {
  SpeciesHistoryEntry,
  TelemetryEntry,
} from '../../shared/neat.shared.types';
import { SPECIES_HISTORY_JSONL_MAX_DEFAULT } from '../../species/history/species.history';
import {
  clearParetoArchive as clearTelemetryFacadeArchive,
  exportParetoFrontJSONL as exportTelemetryFacadeArchiveJsonl,
  getMultiObjectiveMetrics as getTelemetryFacadeArchiveMetrics,
  getParetoArchive as getTelemetryFacadeArchiveEntries,
  getParetoFronts as getTelemetryFacadeArchiveFronts,
  type TelemetryFacadeArchiveHost,
} from './archive/telemetry.facade.archive';
import {
  clearTelemetry as clearTelemetryFacadeBuffer,
  exportTelemetryCSV as exportTelemetryFacadeBufferCsv,
  exportTelemetryJSONL as exportTelemetryFacadeBufferJsonl,
  getTelemetry as getTelemetryFacadeBuffer,
  type TelemetryFacadeBufferHost,
} from './buffer/telemetry.facade.buffer';
import {
  getLineageSnapshot as getTelemetryFacadeLineageSnapshot,
  LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
  type TelemetryFacadeLineageHost,
} from './lineage/telemetry.facade.lineage';
import {
  getNoveltyArchiveSize as getTelemetryFacadeNoveltyArchiveSize,
  resetNoveltyArchive as resetTelemetryFacadeNoveltyArchive,
  type TelemetryFacadeNoveltyHost,
} from './novelty/telemetry.facade.novelty';
import {
  clearTelemetryObjectives as clearTelemetryFacadeObjectives,
  getObjectiveEvents as getTelemetryFacadeObjectiveEvents,
  getObjectiveKeys as getTelemetryFacadeObjectiveKeys,
  getObjectives as getTelemetryFacadeObjectives,
  registerTelemetryObjective as registerTelemetryFacadeObjective,
  type TelemetryFacadeObjectivesHost,
} from './objectives/telemetry.facade.objectives';
import {
  getOperatorStats as getTelemetryFacadeOperatorStats,
  type TelemetryFacadeOperatorStatsHost,
} from './operator-stats/telemetry.facade.operator-stats';
import {
  getDiversityStats as getTelemetryFacadeRuntimeDiversityStats,
  getPerformanceStats as getTelemetryFacadeRuntimePerformanceStats,
  type TelemetryFacadeRuntimeHost,
} from './runtime/telemetry.facade.runtime';
import {
  exportSpeciesHistoryCSV as exportTelemetryFacadeSpeciesHistoryCsv,
  exportSpeciesHistoryJSONL as exportTelemetryFacadeSpeciesHistoryJsonl,
  getSpeciesHistory as getTelemetryFacadeSpeciesHistory,
  getSpeciesStats as getTelemetryFacadeSpeciesStats,
} from './species/telemetry.facade.species';

/**
 * Narrow `Neat` surface needed by the public telemetry, objective, and archive
 * facade methods.
 *
 * This interface exists so the public `src/neat.ts` facade can delegate
 * read-heavy diagnostics and export helpers into one focused module without
 * exposing the entire controller implementation. The host shape keeps the
 * contract small: population snapshots, telemetry caches, objective accessors,
 * and archive buffers.
 */
export interface NeatTelemetryFacadeHost
  extends
    TelemetryFacadeArchiveHost,
    TelemetryFacadeBufferHost,
    TelemetryFacadeLineageHost,
    TelemetryFacadeNoveltyHost,
    TelemetryFacadeOperatorStatsHost,
    TelemetryFacadeRuntimeHost,
    TelemetryFacadeObjectivesHost {
  population: Network[];
  _speciesHistory: SpeciesHistoryEntry[];
  _telemetry?: TelemetryEntry[];
}

/**
 * Return just the registered objective keys in stable order.
 *
 * This is the shortest inspection surface for tests and quick diagnostics that
 * only need to confirm which objectives are active, not the full descriptor
 * payload.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Ordered list of active objective keys.
 */
export function getObjectiveKeys(host: NeatTelemetryFacadeHost): string[] {
  return getTelemetryFacadeObjectiveKeys(host);
}

/**
 * Return the in-memory telemetry buffer.
 *
 * This is the fastest way to inspect the recent rhythm of a run: score trends,
 * diversity changes, objective snapshots, and timing evidence exactly as they
 * were recorded generation by generation.
 *
 * @param host - `Neat` instance storing generation telemetry snapshots.
 * @returns Telemetry entries captured so far, or an empty array when telemetry
 * is not initialized.
 * @example
 * const telemetryWindow = getTelemetry(neat).slice(-5);
 * console.table(telemetryWindow.map((entry) => ({
 *   generation: entry.generation,
 *   bestScore: entry.bestScore,
 *   species: entry.species,
 * })));
 */
export function getTelemetry(host: NeatTelemetryFacadeHost): TelemetryEntry[] {
  return getTelemetryFacadeBuffer(host);
}

/**
 * Export telemetry as JSON Lines so logs can stream into files or post-processors.
 *
 * JSONL is the most automation-friendly telemetry surface: one entry per line,
 * easy to append to a file, easy to pipe into scripts, and stable enough for
 * notebook or CLI post-processing.
 *
 * @param host - `Neat` instance whose telemetry buffer should be serialized.
 * @returns JSONL payload with one telemetry object per line.
 */
export function exportTelemetryJSONL(host: NeatTelemetryFacadeHost): string {
  return exportTelemetryFacadeBufferJsonl(host);
}

/**
 * Export recent telemetry entries as CSV for quick spreadsheet inspection.
 *
 * Prefer this when the reader is a person scanning a table instead of a script
 * parsing nested JSON. The helper intentionally focuses on a recent window so a
 * long run can still produce a compact worksheet.
 *
 * @param host - `Neat` instance whose telemetry buffer should be exported.
 * @param maxEntries - Maximum number of recent entries to include.
 * @returns CSV string containing the requested telemetry window.
 * @example
 * const csv = exportTelemetryCSV(neat, 100);
 * console.log(csv.split('\n').slice(0, 3).join('\n'));
 */
export function exportTelemetryCSV(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = 500,
): string {
  return exportTelemetryFacadeBufferCsv(host, maxEntries);
}

/**
 * Clear cached telemetry entries.
 *
 * @param host - `Neat` instance whose telemetry buffer should be reset.
 * @returns Nothing. The helper mutates the host buffer in place.
 */
export function clearTelemetry(host: NeatTelemetryFacadeHost): void {
  clearTelemetryFacadeBuffer(host);
}

/**
 * Return a compact view of active objective descriptors.
 *
 * The full objective descriptor includes accessors and internal metadata. This
 * read model trims that down to the pieces most useful in UI surfaces and
 * debugging output: the key and whether the objective is minimized or
 * maximized.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Compact objective summaries in evaluation order.
 * @example
 * const objectives = getObjectives(neat);
 * console.table(objectives);
 */
export function getObjectives(
  host: NeatTelemetryFacadeHost,
): { key: string; direction: 'max' | 'min' }[] {
  return getTelemetryFacadeObjectives(host);
}

/**
 * Register or replace a custom objective.
 *
 * @param host - `Neat` instance whose multi-objective registry should change.
 * @param key - Unique objective key.
 * @param direction - Whether lower or higher values are considered better.
 * @param accessor - Function that reads the objective value from a genome.
 * @returns Nothing. The objective registry on `host` is updated in place.
 */
export function registerTelemetryObjective(
  host: NeatTelemetryFacadeHost,
  key: string,
  direction: 'min' | 'max',
  accessor: Parameters<typeof registerTelemetryFacadeObjective>[3],
): void {
  registerTelemetryFacadeObjective(host, key, direction, accessor);
}

/**
 * Remove all registered custom objectives so only the default objective path remains.
 *
 * @param host - `Neat` instance whose objective registry should be cleared.
 * @returns Nothing. The helper mutates the objective registry in place.
 */
export function clearTelemetryObjectives(host: NeatTelemetryFacadeHost): void {
  clearTelemetryFacadeObjectives(host);
}

/**
 * Snapshot recent objective add/remove events for telemetry consumers.
 *
 * @param host - `Neat` instance storing objective lifecycle events.
 * @returns Shallow copy of the recorded objective events.
 */
export function getObjectiveEvents(host: NeatTelemetryFacadeHost): {
  gen: number;
  type: 'add' | 'remove';
  key: string;
}[] {
  return getTelemetryFacadeObjectiveEvents(host);
}

/**
 * Return a compact lineage sample for the first genomes in the current population.
 *
 * This is meant for inspection and teaching, not for full genealogy export. It
 * keeps payloads small by clipping the returned slice while still showing which
 * genomes share parents.
 *
 * @param host - `Neat` instance whose population lineage should be sampled.
 * @param limit - Maximum number of genomes to include in the snapshot.
 * @returns Array of `{ id, parents }` lineage entries.
 */
export function getLineageSnapshot(
  host: NeatTelemetryFacadeHost,
  limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
): { id: number; parents: number[] }[] {
  return getTelemetryFacadeLineageSnapshot(host, limit);
}

/**
 * Export species history as CSV rows.
 *
 * @param host - `Neat` instance whose species history should be exported.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns CSV payload for offline species analysis.
 */
export function exportSpeciesHistoryCSV(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = 200,
): string {
  return exportTelemetryFacadeSpeciesHistoryCsv(host, maxEntries);
}

/**
 * Export species history as JSON Lines.
 *
 * @param host - `Neat` instance whose species history should be serialized.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns JSONL payload describing recent species history snapshots.
 */
export function exportSpeciesHistoryJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
): string {
  return exportTelemetryFacadeSpeciesHistoryJsonl(host, maxEntries);
}

/**
 * Return a concise summary for each current species.
 *
 * This is the fastest species-level diagnostic when you want to see whether the
 * population is still split across several improving families or collapsing
 * toward one dominant cluster.
 *
 * @param host - `Neat` instance whose live species registry should be summarized.
 * @returns Array of current species summaries.
 * @example
 * const speciesSummary = getSpeciesStats(neat);
 * console.table(speciesSummary);
 */
export function getSpeciesStats(host: NeatTelemetryFacadeHost): {
  id: number;
  size: number;
  bestScore: number;
  lastImproved: number;
}[] {
  return getTelemetryFacadeSpeciesStats(host);
}

/**
 * Return recorded species history, lazily backfilling extended metrics when enabled.
 *
 * Use this when you need the story across generations rather than the current
 * snapshot. It is the better surface for plots, regressions, and post-run
 * analysis of stagnation or speciation churn.
 *
 * @param host - `Neat` instance storing species history snapshots.
 * @returns Historical species entries for each recorded generation.
 * @example
 * const historyWindow = getSpeciesHistory(neat).slice(-10);
 * console.log(historyWindow.length);
 */
export function getSpeciesHistory(
  host: NeatTelemetryFacadeHost,
): SpeciesHistoryEntry[] {
  return getTelemetryFacadeSpeciesHistory(host);
}

/**
 * Return the current novelty archive size.
 *
 * @param host - `Neat` instance tracking novelty behavior descriptors.
 * @returns Number of archived novelty descriptors.
 */
export function getNoveltyArchiveSize(host: NeatTelemetryFacadeHost): number {
  return getTelemetryFacadeNoveltyArchiveSize(host);
}

/**
 * Build compact multi-objective metrics for the current population snapshot.
 *
 * This helper is meant for inspection surfaces that need the shape of the
 * current Pareto landscape without pulling every genome field into view.
 *
 * @param host - `Neat` instance whose population should be summarized.
 * @returns Rank, crowding, score, and size metrics per genome.
 */
export function getMultiObjectiveMetrics(host: NeatTelemetryFacadeHost): {
  rank: number;
  crowding: number;
  score: number;
  nodes: number;
  connections: number;
}[] {
  return getTelemetryFacadeArchiveMetrics(host);
}

/**
 * Return aggregated mutation/operator statistics.
 *
 * @param host - `Neat` instance recording operator attempts and successes.
 * @returns Operator summaries suitable for dashboards and debugging.
 */
export function getOperatorStats(host: NeatTelemetryFacadeHost): {
  name: string;
  success: number;
  attempts: number;
}[] {
  return getTelemetryFacadeOperatorStats(host);
}

/**
 * Reconstruct Pareto fronts from current rank annotations.
 *
 * Use this when the question is structural rather than historical: which fronts
 * exist right now, and how many genomes are sitting on each layer of the
 * current tradeoff surface?
 *
 * @param host - `Neat` instance whose population should be partitioned.
 * @param maxFronts - Maximum number of fronts to reconstruct.
 * @returns Pareto fronts ordered from best to worst.
 */
export function getParetoFronts(
  host: NeatTelemetryFacadeHost,
  maxFronts?: number,
): Network[][] {
  return getTelemetryFacadeArchiveFronts(host, maxFronts);
}

/**
 * Return the most recent Pareto archive entries.
 *
 * Unlike `getParetoFronts()`, which reconstructs the current population view,
 * this helper reads the historical archive that was captured while the run was
 * evolving. It is better suited for replaying how the frontier changed over
 * time.
 *
 * @param host - `Neat` instance storing archived Pareto metadata.
 * @param maxEntries - Maximum number of archive entries to return.
 * @returns Slice of the recent Pareto archive.
 */
export function getParetoArchive(
  host: NeatTelemetryFacadeHost,
  maxEntries?: number,
): import('../../shared/neat.shared.types').ParetoArchiveEntry[] {
  return getTelemetryFacadeArchiveEntries(host, maxEntries);
}

/**
 * Export recent Pareto archive entries as JSON Lines.
 *
 * @param host - `Neat` instance storing Pareto objective snapshots.
 * @param maxEntries - Maximum number of entries to serialize.
 * @returns JSONL payload for recent Pareto archive entries.
 */
export function exportParetoFrontJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries?: number,
): string {
  return exportTelemetryFacadeArchiveJsonl(host, maxEntries);
}

/**
 * Return coarse timing metrics for the last evaluation and evolution passes.
 *
 * These timings are intentionally simple. They answer "which phase was expensive
 * last time?" without pretending to replace a profiler.
 *
 * @param host - `Neat` instance tracking performance timings.
 * @returns Snapshot of the last evaluation and evolution durations.
 * @example
 * const timing = getPerformanceStats(neat);
 * console.log(timing.lastEvalMs, timing.lastEvolveMs);
 */
export function getPerformanceStats(host: NeatTelemetryFacadeHost) {
  return getTelemetryFacadeRuntimePerformanceStats(host);
}

/**
 * Return cached diversity metrics, computing a fallback snapshot when needed.
 *
 * This keeps the public facade resilient: callers can always ask for diversity
 * stats even before a full metrics pass has run.
 * The resulting snapshot is especially useful when you need to judge whether a
 * run is still exploring many structural alternatives or converging too hard.
 *
 * @param host - `Neat` instance exposing cached diversity state.
 * @returns Diversity metrics for the current population.
 * @example
 * const diversity = getDiversityStats(neat);
 * console.log(diversity.structuralEntropy, diversity.uniqueStructures);
 */
export function getDiversityStats(
  host: NeatTelemetryFacadeHost,
): DiversityStats {
  return getTelemetryFacadeRuntimeDiversityStats(host);
}

/**
 * Clear the novelty archive.
 *
 * @param host - `Neat` instance whose novelty archive should be reset.
 * @returns Nothing. The archive is mutated in place.
 */
export function resetNoveltyArchive(host: NeatTelemetryFacadeHost): void {
  resetTelemetryFacadeNoveltyArchive(host);
}

/**
 * Clear the Pareto archive metadata stored on the host.
 *
 * @param host - `Neat` instance whose Pareto archive should be emptied.
 * @returns Nothing. The archive buffer is reset in place.
 */
export function clearParetoArchive(host: NeatTelemetryFacadeHost): void {
  clearTelemetryFacadeArchive(host);
}
