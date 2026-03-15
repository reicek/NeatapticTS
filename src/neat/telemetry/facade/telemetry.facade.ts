/**
 * Public read-heavy facade helpers for Neat telemetry, objectives, and archive inspection.
 *
 * This module groups the parts of the Neat surface that mainly expose existing
 * state rather than drive evolution. Keeping the public telemetry root beside
 * its chapter folders lets `src/neat.ts` stay orchestration-first while the
 * telemetry split now reads as one discoverable subtree.
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
 * @param host - `Neat` instance storing generation telemetry snapshots.
 * @returns Telemetry entries captured so far, or an empty array when telemetry
 * is not initialized.
 */
export function getTelemetry(host: NeatTelemetryFacadeHost): TelemetryEntry[] {
  return getTelemetryFacadeBuffer(host);
}

/**
 * Export telemetry as JSON Lines so logs can stream into files or post-processors.
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
 * @param host - `Neat` instance whose telemetry buffer should be exported.
 * @param maxEntries - Maximum number of recent entries to include.
 * @returns CSV string containing the requested telemetry window.
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
 * @param host - `Neat` instance whose live species registry should be summarized.
 * @returns Array of current species summaries.
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
 * @param host - `Neat` instance storing species history snapshots.
 * @returns Historical species entries for each recorded generation.
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
 * @param host - `Neat` instance tracking performance timings.
 * @returns Snapshot of the last evaluation and evolution durations.
 */
export function getPerformanceStats(host: NeatTelemetryFacadeHost) {
  return getTelemetryFacadeRuntimePerformanceStats(host);
}

/**
 * Return cached diversity metrics, computing a fallback snapshot when needed.
 *
 * This keeps the public facade resilient: callers can always ask for diversity
 * stats even before a full metrics pass has run.
 *
 * @param host - `Neat` instance exposing cached diversity state.
 * @returns Diversity metrics for the current population.
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
