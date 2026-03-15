import type {
  SpeciesHistoryEntry,
  SpeciesHistoryStat,
  SpeciesLike,
} from '../../../shared/neat.shared.types';
import { exportSpeciesHistoryCSV as exportSpeciesHistoryCsvImpl } from '../../exports/telemetry.exports';
import {
  getSpeciesHistory as getSpeciesHistoryImpl,
  getSpeciesStats as getSpeciesStatsImpl,
} from '../../../species/species';
import {
  exportSpeciesHistoryJsonl,
  SPECIES_HISTORY_JSONL_MAX_DEFAULT,
} from '../../../species/history/species.history';

/**
 * Narrow telemetry-facade host surface required by the species chapter.
 *
 * These helpers sit between the public telemetry facade and the dedicated
 * species modules, so they only depend on the pieces needed to forward species
 * reads and exports safely.
 */
export interface TelemetryFacadeSpeciesHost {
  options: Record<string, unknown>;
  _speciesHistory?: SpeciesHistoryEntry[];
  _species?: SpeciesLike[] | SpeciesHistoryStat[];
  _fallbackInnov?: ((connection: unknown) => number) | undefined;
  generation?: number;
}

/**
 * Export species history as CSV rows.
 *
 * This chapter keeps the species-oriented forwarding logic out of the broader
 * telemetry facade so the root surface can stay organized by concept instead of
 * accumulating unrelated read helpers in one file.
 *
 * @param host - `Neat` instance whose species history should be exported.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns CSV payload for offline species analysis.
 */
export function exportSpeciesHistoryCSV(
  host: TelemetryFacadeSpeciesHost,
  maxEntries = 200,
): string {
  return exportSpeciesHistoryCsvImpl.call(host as never, maxEntries);
}

/**
 * Export species history as JSON Lines.
 *
 * @param host - `Neat` instance whose species history should be serialized.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns JSONL payload describing recent species history snapshots.
 */
export function exportSpeciesHistoryJSONL(
  host: TelemetryFacadeSpeciesHost,
  maxEntries: number = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
): string {
  return exportSpeciesHistoryJsonl(host._speciesHistory ?? [], maxEntries);
}

/**
 * Return a concise summary for each current species.
 *
 * @param host - `Neat` instance whose live species registry should be summarized.
 * @returns Array of current species summaries.
 */
export function getSpeciesStats(host: TelemetryFacadeSpeciesHost): {
  id: number;
  size: number;
  bestScore: number;
  lastImproved: number;
}[] {
  return getSpeciesStatsImpl.call(host as never);
}

/**
 * Return recorded species history, lazily backfilling extended metrics when enabled.
 *
 * @param host - `Neat` instance storing species history snapshots.
 * @returns Historical species entries for each recorded generation.
 */
export function getSpeciesHistory(
  host: TelemetryFacadeSpeciesHost,
): SpeciesHistoryEntry[] {
  return getSpeciesHistoryImpl.call(host as never) as SpeciesHistoryEntry[];
}
