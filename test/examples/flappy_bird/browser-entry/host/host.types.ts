import type {
  FlappyStatsKey,
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';

/**
 * Public type contracts for the browser-entry host boundary.
 *
 * These types describe what the host builder returns to the runtime and how HUD
 * value updates are represented once the UI tree exists.
 */

/**
 * Result payload returned after constructing the browser host UI tree.
 *
 * This is the runtime's handle into the rendered browser shell: the main canvas,
 * its 2D context, the stats-cell lookup, and the network-panel draw callback.
 */
export interface CanvasHostResult {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  statsValueByKey: FlappyStatsTableCells;
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
}

/**
 * Partial stats update map keyed by stats-table keys.
 *
 * Using a partial map lets the runtime update only the HUD fields that changed
 * on a given tick.
 */
export type HostStatsPartialValues = Partial<Record<FlappyStatsKey, string>>;
