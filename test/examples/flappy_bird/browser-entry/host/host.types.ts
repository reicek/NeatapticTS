import type {
  FlappyStatsKey,
  FlappyStatsTableCells,
  NetworkVisualizationHandle,
} from '../browser-entry.types';

/**
 * Result payload returned after constructing the browser host UI tree.
 */
export interface CanvasHostResult {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  statsValueByKey: FlappyStatsTableCells;
  renderNetworkArchitecture: NetworkVisualizationHandle['renderNetworkArchitecture'];
}

/**
 * Partial stats update map keyed by stats-table keys.
 */
export type HostStatsPartialValues = Partial<Record<FlappyStatsKey, string>>;
