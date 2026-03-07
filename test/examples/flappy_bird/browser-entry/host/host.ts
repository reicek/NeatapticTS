import type { FlappyStatsTableCells } from '../browser-entry.types';
import type { CanvasHostResult, HostStatsPartialValues } from './host.types';
import {
  createCanvasHostInternal,
  updateStatsTableValuesInternal,
} from '../browser-entry.host.utils';

/**
 * Builds the browser demo host tree and returns rendering handles.
 *
 * @param containerElement - Root host container.
 * @returns Canvas handles, stats cells and network render callback.
 */
export function createCanvasHost(
  containerElement: HTMLElement,
): CanvasHostResult {
  return createCanvasHostInternal(containerElement);
}

/**
 * Applies partial stat updates to the rendered stats table.
 *
 * @param statsValueByKey - Lookup of stat keys to value cells.
 * @param partialValues - Subset of values to write this tick.
 * @returns Nothing.
 */
export function updateStatsTableValues(
  statsValueByKey: FlappyStatsTableCells,
  partialValues: HostStatsPartialValues,
): void {
  updateStatsTableValuesInternal(statsValueByKey, partialValues);
}
