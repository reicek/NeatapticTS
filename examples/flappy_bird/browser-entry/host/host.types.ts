import type { ExampleArchitectureProfileId } from '../../../architectureProfiles';
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
  architectureSelectorController: HostArchitectureSelectorController;
}

/**
 * Partial stats update map keyed by stats-table keys.
 *
 * Using a partial map lets the runtime update only the HUD fields that changed
 * on a given tick.
 */
export type HostStatsPartialValues = Partial<Record<FlappyStatsKey, string>>;

/** Render-ready button state for one Flappy architecture profile selector item. */
export interface HostArchitectureSelectorItem {
  caption?: string;
  id: ExampleArchitectureProfileId;
  label: string;
  selected: boolean;
  tooltipBodyLines: string[];
  tooltipHeading: string;
}

/** Browser host callback bundle used by the architecture selector control group. */
export interface CanvasHostOptions {
  architectureSelectorItems: HostArchitectureSelectorItem[];
  onSelectArchitectureProfile?: (
    profileId: ExampleArchitectureProfileId,
  ) => void;
  onResetScores?: () => void;
}

/** Imperative controller returned by the host architecture selector service. */
export interface HostArchitectureSelectorController {
  element: HTMLDivElement;
  setDisabled: (disabled: boolean) => void;
  updateItems: (items: HostArchitectureSelectorItem[]) => void;
}
