/**
 * Shared type contracts for network-view overlays.
 *
 * The most notable overlays are the input-group label bands and the per-input
 * row descriptions. Together they turn a raw input shelf into a readable
 * teaching surface instead of a flat strip of anonymous nodes.
 */

import type { NetworkVisualizationLodSettings } from '../network-visualization.types';

/**
 * Input-group label band geometry and style contract.
 *
 * Each band identifies a contiguous span of input nodes and the visual style
 * used to render that group marker.
 */
export interface InputGroupLabelBand {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  startNodeIndex: number;
  endNodeIndex: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}

/**
 * One horizontal description aligned to a specific input node.
 *
 * The label sits between the semantic group band and the network itself so the
 * viewer can understand each observation channel without inspecting source.
 */
export interface InputNodeDescriptionLabel {
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  nodeIndex: number;
}

/**
 * One reusable description definition before it is bound to a concrete node row.
 */
export interface InputLabelNodeDefinition {
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
}

/**
 * One reusable semantic input group definition for the shared network visualizer.
 */
export interface InputLabelGroupDefinition {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  nodeDescriptionDefinitions: readonly InputLabelNodeDefinition[];
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}

/**
 * Theme palette fields consumed by the shared network visualizer.
 *
 * The contract grows additively as more consumers are threaded through the
 * settings bag; today `currentRunText` drives the above-tier fallback and
 * legend current-run text, while `statusText` drives legend stroke and overlay
 * focus accents.
 *
 * @example
 * ```ts
 * const palette: NetworkVisualizationPalette = {
 *   currentRunText: '#00ff66',
 *   statusText: '#ff5cff',
 * };
 * ```
 */
export interface NetworkVisualizationPalette {
  /** Fallback color for tier-exceeding scale values, reused for legend current-run text. */
  currentRunText: string;
  /** Activation / status label text color. */
  statusText?: string;
}

/**
 * Theme colors used for headers, node labels, and legend chrome.
 *
 * All fields are optional; omitted colors fall back to the NETWORK_* defaults
 * so zero-settings consumers render value-identically.
 */
export interface NetworkVisualizationThemeColors {
  /** Header/architecture label text color. */
  headerText?: string;
  /** Fill color for node activation labels. */
  nodeLabelFill?: string;
  /** Stroke color for hidden and input nodes. */
  hiddenNodeStroke?: string;
  /** Stroke color for output nodes. */
  outputNodeStroke?: string;
  /** Fill color for output nodes. */
  outputNodeFill?: string;
  /** Background fill for the color legend panel. */
  legendBackground?: string;
  /** Legend panel frame stroke color. */
  legendStroke?: string;
  /** Title color for the legend panel. */
  legendHeader?: string;
  /** Section title color for connection-weight legend rows. */
  legendConnectionTitle?: string;
  /** Section title color for node-bias legend rows. */
  legendBiasTitle?: string;
  /** Text color for legend row labels and swatch descriptions. */
  legendRowText?: string;
}

/**
 * Host-supplied settings bag for the shared network visualization panel.
 *
 * Every field is optional so existing host calls keep rendering
 * value-identically; provided fields are layered over the NETWORK_* defaults
 * in network-view.constants.ts.
 *
 * @example
 * ```ts
 * const settings: NetworkVisualizationSettings = {
 *   inputLabelGroupDefinitions: [],
 *   canvasBackground: '#02050c',
 *   palette: { currentRunText: '#00ff66' },
 * };
 * ```
 */
export interface NetworkVisualizationSettings {
  /**
   * Optional semantic input-label group definitions threaded into the
   * positioned graph scene. An empty array disables input-group bands, while
   * an omitted field keeps the resolved domain fallback behavior.
   */
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[];
  /** Optional canvas background fill overriding the network default. */
  canvasBackground?: string;
  /** Optional palette overrides applied to resolved color scales. */
  palette?: Partial<NetworkVisualizationPalette>;
  /** Optional font family used for all overlay text. */
  fontFamily?: string;
  /** Optional viewport width (px) below which the overlay is hidden. */
  overlayHiddenBreakpointPx?: number;
  /** Optional CSS transition duration (ms) for hover highlights. */
  hoverTransitionDurationMs?: number;
  /** Optional light neon color ramp used for topology heatmaps. */
  lightNeonRamp?: readonly string[];
  /** Optional regular neon color ramp used for strong positive/baseline scales. */
  regularNeonRamp?: readonly string[];
  /** Optional neutral-center blue ramp used for near-zero diverging tiers. */
  centerBlueRamp?: readonly string[];
  /** Optional theme colors used for headers, nodes, and legend chrome. */
  theme?: Partial<NetworkVisualizationThemeColors>;
  /** Optional level-of-detail (LOD) settings for dense-network abstraction. */
  lod?: NetworkVisualizationLodSettings;
}
