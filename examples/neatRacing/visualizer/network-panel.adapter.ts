/**
 * NeatRacing network panel adapter over the shared rich browser network visualizer (AC-034).
 *
 * The NeatRacing browser page must not own its own network-frame math. Just like
 * the racing curriculum adapter, this module reuses the shared resolved frame,
 * padding, node sizing, and connection drawing path while only swapping the
 * NeatRacing-specific pieces: the most-forward-agent selection law, the NGE
 * controller output labels, and the NGE input/output sizes.
 *
 * ## Panel law (LD-32 — "most forward agent only", AC-034)
 *
 * - **L1 — Progress is host-owned.** The packed sim frame (`neatracing-packed-v1`)
 *   carries kinematics only: no laps, no along-track distance, no networks. The
 *   adapter therefore NEVER derives race progress from world XY or track
 *   geometry; the host wiring passes an along-track progress metric per agent
 *   (for example `laps * trackLength + along-track distance`). The panel stays
 *   free of track-geometry knowledge by construction.
 * - **L2 — Sticky incumbent.** The current leader keeps the panel unless another
 *   candidate's progress is STRICTLY greater, so an equal-progress tick or a
 *   near-tie never flickers the panel ("the current winner at all times").
 * - **L3 — Non-finite progress never leads.** Candidates whose progress is
 *   `NaN` or infinite are invisible to the selection, mirroring the recycled
 *   finite-number law of the recycled goals machine (`ai/ai-mode.goals.ts`).
 * - **L4 — Deterministic tie-break.** Without a present incumbent, the maximum
 *   finite progress wins; an exact progress tie resolves to the LOWEST
 *   agent id, so repeated ticks resolve identically for the same inputs.
 * - **L5 — Thin over the shared visualizer.** Frames are resolved through
 *   `examples/shared/network-visualization` with the shared LOD defaults and
 *   the shared default input-label path; the NeatRacing observation channels
 *   stay owner-defined in `ai/observation.assembler.ts` and are not restated
 *   here.
 * - **L6 — One sizing authority per canvas.** The racing-parity panel wiring
 *   (P2-S4-C) composes `installNeatRacingNetworkResize` (the backing-store
 *   owner) with `resolveNeatRacingNetworkPanelVisualizationFrame` +
 *   `drawNeatRacingNetworkPanelFromFrame` on the same canvas. The
 *   all-in-one `drawNeatRacingNetworkPanel` is the standalone self-sizing
 *   path and must NOT be composed with the resize service on the same
 *   canvas, because its sync pass would silently override the resize
 *   service's backing store (review F2).
 *
 * @module
 */

import type Network from '../../../src/architecture/network';
import {
  drawResolvedNetworkVisualization,
  resolveNetworkArchitectureLabel as resolveSharedNetworkArchitectureLabel,
  resolveNetworkVisualizationFrame as resolveSharedNetworkVisualizationFrame,
  type NetworkVisualizationResolvedFrame,
} from '../../shared/network-visualization/network-view/network-view';
import type { NetworkVisualizationSettings } from '../../shared/network-visualization/network-view/network-view.types';
import type {
  NetworkVisualizationHoverState,
  NetworkVisualizationPositionedScene,
  PositionedNetworkNodeLike,
} from '../../shared/network-visualization/network-visualization.types';

/**
 * Re-export of the shared cached-frame type so the host wiring (P2-S4-C) can
 * name its resolved frame without reaching past this adapter into
 * `examples/shared/network-visualization` (racing-adapter parity, review F1).
 */
export type { NetworkVisualizationResolvedFrame };

/**
 * Re-export of the shared positioned-scene type so the host wiring (P2-S4-C)
 * can name hover hit-testing state without reaching past this adapter.
 */
export type { NetworkVisualizationPositionedScene };
import {
  NETWORK_LOD_HIDDEN_CLUSTER_COUNT,
  NETWORK_LOD_HIDDEN_NODE_THRESHOLD,
  NETWORK_LOD_HOVER_MAX_LOCAL_NODES,
} from '../../shared/network-visualization/network-visualization.constants';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from '../ai/nge.controller';
import {
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
} from '../ai/observation.assembler';
import { NEATRACING_HUD_COLOR_CYAN } from '../host/hud';

/* ------------------------------------------------------------------------- *
 * Owner-local panel constants
 * ------------------------------------------------------------------------- */

/**
 * Ordered NGE controller output labels.
 *
 * Law: the first two channels are the pinned NGE control axes (index 0 =
 * THROTTLE, index 1 = STEER) and channels 2..8 stream the degenerate
 * single-car self-radio seam writes (`R0`..`R6`) on radio-augmented ladder
 * levels. Output indices past the table fall back to `OUT<index>` so a wider
 * evolved output layer still renders deterministically.
 */
export const NEATRACING_NETWORK_OUTPUT_LABELS: readonly string[] = [
  'THR',
  'STR',
  'R0',
  'R1',
  'R2',
  'R3',
  'R4',
  'R5',
  'R6',
] as const;

/** Minimum network panel canvas backing-store width. */
const PANEL_MIN_CANVAS_WIDTH_PX = 320;

/** Minimum network panel canvas backing-store height. */
const PANEL_MIN_CANVAS_HEIGHT_PX = 240;

/** Network panel canvas height expressed as a fraction of the canvas width. */
const PANEL_CANVAS_ASPECT_RATIO = 0.75;

/** Output-label ink inherits the HUD cyan token (single visual identity). */
const PANEL_OUTPUT_LABEL_COLOR = NEATRACING_HUD_COLOR_CYAN;

/** Output-label font (same compact mono stack as the racing panel). */
const PANEL_OUTPUT_LABEL_FONT = '700 10px Consolas, Menlo, Monaco, monospace';

/** Output-label horizontal offset from the output node anchor, in px. */
const PANEL_OUTPUT_LABEL_OFFSET_PX = 18;

/* ------------------------------------------------------------------------- *
 * Public types
 * ------------------------------------------------------------------------- */

/**
 * One host-supplied fleet candidate for the most-forward network panel.
 *
 * The host wiring owns both the identity and the race progress of every car;
 * the adapter only compares the numbers it is handed.
 */
export interface NeatRacingNetworkPanelCandidate {
  /** Stable in-fleet agent index assigned by the host (0 = focal car slot). */
  readonly agentId: number;
  /** Host-owned along-track race progress; higher means more forward. */
  readonly progress: number;
  /**
   * Evolved controller network snapshot; absent for non-network agents.
   *
   * Selection NEVER inspects this field (review F3): network presence does
   * not affect who wins. When the elected winner carries no network the
   * panel renders its idle state — the panel must NOT fall back to a
   * networked runner-up (AC-034: only the most-forward agent is shown).
   */
  readonly network?: Network | undefined;
}

/* ------------------------------------------------------------------------- *
 * Most-forward selection (pure)
 * ------------------------------------------------------------------------- */

/**
 * Resolves the agent whose network the panel renders this tick.
 *
 * Law (module header L1-L4): progress is host-owned, the incumbent stays
 * leader unless a challenger is STRICTLY more forward, non-finite progress
 * never leads, and the no-incumbent tie-break is lowest agent id.
 *
 * @param candidates - Host-supplied fleet candidates for this tick.
 * @param incumbentAgentId - Agent id that currently owns the panel, if any.
 * @returns The winning candidate, or `undefined` when no finite-progress
 *   candidate exists (the host then parks the panel). Network presence never
 *   affects the selection: when the elected winner carries no network the
 *   panel renders its idle state and the host must NOT fall back to a
 *   networked runner-up (review F3, AC-034).
 *
 * @example
 * ```ts
 * const leader = resolveMostForwardNeatRacingAgent(
 *   fleet.map((car, agentId) => ({
 *     agentId,
 *     progress: car.raceProgressMeters,
 *     network: car.network,
 *   })),
 *   renderedAgentId,
 * );
 * ```
 */
export function resolveMostForwardNeatRacingAgent(
  candidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbentAgentId?: number,
): NeatRacingNetworkPanelCandidate | undefined {
  // Step 1: L3 — drop candidates whose progress can never lead.
  const finiteCandidates = candidates.filter((candidate) =>
    Number.isFinite(candidate.progress),
  );
  if (finiteCandidates.length === 0) {
    return undefined;
  }

  // Step 2: L2 — a present incumbent only vacates on a strictly better run.
  if (incumbentAgentId !== undefined) {
    const incumbent = finiteCandidates.find(
      (candidate) => candidate.agentId === incumbentAgentId,
    );
    if (incumbent) {
      return resolveMostForwardWithoutIncumbent(finiteCandidates, incumbent);
    }
  }

  // Step 3: L4 — no incumbent present; maximum progress, lowest id wins.
  return resolveMostForwardWithoutIncumbent(finiteCandidates);
}

/**
 * Applies the L4 max-progress/lowest-id law against the L2 sticky rule.
 *
 * @param finiteCandidates - Candidates with finite progress only.
 * @param incumbent - Incumbent candidate kept on ties, when supplied.
 * @returns The candidate that owns the panel this tick.
 */
function resolveMostForwardWithoutIncumbent(
  finiteCandidates: readonly NeatRacingNetworkPanelCandidate[],
  incumbent?: NeatRacingNetworkPanelCandidate,
): NeatRacingNetworkPanelCandidate {
  const mostForward = finiteCandidates.reduce((best, candidate) => {
    if (candidate.progress > best.progress) {
      return candidate;
    }

    // L4: exact progress ties resolve to the lowest agent id.
    if (
      candidate.progress === best.progress &&
      candidate.agentId < best.agentId
    ) {
      return candidate;
    }

    return best;
  });

  // L2: the incumbent survives any non-strictly-better challenger.
  if (incumbent && mostForward.progress <= incumbent.progress) {
    return incumbent;
  }

  return mostForward;
}

/* ------------------------------------------------------------------------- *
 * Shared visualizer pass-throughs (thin)
 * ------------------------------------------------------------------------- */

/**
 * Resolves responsive network-panel canvas dimensions from the host panel
 * shelf, mirroring the racing panel dimension law: the measured width is
 * clamped to a positive minimum, the height falls back to the width-driven
 * aspect value, and a positive measured panel height wins over the fallback.
 *
 * @param measuredWidthPx - Current measured canvas width from layout.
 * @param measuredHeightPx - Current measured host-panel height from layout.
 * @returns Width and height for the canvas backing store.
 *
 * @example
 * ```ts
 * resolveNeatRacingNetworkPanelCanvasDimensions(0, 0);
 * // -> { widthPx: 320, heightPx: 240 }
 * ```
 */
export function resolveNeatRacingNetworkPanelCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): {
  widthPx: number;
  heightPx: number;
} {
  // Step 1: Clamp the width onto the positive minimum shelf.
  const resolvedWidthPx = Math.max(
    PANEL_MIN_CANVAS_WIDTH_PX,
    Math.floor(measuredWidthPx),
  );

  // Step 2: Resolve the width-driven aspect height.
  const widthDrivenHeightPx = Math.max(
    PANEL_MIN_CANVAS_HEIGHT_PX,
    Math.floor(resolvedWidthPx * PANEL_CANVAS_ASPECT_RATIO),
  );

  // Step 3: A positive measured panel height overrides the aspect fallback.
  const measuredPanelHeightPx = Math.floor(measuredHeightPx);
  const resolvedHeightPx =
    Number.isFinite(measuredPanelHeightPx) && measuredPanelHeightPx > 0
      ? measuredPanelHeightPx
      : widthDrivenHeightPx;

  return {
    widthPx: resolvedWidthPx,
    heightPx: resolvedHeightPx,
  };
}

/**
 * Resolves the compact architecture summary for the NeatRacing network legend.
 *
 * @param network - Runtime network being visualized.
 * @returns Shared architecture label with NeatRacing IO counts; the input
 *   size falls back to the default ladder level width (Tier A, 82 channels)
 *   when the runtime network does not expose `input`.
 */
export function resolveNeatRacingNetworkPanelArchitectureLabel(
  network: Network,
): string {
  return resolveSharedNetworkArchitectureLabel(
    network,
    network.input ?? NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
  );
}

/**
 * Builds the NeatRacing settings bag that drives the shared visualizer path.
 *
 * LOD stays force-enabled with the shared LOD defaults so dense evolved
 * controllers collapse into the shared abstract renderer. Input-label groups
 * are deliberately NOT overridden (L5): the panel renders the shared default
 * input-label path because the NeatRacing observation channels stay
 * owner-defined in `ai/observation.assembler.ts`.
 *
 * @returns NeatRacing settings bag for the shared visualizer.
 */
function buildNeatRacingNetworkPanelSettings(): NetworkVisualizationSettings {
  return {
    lod: {
      enabled: true,
      hiddenNodeThreshold: NETWORK_LOD_HIDDEN_NODE_THRESHOLD,
      clusterCount: NETWORK_LOD_HIDDEN_CLUSTER_COUNT,
      hoverMaxLocalNodes: NETWORK_LOD_HOVER_MAX_LOCAL_NODES,
    },
  };
}

/**
 * Resolves a reusable network-visualization frame for the NeatRacing panel.
 *
 * Host-owned caching can reuse this frame between pointer-driven redraws so
 * the expensive topology/layout work is not repeated for every hover event.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize, or `undefined` for an idle panel.
 * @returns Reusable resolved frame for subsequent draw passes.
 *
 * @example
 * ```ts
 * const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
 *   context,
 *   leaderNetwork,
 * );
 * drawNeatRacingNetworkPanelFromFrame(context, frame, hoveredNodeIndices);
 * ```
 */
export function resolveNeatRacingNetworkPanelVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
): NetworkVisualizationResolvedFrame {
  // Step 1: The panel input size mirrors the runtime network, Tier A default.
  const inputSize =
    network?.input ?? NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT;

  // Step 2: Delegate the whole frame to the shared path (L5 — thin adapter).
  return resolveSharedNetworkVisualizationFrame(
    context,
    network,
    inputSize,
    NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT,
    undefined,
    buildNeatRacingNetworkPanelSettings(),
  );
}

/* ------------------------------------------------------------------------- *
 * Draw paths
 * ------------------------------------------------------------------------- */

/**
 * Resolves the host-supplied hover indices into the shared hover state.
 *
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @returns Shared hover state, or `undefined` when nothing is hovered.
 */
function resolveNeatRacingHoverState(
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationHoverState | undefined {
  if (!hoveredNodeIndices || hoveredNodeIndices.length === 0) {
    return undefined;
  }

  return {
    hoveredNodeIndices,
    animatedHoveredNodes: hoveredNodeIndices.map((nodeIndex) => ({
      nodeIndex,
      intensity: 1,
    })),
  };
}

/**
 * Sorts the positioned output nodes top-to-bottom so labels map onto the
 * `NEATRACING_NETWORK_OUTPUT_LABELS` table deterministically.
 *
 * @param positionedNodes - Positioned nodes of the current frame.
 * @returns Output nodes sorted by ascending y.
 */
function resolveSortedNeatRacingOutputNodes(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): PositionedNetworkNodeLike[] {
  return positionedNodes
    .filter((positionedNode) => positionedNode.node.type === 'output')
    .toSorted(
      (positionedNodeA, positionedNodeB) =>
        positionedNodeA.yPx - positionedNodeB.yPx,
    );
}

/**
 * Draws the NGE controller output tags next to the positioned output nodes.
 *
 * @param context - Canvas 2D drawing context.
 * @param outputNodes - Output nodes sorted by ascending y.
 */
function drawNeatRacingOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: readonly PositionedNetworkNodeLike[],
): void {
  context.save();
  context.fillStyle = PANEL_OUTPUT_LABEL_COLOR;
  context.font = PANEL_OUTPUT_LABEL_FONT;
  context.textAlign = 'left';
  context.textBaseline = 'middle';

  outputNodes.forEach((outputNode, outputIndex) => {
    const outputLabel =
      NEATRACING_NETWORK_OUTPUT_LABELS[outputIndex] ?? `OUT${outputIndex}`;
    context.fillText(
      outputLabel,
      outputNode.xPx + PANEL_OUTPUT_LABEL_OFFSET_PX,
      outputNode.yPx,
    );
  });

  context.restore();
}

/**
 * Draws a previously resolved NeatRacing network-panel frame.
 *
 * The host uses this path for hover-only repaint work because it can reuse
 * the cached static frame and only vary interactive emphasis.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedFrame - Reusable frame cache.
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawNeatRacingNetworkPanelFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationResolvedFrame,
  hoveredNodeIndices?: readonly number[],
): NetworkVisualizationPositionedScene {
  // Step 1: Paint the shared path with the resolved hover emphasis.
  const positionedScene = drawResolvedNetworkVisualization(
    context,
    resolvedFrame,
    resolveNeatRacingHoverState(hoveredNodeIndices),
  );

  // Step 2: Stamp the NeatRacing NGE controller output tags.
  drawNeatRacingOutputNodeLabels(
    context,
    resolveSortedNeatRacingOutputNodes(positionedScene.positionedNodes),
  );

  return positionedScene;
}

/**
 * Resolves the host content-box dimension without double counting padding.
 *
 * Mirrors the racing adapter: the host element's computed padding is
 * subtracted from the measured client dimension so the backing store fits the
 * visible shelf.
 *
 * @param hostElement - Host content element, or `null`.
 * @param hostClientDimensionPx - Fallback client dimension of the host.
 * @param startPaddingProperty - Padding property on the axis start.
 * @param endPaddingProperty - Padding property on the axis end.
 * @returns Resolved content-box dimension in px.
 */
function resolveHostContentBoxDimensionPx(
  hostElement: HTMLElement | null,
  hostClientDimensionPx: number,
  startPaddingProperty:
    'paddingBottom' | 'paddingLeft' | 'paddingRight' | 'paddingTop',
  endPaddingProperty:
    'paddingBottom' | 'paddingLeft' | 'paddingRight' | 'paddingTop',
): number {
  if (!hostElement) {
    return hostClientDimensionPx;
  }

  const defaultView = hostElement.ownerDocument?.defaultView;
  const computedStyle = defaultView?.getComputedStyle(hostElement);
  const startPaddingPx = Number.parseFloat(
    computedStyle?.[startPaddingProperty] ?? '0',
  );
  const endPaddingPx = Number.parseFloat(
    computedStyle?.[endPaddingProperty] ?? '0',
  );
  const resolvedStartPaddingPx = Number.isFinite(startPaddingPx)
    ? startPaddingPx
    : 0;
  const resolvedEndPaddingPx = Number.isFinite(endPaddingPx) ? endPaddingPx : 0;

  return Math.max(
    0,
    hostClientDimensionPx - resolvedStartPaddingPx - resolvedEndPaddingPx,
  );
}

/**
 * Syncs the network panel canvas backing store to the host panel shelf.
 *
 * @param canvas - Canvas element to keep in sync with its host shelf.
 */
function syncNeatRacingCanvasToPanel(canvas: HTMLCanvasElement): void {
  const hostElement = canvas.parentElement;
  const measuredWidthPx = resolveHostContentBoxDimensionPx(
    hostElement,
    hostElement?.clientWidth ?? canvas.clientWidth,
    'paddingLeft',
    'paddingRight',
  );
  const measuredHeightPx = resolveHostContentBoxDimensionPx(
    hostElement,
    hostElement?.clientHeight ?? canvas.clientHeight,
    'paddingTop',
    'paddingBottom',
  );
  const { widthPx: resolvedWidthPx, heightPx: resolvedHeightPx } =
    resolveNeatRacingNetworkPanelCanvasDimensions(
      measuredWidthPx,
      measuredHeightPx,
    );

  canvas.style.width = `${resolvedWidthPx}px`;
  canvas.style.height = `${resolvedHeightPx}px`;

  if (canvas.width !== resolvedWidthPx || canvas.height !== resolvedHeightPx) {
    canvas.width = resolvedWidthPx;
    canvas.height = resolvedHeightPx;
  }
}

/**
 * Draw the NeatRacing network panel for one network using the shared
 * network visualizer.
 *
 * Composition law (L6, review F2): this is the STANDALONE self-sizing entry
 * point — it owns the canvas backing store through its sync pass. The
 * racing-parity panel wiring must instead compose
 * `installNeatRacingNetworkResize` with
 * `resolveNeatRacingNetworkPanelVisualizationFrame` +
 * `drawNeatRacingNetworkPanelFromFrame`; do not pair this function with the
 * resize service on the same canvas.
 *
 * @param canvas - Canvas element to render onto.
 * @param network - Runtime network of the most-forward agent, or `undefined`
 *   while the panel waits for the first network (renders the idle panel).
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 *
 * @example
 * ```ts
 * const leader = resolveMostForwardNeatRacingAgent(candidates, incumbentId);
 * drawNeatRacingNetworkPanel(networkPanelCanvas, leader?.network);
 * ```
 */
export function drawNeatRacingNetworkPanel(
  canvas: HTMLCanvasElement,
  network: Network | undefined,
  hoveredNodeIndices?: readonly number[],
): NetworkVisualizationPositionedScene {
  // Step 1: Keep the backing store in sync with the host panel shelf.
  syncNeatRacingCanvasToPanel(canvas);

  // Step 2: Park on an empty scene when the canvas has no 2D context.
  const context = canvas.getContext('2d', { desynchronized: true });
  if (!context) {
    return {
      positionedNodes: [],
      nodeDimensions: { widthPx: 0, heightPx: 0 },
      inputDescriptionScenes: [],
      inputGroupLabelBandScenes: [],
      hiddenColumnLabelScenes: [],
    } as NetworkVisualizationPositionedScene;
  }

  // Step 3: Resolve once and paint through the shared path.
  const resolvedFrame = resolveNeatRacingNetworkPanelVisualizationFrame(
    context,
    network,
  );
  return drawNeatRacingNetworkPanelFromFrame(
    context,
    resolvedFrame,
    hoveredNodeIndices,
  );
}

/* ------------------------------------------------------------------------- *
 * Resize service (installRacingNetworkResize pattern)
 * ------------------------------------------------------------------------- */

/**
 * Installs NeatRacing network canvas resize listeners.
 *
 * Mirrors the `installRacingNetworkResize` pattern exactly: on every window
 * `resize` event the service measures the host element and updates the canvas
 * backing store, then invokes `onResize` so the caller can schedule a network
 * redraw. On install the size is applied once immediately. This service is
 * the backing-store OWNER of the racing-parity panel wiring (L6, review F2):
 * pair it with `resolveNeatRacingNetworkPanelVisualizationFrame` +
 * `drawNeatRacingNetworkPanelFromFrame`, never with the self-sizing
 * `drawNeatRacingNetworkPanel`.
 *
 * If the host element has no CSS layout yet (for example in jsdom tests), the
 * service falls back to `window.innerWidth * 0.25` by
 * `window.innerHeight * 0.35` so the canvas still has a positive backing size.
 * The returned `redraw` handle is intentionally a no-op; the host schedules
 * redraws through `onResize`.
 *
 * @param networkCanvas - Network visualization canvas to resize.
 * @param networkCanvasHost - Host element whose bounds drive the canvas size.
 * @param onResize - Callback invoked after the canvas backing size changes.
 * @returns Uninstall, resize, and redraw handles.
 *
 * @example
 * ```ts
 * const { uninstall, resize } = installNeatRacingNetworkResize(
 *   networkPanelCanvas,
 *   networkPanelHost,
 *   () => {
 *     const context = networkPanelCanvas.getContext('2d', {
 *       desynchronized: true,
 *     });
 *     if (!context) {
 *       return;
 *     }
 *     const frame = resolveNeatRacingNetworkPanelVisualizationFrame(
 *       context,
 *       leaderNetwork,
 *     );
 *     drawNeatRacingNetworkPanelFromFrame(context, frame);
 *   },
 * );
 * resize();
 * window.addEventListener('beforeunload', uninstall);
 * ```
 */
export function installNeatRacingNetworkResize(
  networkCanvas: HTMLCanvasElement,
  networkCanvasHost: HTMLElement,
  onResize: () => void,
): { uninstall: () => void; resize: () => void; redraw: () => void } {
  const applySize = (): void => {
    // Step 1: Measure the host shelf, falling back to the viewport fraction.
    const hostWidth = networkCanvasHost.clientWidth;
    const hostHeight = networkCanvasHost.clientHeight;
    const fallbackWidth = Math.max(1, Math.floor(window.innerWidth * 0.25));
    const fallbackHeight = Math.max(1, Math.floor(window.innerHeight * 0.35));
    const widthPx = hostWidth > 0 ? hostWidth : fallbackWidth;
    const heightPx = hostHeight > 0 ? hostHeight : fallbackHeight;

    // Step 2: Only report a resize when the backing store actually changed.
    if (networkCanvas.width !== widthPx || networkCanvas.height !== heightPx) {
      networkCanvas.width = widthPx;
      networkCanvas.height = heightPx;
      onResize();
    }
  };

  const handleWindowResize = (): void => {
    applySize();
  };

  applySize();
  window.addEventListener('resize', handleWindowResize);

  return {
    uninstall: () => {
      window.removeEventListener('resize', handleWindowResize);
    },
    resize: applySize,
    redraw: () => {
      // no-op; host handles redraw
    },
  };
}
