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
 *   service's backing store (review F2). Since P4-S2C (1a) the docked-right
 *   shelf slot RESOLVER is the ONE sizing authority behind both surfaces:
 *   the resize owner sizes the backing store through the resolved slot and
 *   the host mount applies the SAME resolved slot to the panel box (the
 *   zero-measure jsdom fallback stays the pinned windowless lane).
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
import { NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT } from '../ai/observation.assembler';
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

/**
 * Network-panel shelf width expressed as a fraction of the viewport width.
 *
 * The docked-right slot law (P4-S2C 1a) keeps the shelf a clearly bounded
 * side dock: about a third of the viewport, and never wider than half (the
 * resolver clamps onto that cap, so a full-viewport box is unreachable).
 */
const PANEL_SHELF_WIDTH_VIEWPORT_FRACTION = 0.3;

/**
 * Shelf top clearance in px under the top HUD band.
 *
 * Keeps the docked-right shelf clear of the top-right race mode selector
 * (`top: 0; right: 0`, Neatenstein precedent) and the top-left boot HUD at
 * the recorded example viewport (the S2C-SIBLINGS sibling geometry).
 */
const PANEL_SHELF_TOP_CLEARANCE_PX = 48;

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
 * One live-carrier panel feed for the live evolved-network resolution
 * (AC-102, Step 26 P4-S4-live-feed).
 *
 * Educational note — WHY THIS SHAPE: the most-forward panel watches the
 * LIVE evolved lane, so the host hands the resolver the watched carrier —
 * the carrier's own live evolved network (real node / connection /
 * activation data) wrapped with the elected candidate view it rode in on.
 * The carrier's OWN network is the authoritative live mind; the candidate
 * view is the fallback so the elected NeatRacingNetworkPanelCandidate can
 * ride into the resolver alone.
 */
export interface NeatRacingNetworkPanelLiveCarrier {
  /** Stable in-fleet agent index assigned by the host. */
  readonly agentId: number;
  /** Host-owned along-track race progress of the watched carrier. */
  readonly progress: number;
  /**
   * The carrier's OWN live evolved network (AC-102). Absent while the live
   * feed has not landed for this carrier — the resolver then raises the
   * named live-carrier error; it NEVER falls back to the retired
   * idle-path stub frame.
   */
  readonly network?: Network | undefined;
  /**
   * The elected fleet candidate this carrier wraps (the carrier's fallback
   * network view).
   */
  readonly candidate?: NeatRacingNetworkPanelCandidate | undefined;
}

/**
 * Named live-carrier resolution error prefix (the AC-102 stub-retirement
 * law): a network-less carrier is never silently answered with the retired
 * idle-path stub frame — the failure NAMES the live-carrier contract.
 */
const PANEL_LIVE_CARRIER_ERROR_PREFIX =
  'NeatRacingNetworkPanelLiveCarrierError';

/**
 * Reads the live network slot off a live-carrier-shaped argument (any
 * object carrying a `network`, or a `candidate` view carrying one).
 *
 * The carrier's OWN network wins; the candidate view is the fallback (the
 * elected NeatRacingNetworkPanelCandidate may ride into the resolver alone).
 * A non-object argument or a carrier that carries no network at either
 * slot resolves to undefined and the caller raises the named error.
 *
 * @param carrier - The carrier-shaped argument to read.
 * @returns The carried live network object, or undefined.
 */
function readNeatRacingNetworkPanelCarrierNetwork(
  carrier: unknown,
): object | undefined {
  // Step 1: A non-object argument carries no network slot at all.
  if (typeof carrier !== 'object' || carrier === null) {
    return undefined;
  }
  const carrierRecord = carrier as Record<string, unknown>;
  // Step 2: The carrier's OWN network is the authoritative live mind.
  const ownNetwork = carrierRecord['network'];
  if (typeof ownNetwork === 'object' && ownNetwork !== null) {
    return ownNetwork;
  }
  // Step 3: The candidate view is the fallback network slot.
  const candidate = carrierRecord['candidate'];
  if (typeof candidate !== 'object' || candidate === null) {
    return undefined;
  }
  const candidateNetwork = (candidate as Record<string, unknown>)['network'];
  return typeof candidateNetwork === 'object' && candidateNetwork !== null
    ? candidateNetwork
    : undefined;
}

/**
 * Resolves the shared visualizer frame for a LIVE evolved carrier (AC-102,
 * Step 26 P4-S4-live-feed — the RED-P4-9 live-feed seam).
 *
 * Educational note — WHAT/WHY: the docked-right panel's idle-path stub
 * frame (the 82-channel/2-output placeholder) retires with the live feed —
 * with a live carrier present the most-forward panel renders the carrier's
 * OWN evolved network (real node / connection / activation data). This
 * resolver is the adapter's live-carrier seam: it re-shapes the CARRIER
 * argument into the existing live-network frame path and delegates the
 * whole frame to the shared visualizer (L5 — thin adapter), so the frame
 * topology, LOD behavior and activation reads stay the shared laws.
 *
 * Educational note — ACTIVATIONS ARE CAPTURED AT RESOLVE TIME: the shared
 * builder normalizes the carrier network's runtime nodes into fresh
 * view-layer records (`activation: runtimeNode.activation ?? 0`), so the
 * resolved frame carries the activation values AS OF THIS RESOLVE — a
 * consumer that CACHES the frame (the documented
 * drawNeatRacingNetworkPanelFromFrame hover-repaint path — and the
 * fullscreen surface slice, the next frame consumer) MUST re-resolve to
 * observe a later activation change. The resolution path itself adds NO
 * per-call allocation beyond the delegated shared frame build (AC-102's
 * zero-per-frame-allocation law). An extra `activations` argument on the
 * carrier is deliberately ignored: the network's nodes are the source of
 * truth for the activations captured at resolve time.
 *
 * @param context - Canvas 2D drawing context of the network panel.
 * @param carrier - The watched live carrier (its own network, or an
 *   elected {@link NeatRacingNetworkPanelCandidate} riding in alone).
 * @returns The shared visualizer frame for the carrier's live network, with
 *   the network's activation values captured as of this resolve time
 *   (re-resolve to observe later activation changes).
 * @throws When the carrier carries no network at either slot (its own slot
 *   or its candidate view) — the named live-carrier error; the retired
 *   idle-path stub frame is NEVER a legal live-carrier answer (AC-102).
 *
 * @example
 * ```ts
 * const carrier: NeatRacingNetworkPanelLiveCarrier = {
 *   agentId: 2,
 *   progress: 120,
 *   network: liveEvolvedNetwork,
 * };
 * const frame = resolveNeatRacingNetworkPanelLiveCarrierFrame(
 *   panelContext,
 *   carrier,
 * );
 * drawNeatRacingNetworkPanelFromFrame(panelContext, frame);
 * ```
 */
export function resolveNeatRacingNetworkPanelLiveCarrierFrame(
  context: CanvasRenderingContext2D,
  carrier: NeatRacingNetworkPanelLiveCarrier | NeatRacingNetworkPanelCandidate,
): NetworkVisualizationResolvedFrame {
  // Step 1: Read the carried live network (own slot first, candidate view
  //         as the fallback). A network-less carrier NEVER degrades into
  //         the idle stub — it fails LOUDLY with the named error.
  const liveNetwork = readNeatRacingNetworkPanelCarrierNetwork(carrier);
  if (liveNetwork === undefined) {
    throw new Error(
      `${PANEL_LIVE_CARRIER_ERROR_PREFIX}: the live-carrier frame resolution requires a carrier carrying a live evolved network — a network-less carrier is NEVER answered with the retired idle-path stub frame (AC-102)`,
    );
  }
  // Step 2: Delegate the whole frame to the shared visualizer path (L5).
  //         The carrier carries the live Network instance per the panel
  //         candidate contract (`NeatRacingNetworkPanelCandidate.network`).
  return resolveNeatRacingNetworkPanelVisualizationFrame(
    context,
    liveNetwork as Network,
  );
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
 * Docked-right shelf slot resolver (pure, DOM-free — P4-S2C 1a)
 * ------------------------------------------------------------------------- */

/**
 * One resolved docked-right shelf slot for the network panel.
 *
 * @property leftPx - Slot left edge in viewport px (viewport-anchored).
 * @property topPx - Slot top edge in viewport px (below the top HUD band).
 * @property widthPx - Slot width in px (strictly bounded, at most half the
 *   viewport width — never a full-viewport box).
 * @property heightPx - Slot height in px (strictly bounded inside the
 *   viewport height).
 */
export interface NeatRacingNetworkPanelShelfSlot {
  readonly leftPx: number;
  readonly topPx: number;
  readonly widthPx: number;
  readonly heightPx: number;
}

/**
 * Resolves the deterministic docked-right network-panel shelf slot
 * (P4-S2C 1a — RED-P4-20).
 *
 * The pure, DOM-free ONE sizing authority of the 1a panel law: the viewport
 * is mapped onto the slot the right-docked panel rides, so a full-viewport
 * panel box is unreachable by construction. The resolver owns four laws:
 *
 * - right-anchored x (`leftPx + widthPx === viewportWidthPx`),
 * - strictly inside the viewport on both axes,
 * - strictly bounded extent (at least 1px and never a full-viewport side),
 * - the width never exceeds half the viewport width.
 *
 * Determinism claim (reproducibility rank: deterministic on the same
 * runtime): the slot is a pure function of its two arguments — no window
 * reads, no host measure, no DOM — so the same arguments always resolve the
 * same slot regardless of the surrounding window state. The host mount and
 * the resize service compose that one result instead of inventing separate
 * geometry (ONE sizing authority, L6).
 *
 * @param viewportWidthPx - Viewport width in CSS px (the panel docks into
 *   the full-viewport hud-output layer, so viewport px map 1:1 onto it).
 * @param viewportHeightPx - Viewport height in CSS px.
 * @returns The docked-right bounded slot to size and position the panel
 *   with.
 *
 * @example
 * ```ts
 * const slot = resolveNeatRacingNetworkPanelDockedRightShelfSlot(1689, 1325);
 * // -> { leftPx: 1183, topPx: 48, widthPx: 506, heightPx: 379 }
 * panelCanvas.style.left = `${slot.leftPx}px`;
 * panelCanvas.style.width = `${slot.widthPx}px`;
 * ```
 */
export function resolveNeatRacingNetworkPanelDockedRightShelfSlot(
  viewportWidthPx: number,
  viewportHeightPx: number,
): NeatRacingNetworkPanelShelfSlot {
  // Step 1: The bounded shelf width — a fraction of the viewport, clamped
  // onto the half-viewport cap so a full-viewport box is unreachable.
  const widthPx = Math.min(
    Math.ceil(viewportWidthPx / 2),
    Math.max(
      1,
      Math.floor(viewportWidthPx * PANEL_SHELF_WIDTH_VIEWPORT_FRACTION),
    ),
  );

  // Step 2: The aspect-locked height, strictly bounded inside the viewport.
  const heightPx = Math.min(
    Math.max(1, viewportHeightPx - 1),
    Math.max(1, Math.floor(widthPx * PANEL_CANVAS_ASPECT_RATIO)),
  );

  // Step 3: Right-anchor the slot onto the viewport's right edge.
  const leftPx = viewportWidthPx - widthPx;

  // Step 4: Dock below the top HUD band, clamped so the slot stays inside.
  const topPx = Math.max(
    0,
    Math.min(PANEL_SHELF_TOP_CLEARANCE_PX, viewportHeightPx - heightPx),
  );

  return { leftPx, topPx, widthPx, heightPx };
}

/* ------------------------------------------------------------------------- *
 * Resize service (installRacingNetworkResize pattern)
 * ------------------------------------------------------------------------- */

/**
 * Installs NeatRacing network canvas resize listeners.
 *
 * Mirrors the `installRacingNetworkResize` pattern exactly on the wiring
 * side: on every window `resize` event the service sizes the canvas backing
 * store once per event, then invokes `onResize` so the caller can schedule a
 * network redraw (only when the store actually changed — a no-op resize
 * never repaints the idle draw path). On install the size is applied once
 * immediately. This service is the backing-store OWNER of the racing-parity
 * panel wiring (L6, review F2): pair it with
 * `resolveNeatRacingNetworkPanelVisualizationFrame` +
 * `drawNeatRacingNetworkPanelFromFrame`, never with the self-sizing
 * `drawNeatRacingNetworkPanel`.
 *
 * Since P4-S2C (1a — RED-P4-21) a MEASURABLE host (positive `clientWidth`
 * AND `clientHeight`) rides the docked-right slot: the service resolves the
 * same pure `resolveNeatRacingNetworkPanelDockedRightShelfSlot` slot from
 * the WINDOW dimensions (ONE sizing authority with the panel box the host
 * mount applies), never from the raw host bounds that defined the recorded
 * full-viewport defect.
 *
 * If the host element has no CSS layout yet (for example in jsdom tests), the
 * service falls back to `window.innerWidth * 0.25` by
 * `window.innerHeight * 0.35` so the canvas still has a positive backing size
 * (the pinned zero-measure lane, unchanged).
 * The returned `redraw` handle is intentionally a no-op; the host schedules
 * redraws through `onResize`.
 *
 * @param networkCanvas - Network visualization canvas to resize.
 * @param networkCanvasHost - Host element providing the measurability gate
 *   (its zero client extents route the pinned fallback lane).
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
    // Step 1: Resolve the ONE sizing authority (the 1a slot law). A
    // measurable host (the real browser mount) rides the docked-right slot
    // through the pure resolver — never its raw bounds; the zero-measure
    // fallback lane keeps the pinned viewport fractions.
    const hostMeasurable =
      networkCanvasHost.clientWidth > 0 && networkCanvasHost.clientHeight > 0;
    const shelfSlot = hostMeasurable
      ? resolveNeatRacingNetworkPanelDockedRightShelfSlot(
          window.innerWidth,
          window.innerHeight,
        )
      : null;
    const fallbackWidth = Math.max(1, Math.floor(window.innerWidth * 0.25));
    const fallbackHeight = Math.max(1, Math.floor(window.innerHeight * 0.35));
    const widthPx = shelfSlot !== null ? shelfSlot.widthPx : fallbackWidth;
    const heightPx = shelfSlot !== null ? shelfSlot.heightPx : fallbackHeight;

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
