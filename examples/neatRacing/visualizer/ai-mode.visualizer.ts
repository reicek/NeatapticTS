/**
 * AI-mode visualizer surface for the NeatRacing demo (P4-S4, Step 26 —
 * RED-P4-10).
 *
 * The module carries the FULLSCREEN-CAPABLE AI-mode network visualizer
 * surface (AC-103) over the shared network visualization: the surface owns
 * the AI-mode presentation state (visible by default at the docked shelf —
 * the CLAR-4-02 recorded default — and togglable between the docked shelf
 * and the fullscreen presentation) and the AI lane's live-carrier draw
 * seam (the elected winner rides the adapter's live-carrier frame law).
 *
 * The module is DOM-FREE by contract: the factory resolves a plain state
 * record (no document, no window, no canvas), and the frame resolution is
 * a delegation onto the panel adapter's live-carrier law with the
 * host-owned canvas 2D context handed in per call. The DOM wiring (mount,
 * presentation slot application, the EXPLICIT HUD toggle control and
 * the resize authority) lives in the browser entry, which keeps ALL
 * sizing/presentation decisions flowing through pure resolvers (the L6
 * slot law: the docked-right shelf resolver in the panel adapter stays
 * untouched and remains the DOCKED presentation's ONE sizing authority;
 * the fullscreen slot below is the surface's OWN presentation law).
 *
 * Educational note — THE ADAPTER SEAM (AC-034): the surface never
 * re-implements topology layout, activation capture or drawing. The
 * surface's `resolveLiveCarrierFrame` delegates to
 * `resolveNeatRacingNetworkPanelLiveCarrierFrame`, which reads the
 * carried live network (own slot first, candidate view as fallback),
 * fails LOUDLY on a network-less carrier (never the retired idle-path
 * stub — AC-102) and delegates the whole frame to the shared visualizer
 * path (the L5 thin law). Activations are captured at resolution time,
 * so consumers re-resolve per frame.
 *
 * @example
 * ```ts
 * const surface = resolveNeatRacingAiModeVisualizerSurface({ mode: 'ai' });
 * // -> { mode: 'ai', visible: true, fullscreen: false, ... }
 * surface.toggle(); // -> fullscreen presentation state
 * const frame = surface.resolveLiveCarrierFrame(panelContext, winner);
 * drawNeatRacingNetworkPanelFromFrame(panelContext, frame);
 * const manualSurface = resolveNeatRacingAiModeVisualizerSurface('manual');
 * // -> null (manual mode keeps its own lane: UNCHANGED per AC-103)
 * ```
 */

import { resolveNeatRacingNetworkPanelLiveCarrierFrame } from './network-panel.adapter';
import type {
  NeatRacingNetworkPanelCandidate,
  NeatRacingNetworkPanelLiveCarrier,
  NeatRacingNetworkPanelShelfSlot,
  NetworkVisualizationResolvedFrame,
} from './network-panel.adapter';

/**
 * Named AI-mode visualizer resolution error prefix: invalid mode inputs
 * and invalid viewport dimensions fail LOUDLY with the named surface law
 * instead of silently degrading into a wrong presentation state.
 */
const NEATRACING_AI_MODE_VISUALIZER_ERROR_PREFIX =
  'NeatRacingAiModeVisualizerError';

/**
 * The AI-mode visualizer's presentation modes (AC-103): the 'ai' mode owns
 * the fullscreen-capable network surface; the 'manual' mode keeps its own
 * lane UNCHANGED (no visualizer surface is resolved for it).
 */
export type NeatRacingAiModeVisualizerMode = 'ai' | 'manual';

/**
 * Object form of the surface mode input: the host may hand the factory
 * whichever mode field its own state carries.
 */
export type NeatRacingAiModeVisualizerModeInput = {
  /** The presentation mode the surface is resolved for. */
  mode?: NeatRacingAiModeVisualizerMode | undefined;
  /** Alias for the presentation mode (the race-mode selector's field). */
  raceMode?: NeatRacingAiModeVisualizerMode | undefined;
  /** Boolean alias: true resolves the AI surface, false resolves manual. */
  isAiMode?: boolean | undefined;
};

/**
 * Acceptable surface factory inputs: the bare mode string, the object
 * form, or `undefined` (the recorded default resolves the AI-mode surface
 * — the AI lane is the mode the visualizer serves).
 */
export type NeatRacingAiModeVisualizerSurfaceInput =
  NeatRacingAiModeVisualizerMode | NeatRacingAiModeVisualizerModeInput;

/**
 * The AI-mode visualizer surface record (AC-103): a plain, DOM-free
 * presentation-state holder that the browser entry wires onto the panel
 * canvas.
 */
export type NeatRacingAiModeVisualizerSurface = {
  /** The owning mode record ('ai' — the surface never represents manual). */
  mode: 'ai';
  /**
   * VISIBLE by default (the CLAR-4-02 recorded default state: the AI-mode
   * surface inherits the docked shelf presentation on mount).
   */
  visible: boolean;
  /**
   * The fullscreen presentation state: false rides the docked shelf slot
   * (P4-S2C 1a), true rides the surface's fullscreen slot law.
   */
  fullscreen: boolean;
  /**
   * Toggles the fullscreen presentation state in place (no allocation; the
   * SAME record reads the flipped state after the call).
   */
  toggle: () => void;
  /**
   * The AI lane's live-carrier draw seam: delegates the elected winner to
   * the adapter's live-carrier frame law over the shared network
   * visualization (never the retired idle-path stub — AC-102).
   *
   * @param context - Host-owned canvas 2D drawing context.
   * @param carrier - The live carrier (or the elected candidate view) whose
   *   live evolved network resolves the frame.
   * @returns The resolved network-visualization frame to draw.
   * @throws Error when the carrier carries no live network (the named
   *   live-carrier contract error — the retired idle stub never returns).
   */
  resolveLiveCarrierFrame: (
    context: CanvasRenderingContext2D,
    carrier:
      NeatRacingNetworkPanelLiveCarrier | NeatRacingNetworkPanelCandidate,
  ) => NetworkVisualizationResolvedFrame;
};

/**
 * Resolves the presentation mode off the factory input (the recorded
 * default is 'ai').
 *
 * @param input - Bare mode string, object form, or `undefined`.
 * @returns The validated presentation mode.
 * @throws Error when the input carries an unknown mode value.
 */
function resolveNeatRacingAiModeVisualizerMode(
  input?: NeatRacingAiModeVisualizerSurfaceInput,
): NeatRacingAiModeVisualizerMode {
  // Step 1: The bare-string form is validated against the mode union.
  if (typeof input === 'string') {
    if (input !== 'ai' && input !== 'manual') {
      throw new Error(
        `${NEATRACING_AI_MODE_VISUALIZER_ERROR_PREFIX}: unknown AI-mode visualizer mode "${input}" — expected 'ai' or 'manual'`,
      );
    }
    return input;
  }
  // Step 2: The object form reads mode, raceMode, then the isAiMode alias.
  if (input !== undefined) {
    if (input.mode !== undefined) {
      return resolveNeatRacingAiModeVisualizerMode(input.mode);
    }
    if (input.raceMode !== undefined) {
      return resolveNeatRacingAiModeVisualizerMode(input.raceMode);
    }
    if (input.isAiMode !== undefined) {
      return input.isAiMode ? 'ai' : 'manual';
    }
  }
  // Step 3: The recorded default (AC-103): the visualizer serves the AI
  //         lane, so an absent mode resolves the AI-mode surface.
  return 'ai';
}

/**
 * Resolves the fullscreen-capable AI-mode visualizer surface (AC-103,
 * RED-P4-10 — the factory the red capability probe discovers FIRST).
 *
 * Laws the surface carries:
 *
 * - VISIBLE by default (the CLAR-4-02 recorded default state): `visible`
 *   starts as `true` so the surface inherits the docked shelf presentation
 *   on mount — never fullscreen-by-default (the recorded rejection of the
 *   CLAR-4-02 scope swallow).
 * - TOGGLABLE: `toggle()` flips the `fullscreen` presentation state in
 *   place on the SAME record (no allocation, no DOM); the docked and
 *   fullscreen states are the only two presentation states.
 * - MANUAL UNCHANGED (AC-103): the 'manual' mode resolves NO surface —
 *   `null` — so the manual lane keeps its own behavior untouched.
 * - LIVE-CARRIER draw seam (AC-102 + AC-034): the surface's frame
 *   resolution delegates to the adapter's live-carrier law over the shared
 *   network visualization; a network-less carrier fails loudly (never the
 *   retired idle-path stub frame).
 *
 * The surface is DOM-FREE: it records presentation state only; the browser
 * entry applies the state onto the panel canvas through pure slot
 * resolvers (the docked-right shelf resolver for the docked state,
 * `resolveNeatRacingAiModeVisualizerFullscreenSlot` for the fullscreen
 * state).
 *
 * @param input - Bare mode string, object form, or `undefined` (defaults
 *   to the AI mode).
 * @returns The AI-mode surface record, or `null` for the manual mode.
 * @throws Error when the input carries an unknown mode value.
 *
 * @example
 * ```ts
 * const surface = resolveNeatRacingAiModeVisualizerSurface('ai');
 * if (surface !== null) {
 *   surface.toggle(); // -> fullscreen presentation state on the same record
 *   surface.toggle(); // -> back to the docked shelf presentation
 * }
 * ```
 */
export function resolveNeatRacingAiModeVisualizerSurface(
  input?: NeatRacingAiModeVisualizerSurfaceInput,
): NeatRacingAiModeVisualizerSurface | null {
  // Step 1: Resolve + validate the presentation mode (manual resolves NO
  //         surface — the manual lane stays unchanged, AC-103).
  const mode = resolveNeatRacingAiModeVisualizerMode(input);
  if (mode === 'manual') {
    return null;
  }
  // Step 2: Resolve the AI-mode surface at the recorded DEFAULT state: the
  //         visible docked shelf presentation (never fullscreen-by-default).
  //         The record is frozen-free but state-mutable BY DESIGN: the
  //         toggle mutates the SAME record in place (no allocation).
  const surface: NeatRacingAiModeVisualizerSurface = {
    mode: 'ai',
    visible: true,
    fullscreen: false,
    toggle: (): void => {
      surface.fullscreen = !surface.fullscreen;
    },
    resolveLiveCarrierFrame: (
      context: CanvasRenderingContext2D,
      carrier:
        NeatRacingNetworkPanelLiveCarrier | NeatRacingNetworkPanelCandidate,
    ): NetworkVisualizationResolvedFrame =>
      // Step 3: The live-carrier draw seam (L5 thin law): the adapter
      //         owns the carrier read, the loud network-less failure and
      //         the shared-visualization frame resolution. The surface
      //         adds NO second resolution path.
      resolveNeatRacingNetworkPanelLiveCarrierFrame(context, carrier),
  };
  return surface;
}

/**
 * Resolves the AI-mode surface's fullscreen presentation slot (AC-103).
 *
 * The fullscreen presentation rides the FULL viewport: the surface's
 * fullscreen slot law is the full-viewport box (the hud-output layer maps
 * viewport px 1:1 onto its box), so the toggled panel fills the reserved
 * layer instead of the docked shelf. This is the surface's OWN presentation
 * law — the docked-right shelf resolver (P4-S2C 1a) stays untouched and
 * remains the DOCKED presentation's ONE sizing authority.
 *
 * Determinism claim (reproducibility rank: deterministic on the same
 * runtime): the slot is a pure function of its two arguments — no window
 * reads, no host measure, no DOM — so the same arguments always resolve the
 * same slot regardless of the surrounding window state.
 *
 * @param viewportWidthPx - Viewport width in CSS px (must be finite
 *   positive; the fullscreen box fills the viewport).
 * @param viewportHeightPx - Viewport height in CSS px (must be finite
 *   positive).
 * @returns The full-viewport presentation slot to size and position the
 *   panel with.
 * @throws Error when either viewport dimension is not a finite positive
 *   number.
 *
 * @example
 * ```ts
 * const slot = resolveNeatRacingAiModeVisualizerFullscreenSlot(1689, 1325);
 * // -> { leftPx: 0, topPx: 0, widthPx: 1689, heightPx: 1325 }
 * panelCanvas.style.left = `${slot.leftPx}px`;
 * panelCanvas.width = slot.widthPx;
 * ```
 */
export function resolveNeatRacingAiModeVisualizerFullscreenSlot(
  viewportWidthPx: number,
  viewportHeightPx: number,
): NeatRacingNetworkPanelShelfSlot {
  // Step 1: Validate the viewport dimensions (a non-finite or non-positive
  //         viewport NEVER degrades into a wrong box — it fails loudly).
  if (
    !Number.isFinite(viewportWidthPx) ||
    viewportWidthPx <= 0 ||
    !Number.isFinite(viewportHeightPx) ||
    viewportHeightPx <= 0
  ) {
    throw new Error(
      `${NEATRACING_AI_MODE_VISUALIZER_ERROR_PREFIX}: the fullscreen presentation slot requires finite positive viewport dimensions — received ${viewportWidthPx}x${viewportHeightPx}`,
    );
  }
  // Step 2: The fullscreen box IS the viewport (floored to whole px so the
  //         box and the backing store agree on integer measures).
  return {
    leftPx: 0,
    topPx: 0,
    widthPx: Math.floor(viewportWidthPx),
    heightPx: Math.floor(viewportHeightPx),
  };
}
