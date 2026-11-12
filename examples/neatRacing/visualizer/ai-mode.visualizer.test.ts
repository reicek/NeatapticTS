/**
 * @file RED contracts for the AI-mode LIVE visualizer (Step 26).
 *
 * Slice `P4-S4-red` — Phase 4, Step 26 of
 * `plans/neatracing-voxel-racer.plans.md` (the GAP-4 close: the AI lane
 * stops drawing the idle-path stub frame and gains the FULLSCREEN-CAPABLE
 * live visualizer surface).
 *
 * Educational note — WHAT THIS FILE OWNS:
 * GAP-4 is the last unclosed gap of the NeatRacing demo: the docked-right
 * network panel draws the IDLE-PATH STUB frame (the observation-input /
 * control-output placeholder the shared visualizer builds when no network is
 * handed in), so the user watching the AI lane never sees the NGE staged
 * evolution it is supposed to show. Step 26 closes the gap with a LIVE
 * EVOLVED carrier feed (the most-forward agent's own network plus its live
 * per-node activations) plus a FULLSCREEN-CAPABLE AI-mode visualizer
 * surface over `examples/shared/network-visualization`.
 *
 * Educational note — THE TWO CONTRACTS:
 * - RED-P4-9 — THE LIVE EVOLVED CARRIER FRAME. With a live evolved carrier
 *   present, the rendered most-forward frame resolves the carrier's LIVE
 *   network (real node / connection / activation data) and NEVER the
 *   idle-path stub frame (AC-102; the idle honesty branch retires in the
 *   SAME step the live feed lands — no dual path, no deferred cleanup).
 * - RED-P4-10 — THE FULLSCREEN-CAPABLE AI-MODE SURFACE. The AI mode mounts
 *   the fullscreen-capable visualizer surface over the shared visualization
 *   (visible by default, togglable) and the manual mode is NEVER changed
 *   (AC-103, the CLAR-4-02 recorded default).
 *
 * Educational note — THE RED GATE (why the surface module is widened). At
 * HEAD `examples/neatRacing/visualizer/ai-mode.visualizer.ts` does NOT
 * exist, so a static import of it would fail the suite at COMPILE time
 * (TS 2307) — the WRONG red reason. The surface module is therefore loaded
 * through a WIDENED dynamic import inside the gate helpers, and the gate
 * answers with a NAMED error that names the MISSING BEHAVIOUR (the AI lane
 * still draws the idle stub frame on the docked-right shelf, so there is no
 * fullscreen-capable surface to watch the evolving network on), never a
 * bare "module not found" (the house pattern of
 * `examples/neatRacing/track/track.catalog.test.ts`).
 *
 * Educational note — WHY THE CONTRACTS ARE DISCOVERY-BASED. The LIVE FEED
 * seam and the SURFACE core are discovered BY CAPABILITY: the exported
 * resolver that answers a live carrier with the shared visualizer's frame,
 * and the exported factory that answers a race mode with a togglable
 * surface. The contracts therefore pin the BEHAVIOUR, not a spelling: the
 * implementing slices may name their resolver and their surface factory
 * freely as long as the behaviour holds. A wrong pick fails LOUDLY (the
 * assertions are written against observable facts), so the discovery can
 * never make a contract pass vacuously.
 *
 * Educational note — THE ENVIRONMENT. This file runs in the NODE
 * environment (there is deliberately no `@jest-environment jsdom` docblock
 * here): every contract is DOM-free, so the AI-mode surface must expose a
 * DOM-free core — the same W2-PURE law the debug-rung seam carries. The
 * canvas context is a plain stub object (no `document`, no `window`).
 *
 * Educational note — THE PINNED SURFACES the implementing slices must
 * satisfy (the RED home is byte-frozen; the implementing slices edit the
 * production code and NEVER these contracts):
 * - `examples/neatRacing/visualizer/network-panel.adapter.ts` must expose a
 *   LIVE-CARRIER frame resolution: an exported resolver that answers a live
 *   carrier (the carrier's own network plus its live per-node activations)
 *   with the shared visualizer's frame for that network. The idle-path stub
 *   frame is not a legal answer while a live carrier exists.
 * - `examples/neatRacing/visualizer/ai-mode.visualizer.ts` (NEW) must expose
 *   a DOM-free AI-mode surface core: an exported factory that answers a race
 *   mode with a surface that (a) reports whether it is visible, (b) reports
 *   whether it is in the fullscreen presentation, and (c) exposes a toggle
 *   that flips that presentation. In the `manual` mode the factory must
 *   resolve NO mounted AI surface (the manual lane is never changed).
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import Network from '../../../src/architecture/network';
import { NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT } from '../ai/nge.controller';
import { NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT } from '../ai/observation.assembler';
import * as networkPanelAdapter from './network-panel.adapter';

/* ------------------------------------------------------------------------- *
 * The pinned world of the RED contracts.
 * ------------------------------------------------------------------------- */

/**
 * The AI-mode visualizer surface module path (WIDENED on purpose — see the
 * red gate above).
 *
 * The literal does not exist yet at HEAD, so a STATIC import of it would
 * fail the suite at compile time (TS 2307) — the wrong red reason.
 */
const RED_AI_MODE_VISUALIZER_MODULE_PATH = './ai-mode.visualizer';

/** The agent id of the live evolved carrier the panel watches. */
const RED_LIVE_CARRIER_AGENT_ID = 2;

/** The along-track progress of the live evolved carrier (it leads). */
const RED_LIVE_CARRIER_PROGRESS = 120;

/** The live carrier controller's observation width (Tier A default). */
const RED_LIVE_NETWORK_INPUT_COUNT = 4;

/** The live carrier controller's control-axis width. */
const RED_LIVE_NETWORK_OUTPUT_COUNT = 2;

/** The hidden nodes the live carrier's topology grew (node splitting). */
const RED_LIVE_NETWORK_HIDDEN_COUNT = 3;

/** The deterministic seed of the live carrier's evolved topology. */
const RED_LIVE_NETWORK_SEED = 4242;

/**
 * The node count of the IDLE-PATH STUB frame.
 *
 * Educational note — WHY A CONSTANT AND NOT A RESOLVED FRAME. The idle
 * honesty branch RETIRES in the same step the live feed lands (AC-102), so
 * the stub frame may stop being resolvable at GREEN. The stub frame's shape
 * is fixed by the observation/control widths though: the shared visualizer
 * builds `inputSize` input nodes plus `outputSize` output nodes when no
 * network is handed in. The stub frame is therefore pinned by its COUNT
 * (the live carrier's frame must never carry it).
 */
const RED_IDLE_STUB_FRAME_NODE_COUNT =
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT +
  NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT;

/**
 * The field names a surface may carry its visibility state under.
 */
const RED_VISIBILITY_FIELD_NAMES: readonly string[] = [
  'visible',
  'isVisible',
  'shown',
  'isShown',
  'mounted',
  'isMounted',
  'surfaceVisible',
  'isSurfaceVisible',
  'visibility',
];

/**
 * The method names a surface may expose its visibility state under.
 */
const RED_VISIBILITY_GETTER_NAMES: readonly string[] = [
  'isVisible',
  'getVisible',
  'isMounted',
  'getMounted',
  'isShown',
  'getShown',
  'visible',
];

/**
 * The field names a surface may carry its fullscreen presentation under.
 */
const RED_FULLSCREEN_FIELD_NAMES: readonly string[] = [
  'fullscreen',
  'isFullscreen',
  'fullscreenActive',
  'isFullscreenActive',
  'fullscreenEnabled',
  'expanded',
  'isExpanded',
  'maximized',
  'isMaximized',
  'presentation',
  'presentationActive',
  'mode',
];

/**
 * The method names a surface may expose its fullscreen presentation under.
 */
const RED_FULLSCREEN_GETTER_NAMES: readonly string[] = [
  'isFullscreen',
  'getFullscreen',
  'isFullscreenActive',
  'getFullscreenActive',
  'isExpanded',
  'getExpanded',
  'isMaximized',
  'getMaximized',
  'getPresentation',
];

/**
 * The method names a surface may expose its presentation toggle under.
 */
const RED_TOGGLE_METHOD_NAMES: readonly string[] = [
  'toggle',
  'toggleFullscreen',
  'togglePresentation',
  'toggleVisible',
  'toggleExpanded',
  'flip',
  'expand',
  'collapse',
  'enterFullscreen',
  'exitFullscreen',
  'setFullscreen',
  'setVisible',
  'show',
  'hide',
];

/**
 * The name pattern a presentation toggle may carry when it is spelled
 * outside the named list above.
 */
const RED_TOGGLE_NAME_PATTERN =
  /toggle|flip|expand|collapse|present|fullscreen|full-screen|maximi|show|hide/i;

/**
 * The field names a surface may carry its race mode under.
 */
const RED_MODE_FIELD_NAMES: readonly string[] = ['mode', 'raceMode'];

/**
 * The field names a surface may nest its state object under.
 */
const RED_NESTED_STATE_FIELD_NAMES: readonly string[] = [
  'state',
  'surfaceState',
  'viewState',
  'ui',
];

/* ------------------------------------------------------------------------- *
 * The gated surfaces.
 * ------------------------------------------------------------------------- */

/** A plain record view of an unknown object. */
type RedRecord = Record<string, unknown>;

/**
 * The live evolved carrier the panel watches.
 *
 * Educational note — WHY THE CARRIER IS RICH. The panel seam owns the
 * spelling of its carrier, so the fixture carries the live data under the
 * plausible field names (`network`, `activations`, `nodeActivations`,
 * `agentId`, `carIndex`, `progress`) plus a plain candidate view. The
 * contract pins the BEHAVIOUR (the rendered frame carries the carrier's live
 * node / connection / activation data), not the spelling.
 */
interface RedLiveCarrier {
  /** The carrier's stable fleet slot. */
  readonly agentId: number;
  /** The carrier's car index (the transport's carrier-match key). */
  readonly carIndex: number;
  /** The carrier's along-track progress. */
  readonly progress: number;
  /** The carrier's LIVE evolved topology. */
  readonly network: Network;
  /** The carrier's LIVE per-node activations. */
  readonly activations: readonly number[];
  /** The carrier's live activations under the alternative field name. */
  readonly nodeActivations: readonly number[];
  /** The candidate view of the carrier (the host's fleet candidate). */
  readonly candidate: RedRecord;
}

/**
 * The shared visualizer's frame as the contracts read it.
 */
interface RedFrameView {
  /** The positioned node count of the frame. */
  readonly nodeCount: number;
  /** The runtime connection count of the frame. */
  readonly connectionCount: number;
  /** The sorted multiset of the frame's node activations. */
  readonly activations: readonly number[];
  /** The frame's architecture label. */
  readonly architectureLabel: string;
}

/**
 * The gated RED-P4-9 context.
 */
interface RedLiveFrameContext {
  /** The rendered frame of the live evolved carrier. */
  readonly frame: RedFrameView;
  /** The live carrier the frame was rendered from. */
  readonly carrier: RedLiveCarrier;
  /** The live carrier's own node count. */
  readonly liveNodeCount: number;
  /** The live carrier's own connection count. */
  readonly liveConnectionCount: number;
  /** The live carrier's own activation values (sorted multiset). */
  readonly liveActivations: readonly number[];
}

/**
 * The DOM-free AI-mode surface the contracts read.
 */
interface RedAiModeSurface {
  /** The surface's own record view. */
  readonly record: RedRecord;
  /** The surface module's exports (the secondary state target). */
  readonly moduleRecord: RedRecord;
  /** Reads the surface's visibility state (null when it exposes none). */
  readonly readVisible: () => boolean | null;
  /** Reads the surface's fullscreen presentation state (null when none). */
  readonly readFullscreen: () => boolean | null;
  /** The surface's presentation toggle (null when it exposes none). */
  readonly toggle: ((...args: unknown[]) => unknown) | null;
  /** The surface's race mode (null when it exposes none). */
  readonly mode: string | null;
}

/**
 * The factory that resolves a race mode into an AI-mode surface.
 */
interface RedAiModeSurfaceFactory {
  /** The factory's exported name (for the gate's error voice). */
  readonly name: string;
  /** The AI-mode surface the factory resolved for the AI lane. */
  readonly aiSurface: RedAiModeSurface;
  /** Resolves the manual lane's surface (null when it mounts none). */
  readonly resolveManual: () => RedAiModeSurface | null;
}

/**
 * The presentation flip the toggle performs.
 */
interface RedPresentationFlip {
  /** Which state the toggle flipped. */
  readonly kind: 'fullscreen' | 'visible';
  /** The state before the toggle. */
  readonly before: boolean;
  /** The state after the toggle. */
  readonly after: boolean;
  /** Reads the flipped state right now. */
  readonly readCurrent: () => boolean | null;
  /** Performs the toggle again. */
  readonly toggle: () => Promise<void>;
}

/**
 * The gated RED-P4-10 context.
 */
interface RedAiModeSurfaceContext {
  /** The AI-mode surface (the AI lane's mounted surface). */
  readonly surface: RedAiModeSurface;
  /** The manual-mode surface (null when the manual mode mounts none). */
  readonly manualSurface: RedAiModeSurface | null;
  /** The AI-mode surface's AS-MOUNTED visibility state. */
  readonly visible: boolean | null;
  /** The AI-mode surface's AS-MOUNTED fullscreen presentation state. */
  readonly fullscreen: boolean | null;
  /** The presentation flip the surface's toggle performs. */
  readonly flip: RedPresentationFlip;
}

/* ------------------------------------------------------------------------- *
 * The red gate helpers.
 * ------------------------------------------------------------------------- */

/**
 * Loads a module that does not exist yet, WITHOUT failing the suite.
 *
 * Educational note — the widened import. The specifier is held in a
 * VARIABLE on purpose: a literal specifier for a missing module is a
 * compile-time error (TS 2307), which is the WRONG red reason. The widened
 * string keeps the dynamic import compilable at HEAD and at GREEN.
 *
 * @param modulePath - The widened module specifier to load.
 * @returns The loaded module namespace, or undefined when it is absent.
 */
async function loadRedModule(
  modulePath: string,
): Promise<RedRecord | undefined> {
  let loaded: RedRecord | undefined;
  try {
    loaded = (await import(modulePath)) as RedRecord;
  } catch {
    loaded = undefined;
  }
  return loaded;
}

/**
 * Builds the DOM-free 2D canvas context stub the contracts render with.
 *
 * Educational note — no `document`, no `window`: the stub is a plain object
 * so this suite can run in the NODE environment.
 *
 * @returns The stub canvas context.
 */
function buildRedStubContext(): CanvasRenderingContext2D {
  return {
    canvas: { width: 900, height: 240 },
    measureText: (): { width: number } => ({ width: 0 }),
    font: '10px sans-serif',
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    textAlign: 'center',
    textBaseline: 'middle',
    globalAlpha: 1,
    save: (): void => undefined,
    restore: (): void => undefined,
    clearRect: (): void => undefined,
    beginPath: (): void => undefined,
    closePath: (): void => undefined,
    moveTo: (): void => undefined,
    lineTo: (): void => undefined,
    stroke: (): void => undefined,
    fill: (): void => undefined,
    fillRect: (): void => undefined,
    strokeRect: (): void => undefined,
    fillText: (): void => undefined,
    strokeText: (): void => undefined,
    rect: (): void => undefined,
    arc: (): void => undefined,
    translate: (): void => undefined,
    scale: (): void => undefined,
    rotate: (): void => undefined,
    clip: (): void => undefined,
    setLineDash: (): void => undefined,
    getLineDash: (): number[] => [],
    setTransform: (): void => undefined,
    drawImage: (): void => undefined,
  } as unknown as CanvasRenderingContext2D;
}

/**
 * Builds the deterministic live evolved carrier the panel watches.
 *
 * Educational note — the fixture is a REAL evolved topology: a seeded
 * controller whose hidden nodes were grown by node splitting (so the live
 * frame carries real node AND connection data) and whose nodes carry
 * distinct live activations (so the activation witness is not vacuous).
 *
 * @returns The live evolved carrier.
 */
function buildRedLiveCarrier(): RedLiveCarrier {
  const network = new Network(
    RED_LIVE_NETWORK_INPUT_COUNT,
    RED_LIVE_NETWORK_OUTPUT_COUNT,
    {
      seed: RED_LIVE_NETWORK_SEED,
      minHidden: RED_LIVE_NETWORK_HIDDEN_COUNT,
    },
  );
  const activations: number[] = [];
  network.nodes.forEach((node, nodeIndex) => {
    const activation = Math.round(((nodeIndex + 1) / 10) * 1e4) / 1e4;
    node.activation = activation;
    activations.push(activation);
  });
  return {
    agentId: RED_LIVE_CARRIER_AGENT_ID,
    carIndex: RED_LIVE_CARRIER_AGENT_ID,
    progress: RED_LIVE_CARRIER_PROGRESS,
    network,
    activations,
    nodeActivations: activations,
    candidate: {
      agentId: RED_LIVE_CARRIER_AGENT_ID,
      progress: RED_LIVE_CARRIER_PROGRESS,
      network,
    },
  };
}

/**
 * Reads the shared visualizer's frame out of a resolved value.
 *
 * @param value - The resolved value (the frame itself or a wrapper around it).
 * @returns The frame view, or null when the value carries no frame.
 */
function redReadFrameView(value: unknown): RedFrameView | null {
  if (typeof value !== 'object' || value === null) {
    return null;
  }
  const record = value as RedRecord;
  const positionedScene = record['positionedScene'];
  if (typeof positionedScene !== 'object' || positionedScene === null) {
    return null;
  }
  const positionedNodes = (positionedScene as RedRecord)['positionedNodes'];
  const runtimeConnections = record['runtimeConnections'];
  if (!Array.isArray(positionedNodes) || !Array.isArray(runtimeConnections)) {
    return null;
  }
  const activations = positionedNodes.map((positionedNode): number => {
    const holder = positionedNode as RedRecord;
    const nodeHolder = holder['node'];
    const nestedActivation =
      typeof nodeHolder === 'object' && nodeHolder !== null
        ? (nodeHolder as RedRecord)['activation']
        : undefined;
    const directActivation = holder['activation'];
    const activation =
      typeof nestedActivation === 'number'
        ? nestedActivation
        : directActivation;
    return typeof activation === 'number' && Number.isFinite(activation)
      ? Math.round(activation * 1e4) / 1e4
      : 0;
  });
  const architectureLabel = record['architectureLabel'];
  return {
    nodeCount: positionedNodes.length,
    connectionCount: runtimeConnections.length,
    activations: [...activations].sort((left, right) => left - right),
    architectureLabel:
      typeof architectureLabel === 'string' ? architectureLabel : '',
  };
}

/**
 * The argument shapes a live-carrier frame resolver may accept.
 *
 * Educational note — why the shapes. The panel seam owns the spelling of
 * its resolver AND of its carrier argument, so the probe tries the natural
 * call shapes (context first, carrier first, carrier alone, the candidate
 * view alone). The bare network is NEVER among them: a resolver that only
 * accepts a bare network is the OLD idle-path behaviour, and letting it
 * answer would make the contract pass without the live feed.
 */
const RED_LIVE_FRAME_ARGUMENT_SHAPES: ReadonlyArray<
  (context: CanvasRenderingContext2D, carrier: RedLiveCarrier) => unknown[]
> = [
  (context, carrier) => [context, carrier],
  (context, carrier) => [context, carrier, carrier.activations],
  (context, carrier) => [carrier, context],
  (context, carrier) => [carrier],
  (context, carrier) => [context, carrier.candidate],
  (context, carrier) => [carrier.candidate],
];

/**
 * Probes the network-panel adapter for the live-carrier frame resolution.
 *
 * Educational note — the capability probe. The adapter owns the resolver's
 * NAME, so every exported function is offered the live carrier and only a
 * resolved SHARED FRAME (real nodes AND real connections) is accepted. The
 * idle-path stub frame carries no connections at all, so it can never be
 * mistaken for the live feed.
 *
 * @param context - The stub canvas context.
 * @param carrier - The live evolved carrier.
 * @returns The live frame, or null when no resolver answers with one.
 */
function redProbeAdapterLiveFrame(
  context: CanvasRenderingContext2D,
  carrier: RedLiveCarrier,
): RedFrameView | null {
  const adapterExports = networkPanelAdapter as unknown as RedRecord;
  const candidates = Object.entries(adapterExports).filter(
    ([, value]) => typeof value === 'function',
  );
  for (const [, value] of candidates) {
    const resolver = value as (...args: unknown[]) => unknown;
    for (const shape of RED_LIVE_FRAME_ARGUMENT_SHAPES) {
      let produced: unknown;
      try {
        produced = resolver(...shape(context, carrier));
      } catch {
        continue;
      }
      const frame = redReadFrameView(produced);
      if (frame !== null && frame.nodeCount > 0 && frame.connectionCount > 0) {
        return frame;
      }
    }
  }
  return null;
}

/**
 * Gates the live evolved carrier frame.
 *
 * Educational note — the red gate. At HEAD the network-panel adapter only
 * resolves a frame from a bare network (the idle-path stub when the
 * network is absent) and the browser entry feeds it the sim worker's
 * SPAWN-MIND transport snapshot, so NO exported resolver answers a live
 * carrier with the carrier's own node / connection / activation data. The
 * gate then answers the NAMED missing-behaviour error.
 *
 * @param contractId - The RED contract the gate speaks for.
 * @returns The gated live-frame context.
 * @throws When no live-carrier frame resolution path exists.
 */
function gateRedLiveCarrierFrame(contractId: string): RedLiveFrameContext {
  const carrier = buildRedLiveCarrier();
  const context = buildRedStubContext();
  const frame = redProbeAdapterLiveFrame(context, carrier);
  if (frame === null) {
    throw new Error(
      `${contractId} red gate: the network panel has NO live-carrier frame resolution path — the AI lane still draws the sim worker's SPAWN-MIND transport snapshot or the idle-path stub frame, so the LIVE EVOLVED network (node / connection / activation data) of the watched carrier is never rendered`,
    );
  }
  return {
    frame,
    carrier,
    liveNodeCount: carrier.network.nodes.length,
    liveConnectionCount: carrier.network.connections.length,
    liveActivations: [...carrier.activations].sort(
      (left, right) => left - right,
    ),
  };
}

/**
 * Reads the state records a surface may nest its state under.
 *
 * @param record - The record view to inspect.
 * @returns The record itself plus its nested state records.
 */
function redStateTargets(record: RedRecord): RedRecord[] {
  const targets: RedRecord[] = [record];
  for (const nestedName of RED_NESTED_STATE_FIELD_NAMES) {
    const nested = record[nestedName];
    if (typeof nested === 'object' && nested !== null) {
      targets.push(nested as RedRecord);
    }
  }
  return targets;
}

/**
 * Reads a boolean state out of the given records.
 *
 * @param records - The records to read (in probe order).
 * @param fieldNames - The accepted field names, in probe order.
 * @param getterNames - The accepted getter names, in probe order.
 * @param stringMatcher - The matcher for a string-spelled state.
 * @returns The state, or null when no record exposes one.
 */
function redReadBooleanState(
  records: readonly RedRecord[],
  fieldNames: readonly string[],
  getterNames: readonly string[],
  stringMatcher: (value: string) => boolean | null,
): boolean | null {
  for (const record of records) {
    for (const fieldName of fieldNames) {
      const value = record[fieldName];
      if (typeof value === 'boolean') {
        return value;
      }
      if (typeof value === 'string') {
        const matched = stringMatcher(value);
        if (matched !== null) {
          return matched;
        }
      }
    }
  }
  for (const record of records) {
    for (const getterName of getterNames) {
      const candidate = record[getterName];
      if (typeof candidate !== 'function') {
        continue;
      }
      try {
        const value = (candidate as () => unknown).call(record);
        if (typeof value === 'boolean') {
          return value;
        }
        if (typeof value === 'string') {
          const matched = stringMatcher(value);
          if (matched !== null) {
            return matched;
          }
        }
      } catch {
        continue;
      }
    }
  }
  return null;
}

/**
 * Matches a string-spelled VISIBILITY state.
 *
 * @param value - The string state.
 * @returns The boolean state, or null when the string is not a state.
 */
function redMatchVisibilityString(value: string): boolean | null {
  if (/hidden|invisible|closed|dismiss|collapsed/i.test(value)) {
    return false;
  }
  if (/visible|shown|mounted|open|active|live/i.test(value)) {
    return true;
  }
  return null;
}

/**
 * Matches a string-spelled FULLSCREEN presentation state.
 *
 * @param value - The string state.
 * @returns The boolean state, or null when the string is not a state.
 */
function redMatchFullscreenString(value: string): boolean | null {
  if (/full|maximi|expand/i.test(value)) {
    return true;
  }
  if (/shelf|dock|panel|inline|compact|collapsed|inset/i.test(value)) {
    return false;
  }
  return null;
}

/**
 * Reads the visibility state a surface may expose.
 *
 * @param record - The record view to read.
 * @returns The visibility state, or null when it exposes none.
 */
function redReadVisibility(record: RedRecord): boolean | null {
  return redReadBooleanState(
    redStateTargets(record),
    RED_VISIBILITY_FIELD_NAMES,
    RED_VISIBILITY_GETTER_NAMES,
    redMatchVisibilityString,
  );
}

/**
 * Reads the fullscreen presentation state a surface may expose.
 *
 * Educational note — booleans AND presentations. A surface may carry the
 * fullscreen state as a boolean (`fullscreen`) or as a presentation name
 * (`presentation: 'fullscreen'`), so both spellings are read.
 *
 * @param record - The record view to read.
 * @returns The presentation state, or null when it exposes none.
 */
function redReadFullscreen(record: RedRecord): boolean | null {
  return redReadBooleanState(
    redStateTargets(record),
    RED_FULLSCREEN_FIELD_NAMES,
    RED_FULLSCREEN_GETTER_NAMES,
    redMatchFullscreenString,
  );
}

/**
 * Finds the presentation toggle a record exposes.
 *
 * @param record - The record view to read.
 * @returns The toggle, or null when the record exposes none.
 */
function redFindToggle(
  record: RedRecord,
): ((...args: unknown[]) => unknown) | null {
  for (const methodName of RED_TOGGLE_METHOD_NAMES) {
    const candidate = record[methodName];
    if (typeof candidate === 'function') {
      return candidate as (...args: unknown[]) => unknown;
    }
  }
  for (const [methodName, candidate] of Object.entries(record)) {
    if (
      typeof candidate === 'function' &&
      RED_TOGGLE_NAME_PATTERN.test(methodName)
    ) {
      return candidate as (...args: unknown[]) => unknown;
    }
  }
  return null;
}

/**
 * Reads the race mode a surface exposes.
 *
 * @param record - The surface's record view.
 * @returns The race mode, or null when the surface exposes none.
 */
function redReadSurfaceMode(record: RedRecord): string | null {
  for (const fieldName of RED_MODE_FIELD_NAMES) {
    const value = record[fieldName];
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  return null;
}

/**
 * Reads a surface out of a resolved value.
 *
 * @param produced - The resolved value (the surface itself).
 * @param moduleRecord - The surface module's exports.
 * @returns The surface view, or null when the value carries no surface.
 */
function redReadAiModeSurface(
  produced: unknown,
  moduleRecord: RedRecord,
): RedAiModeSurface | null {
  if (typeof produced !== 'object' || produced === null) {
    return null;
  }
  const record = produced as RedRecord;
  const moduleToggle = redFindToggle(moduleRecord);
  const toggle = redFindToggle(record) ?? moduleToggle;
  const readVisible = (): boolean | null =>
    redReadVisibility(record) ?? redReadVisibility(moduleRecord);
  const readFullscreen = (): boolean | null =>
    redReadFullscreen(record) ?? redReadFullscreen(moduleRecord);
  if (readVisible() === null && readFullscreen() === null && toggle === null) {
    return null;
  }
  return {
    record,
    moduleRecord,
    readVisible,
    readFullscreen,
    toggle,
    mode: redReadSurfaceMode(record),
  };
}

/**
 * The argument shapes a race-mode surface factory may accept.
 */
const RED_AI_MODE_ARGUMENT_SHAPES: ReadonlyArray<
  (mode: 'ai' | 'manual') => unknown
> = [
  (mode) => mode,
  (mode) => ({ mode }),
  (mode) => ({ raceMode: mode }),
  (mode) => ({ mode, raceMode: mode }),
  (mode) => ({ raceMode: mode, isAiMode: mode === 'ai' }),
  () => undefined,
];

/**
 * Discovers the DOM-free AI-mode surface factory inside the surface module.
 *
 * Educational note — the capability probe. The surface module owns the
 * factory's NAME and the argument shape it takes, so every exported
 * function is offered the AI race mode and only a surface that reports a
 * state (visibility / presentation) and exposes a toggle is accepted.
 *
 * @param moduleExports - The surface module's exported surface.
 * @returns The discovered factory, or null when there is none.
 */
function redFindAiModeSurfaceFactory(
  moduleExports: RedRecord,
): RedAiModeSurfaceFactory | null {
  const candidates = Object.entries(moduleExports).filter(
    ([, value]) => typeof value === 'function',
  );
  for (const [name, value] of candidates) {
    const factory = value as (argument: unknown) => unknown;
    for (const shape of RED_AI_MODE_ARGUMENT_SHAPES) {
      let produced: unknown;
      try {
        produced = factory(shape('ai'));
      } catch {
        continue;
      }
      const aiSurface = redReadAiModeSurface(produced, moduleExports);
      if (aiSurface === null || aiSurface.toggle === null) {
        continue;
      }
      return {
        name,
        aiSurface,
        resolveManual: (): RedAiModeSurface | null => {
          let resolved: unknown;
          try {
            resolved = factory(shape('manual'));
          } catch {
            return null;
          }
          return redReadAiModeSurface(resolved, moduleExports);
        },
      };
    }
  }
  return null;
}

/**
 * Reads the presentation flip a surface's toggle performs.
 *
 * Educational note — the tolerant flip probe. A toggle may mutate the
 * surface in place, hand back the next surface, or answer with the next
 * state; all three shapes are read, and the flip is accepted only when the
 * presentation (or the visibility) observably CHANGES.
 *
 * @param surface - The surface the contracts read.
 * @returns The flip, or null when the toggle flips no state.
 */
async function redReadPresentationFlip(
  surface: RedAiModeSurface,
): Promise<RedPresentationFlip | null> {
  if (surface.toggle === null) {
    return null;
  }
  let currentRecord = surface.record;
  const readFullscreen = (): boolean | null =>
    redReadFullscreen(currentRecord) ?? redReadFullscreen(surface.moduleRecord);
  const readVisible = (): boolean | null =>
    redReadVisibility(currentRecord) ?? redReadVisibility(surface.moduleRecord);
  const fullscreenBefore = readFullscreen();
  const visibleBefore = readVisible();
  const performToggle = async (): Promise<void> => {
    let result: unknown;
    try {
      result = await surface.toggle?.call(currentRecord);
    } catch {
      return;
    }
    if (typeof result === 'object' && result !== null) {
      const produced = result as RedRecord;
      if (
        redReadVisibility(produced) !== null ||
        redReadFullscreen(produced) !== null
      ) {
        currentRecord = produced;
      }
    }
  };
  await performToggle();
  const fullscreenAfter = readFullscreen();
  const visibleAfter = readVisible();
  if (
    fullscreenBefore !== null &&
    fullscreenAfter !== null &&
    fullscreenBefore !== fullscreenAfter
  ) {
    return {
      kind: 'fullscreen',
      before: fullscreenBefore,
      after: fullscreenAfter,
      readCurrent: readFullscreen,
      toggle: performToggle,
    };
  }
  if (
    visibleBefore !== null &&
    visibleAfter !== null &&
    visibleBefore !== visibleAfter
  ) {
    return {
      kind: 'visible',
      before: visibleBefore,
      after: visibleAfter,
      readCurrent: readVisible,
      toggle: performToggle,
    };
  }
  return null;
}

/**
 * Gates the DOM-free AI-mode visualizer surface.
 *
 * Educational note — the red gate. At HEAD the file
 * `examples/neatRacing/visualizer/ai-mode.visualizer.ts` does not exist, so
 * the widened import resolves nothing and the gate answers the NAMED
 * missing-behaviour error. At GREEN the import resolves and the pinned
 * surface core is read.
 *
 * @param contractId - The RED contract the gate speaks for.
 * @returns The gated AI-mode surface context.
 * @throws When the surface module, its surface core, or its toggle is absent.
 */
async function gateRedAiModeSurface(
  contractId: string,
): Promise<RedAiModeSurfaceContext> {
  const moduleExports = await loadRedModule(RED_AI_MODE_VISUALIZER_MODULE_PATH);
  if (moduleExports === undefined) {
    throw new Error(
      `${contractId} red gate: the AI-mode visualizer surface does not exist yet at examples/neatRacing/visualizer/ai-mode.visualizer.ts — the AI lane still draws the idle stub frame on the docked-right shelf, so there is no fullscreen-capable surface to watch the evolving network on`,
    );
  }
  const factory = redFindAiModeSurfaceFactory(moduleExports);
  if (factory === null) {
    throw new Error(
      `${contractId} red gate: the AI-mode visualizer module exposes no DOM-free surface core — the AI mode must mount a fullscreen-capable surface that reports its visibility, reports its fullscreen presentation, and exposes a toggle for it, and that core must answer without a browser environment`,
    );
  }
  const surface = factory.aiSurface;
  const visible = surface.readVisible();
  const fullscreen = surface.readFullscreen();
  const flip = await redReadPresentationFlip(surface);
  if (flip === null) {
    throw new Error(
      `${contractId} red gate: the AI-mode visualizer surface exposes no working presentation toggle — the fullscreen-capable surface must be TOGGLEABLE (the CLAR-4-02 recorded default), so its toggle must flip the fullscreen presentation (or the surface visibility)`,
    );
  }
  return {
    surface,
    manualSurface: factory.resolveManual(),
    visible,
    fullscreen,
    flip,
  };
}

/**
 * Resolves the frame the AI-mode surface renders for the live carrier.
 *
 * Educational note — the surface seam first. The surface may carry its own
 * frame resolver; when it does not, the surface rides the adapter's
 * live-carrier resolver (the shared visualizer frame it draws). Either way
 * the witness is the SHARED visualizer's frame for the live carrier.
 *
 * @param surface - The AI-mode surface.
 * @param context - The stub canvas context.
 * @param carrier - The live evolved carrier.
 * @returns The frame, or null when no seam resolves one.
 */
function redResolveSurfaceFrame(
  surface: RedAiModeSurface,
  context: CanvasRenderingContext2D,
  carrier: RedLiveCarrier,
): RedFrameView | null {
  const candidates = Object.entries(surface.record).filter(
    ([name, value]) =>
      typeof value === 'function' && /frame|resolve|render|draw/i.test(name),
  );
  for (const [, value] of candidates) {
    const resolver = value as (...args: unknown[]) => unknown;
    for (const args of [
      [context, carrier],
      [carrier],
      [context, carrier.network],
      [carrier.network],
    ]) {
      let produced: unknown;
      try {
        produced = resolver(...args);
      } catch {
        continue;
      }
      const frame = redReadFrameView(produced);
      if (frame !== null && frame.nodeCount > 0) {
        return frame;
      }
    }
  }
  return redProbeAdapterLiveFrame(context, carrier);
}

/* ------------------------------------------------------------------------- *
 * RED-P4-9 — the live evolved carrier frame.
 * ------------------------------------------------------------------------- */

describe('RED-P4-9 — the live evolved carrier frame', () => {
  it('renders the LIVE carrier node data (never the idle stub frame)', () => {
    // Arrange: the live evolved carrier behind the red gate (no live-carrier
    // frame resolution path exists yet).
    const context = gateRedLiveCarrierFrame('RED-P4-9');

    // Act: the rendered frame's positioned node count.

    // Assert: the rendered frame carries the LIVE carrier's own nodes — a
    // panel that drew the idle stub would carry the placeholder nodes
    // instead (AC-102).
    expect(context.frame.nodeCount).toBe(context.liveNodeCount);
  });

  it('renders the LIVE carrier connection data', () => {
    // Arrange: the live evolved carrier behind the red gate.
    const context = gateRedLiveCarrierFrame('RED-P4-9');

    // Act: the rendered frame's runtime connection count.

    // Assert: the rendered frame carries the LIVE carrier's own connections
    // — the idle stub frame carries none at all, so a live feed that
    // resolves no connections is not the live network (AC-102).
    expect(context.frame.connectionCount).toBe(context.liveConnectionCount);
  });

  it('renders the LIVE carrier activation data', () => {
    // Arrange: the live evolved carrier behind the red gate.
    const context = gateRedLiveCarrierFrame('RED-P4-9');

    // Act: the rendered frame's node activations.

    // Assert: the rendered frame carries the LIVE carrier's own activation
    // values (the shared visualizer reorders nodes into layers, so the
    // activations are compared as a sorted multiset) — the idle stub frame
    // carries zeros (AC-102).
    expect(context.frame.activations).toEqual(context.liveActivations);
  });

  it('never draws the idle-path stub frame while a live carrier exists', () => {
    // Arrange: the live evolved carrier behind the red gate.
    const context = gateRedLiveCarrierFrame('RED-P4-9');

    // Act: the rendered frame's node count against the idle stub frame's
    // node count (the observation inputs plus the control outputs the
    // shared visualizer builds when no network is handed in).

    // Assert: the rendered frame is NOT the idle-path stub frame — the
    // honesty branch retires in the SAME step the live feed lands, so no
    // dual path survives (AC-102).
    expect(context.frame.nodeCount).not.toBe(RED_IDLE_STUB_FRAME_NODE_COUNT);
  });
});

/* ------------------------------------------------------------------------- *
 * RED-P4-10 — the fullscreen-capable AI-mode visualizer surface.
 * ------------------------------------------------------------------------- */

describe('RED-P4-10 — the fullscreen-capable AI-mode visualizer surface', () => {
  it('mounts the AI-mode surface VISIBLE BY DEFAULT', async () => {
    // Arrange: the AI-mode surface behind the red gate (the surface module
    // does not exist yet).
    const context = await gateRedAiModeSurface('RED-P4-10');

    // Act: the surface's AS-MOUNTED visibility state.

    // Assert: the AI mode mounts the visualizer surface VISIBLE BY DEFAULT
    // (the CLAR-4-02 recorded default).
    expect(context.visible).toBe(true);
  });

  it('exposes a FULLSCREEN-CAPABLE presentation the user can toggle', async () => {
    // Arrange: the AI-mode surface behind the red gate.
    const context = await gateRedAiModeSurface('RED-P4-10');

    // Act: the surface's presentation state before and after its toggle.

    // Assert: the surface reports a fullscreen presentation AND its toggle
    // flips that presentation — a fullscreen-capable surface that cannot be
    // toggled is not fullscreen-capable (AC-103).
    expect(context.fullscreen).not.toBeNull();
    expect(context.flip.after).not.toBe(context.flip.before);
  });

  it('toggles the AI-mode surface back out of the fullscreen presentation', async () => {
    // Arrange: the AI-mode surface behind the red gate plus the toggle it
    // already performed once.
    const context = await gateRedAiModeSurface('RED-P4-10');

    // Act: the toggle a second time.

    // Assert: the second toggle restores the initial presentation — the
    // surface is TOGGLEABLE in both directions (AC-103).
    await context.flip.toggle();
    expect(context.flip.readCurrent()).toBe(context.flip.before);
  });

  it('never mounts the AI-mode surface in MANUAL mode', async () => {
    // Arrange: the AI-mode surface behind the red gate plus the manual
    // mode's own answer.
    const context = await gateRedAiModeSurface('RED-P4-10');

    // Act: the manual mode's surface answer.

    // Assert: the manual mode mounts NO AI-mode visualizer surface — the
    // manual lane is never changed (AC-103).
    const manualVisible = context.manualSurface?.readVisible() ?? null;
    expect(context.manualSurface === null || manualVisible === false).toBe(
      true,
    );
  });

  it('renders the LIVE carrier frame over the shared network visualization', async () => {
    // Arrange: the AI-mode surface behind the red gate plus the live
    // evolved carrier the surface renders.
    const context = await gateRedAiModeSurface('RED-P4-10');
    const carrier = buildRedLiveCarrier();
    const canvasContext = buildRedStubContext();

    // Act: the frame the surface renders for the live carrier.

    // Assert: the surface renders the SHARED visualizer's frame for the live
    // carrier — the AI-mode surface rides the shared network visualization
    // instead of re-implementing the network rendering (AC-103).
    const frame = redResolveSurfaceFrame(
      context.surface,
      canvasContext,
      carrier,
    );
    expect(frame?.nodeCount).toBe(carrier.network.nodes.length);
  });
});
