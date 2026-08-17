/**
 * Host-level type definitions for the Neatenstein neon raycasting demo.
 *
 * Consolidates every public and internal type that was previously scattered
 * across individual host and host/game modules into a single authoritative
 * source. Each module re-exports the types it owns so existing imports remain
 * backward-compatible.
 *
 * @module
 */

import type { NeatensteinTier } from '../constants';
import type {
  NeatensteinRenderFrame,
  NeatensteinRenderState,
} from '../renderer/frame';
import type { MlpEnemyPopulation } from '../harness/enemy-mlp';
import type { Snapshot } from '../harness/types';
import type {
  BoltState,
  EnemyBoltState,
  GameState,
  Vector2,
} from './game/types';

// ---------------------------------------------------------------------------
// Input binding types (previously in game/controls.ts)
// ---------------------------------------------------------------------------

/**
 * Look delta produced by mouse, keyboard, or touch bindings.
 *
 * Values are in radians and are signed so callers can add them directly to
 * camera yaw/pitch accumulators.
 */
export interface LookDelta {
  /** Horizontal rotation delta. */
  yawDelta: number;

  /** Vertical rotation delta. */
  pitchDelta: number;
}

/** Callback invoked whenever a look binding emits a new delta. */
export type LookCallback = (delta: LookDelta) => void;

/**
 * Callback invoked when the primary fire input is pressed.
 *
 * The central router decides how to latch/consume the event.
 */
export type FireCallback = () => void;

/**
 * Callback invoked when the dynamic light toggle input is pressed.
 *
 * The binding is responsible for the `keydown` edge only; the central router
 * decides whether the toggle is latched or consumed immediately.
 */
export type LightToggleCallback = () => void;

/** Callback invoked when touch look starts or ends tracking an active touch. */
export type TouchActiveCallback = (active: boolean) => void;

/**
 * Detaches a previously installed input binding.
 *
 * Detach functions returned by the controls module are idempotent.
 */
export type BindingDetach = () => void;

// ---------------------------------------------------------------------------
// Input router types (previously in input.ts)
// ---------------------------------------------------------------------------

/**
 * Snapshot of raw player input at a single point in time.
 *
 * The snapshot is intentionally clone-safe so it can be passed across the
 * worker boundary via `postMessage` without losing information.
 */
export interface InputSnapshot {
  /** Milliseconds since epoch when the snapshot was captured. */
  timestamp: number;

  /** Directional movement intent from movement keys. */
  movement: {
    /** True while the forward key is held. */
    forward: boolean;
    /** True while the backward key is held. */
    backward: boolean;
    /** True while the strafe-left key is held. */
    left: boolean;
    /** True while the strafe-right key is held. */
    right: boolean;
  };

  /**
   * Accumulated orientation change since the last snapshot, in radians.
   *
   * Mouse, touch, and keyboard look deltas are consumed when the snapshot is
   * read so the same physical movement is never applied twice.
   */
  look: {
    /** Horizontal rotation delta. */
    yawDelta: number;
    /** Vertical rotation delta. */
    pitchDelta: number;
  };

  /** Current touch drag-to-look state for mobile fallback. */
  touch: {
    /** True while at least one active touch is being tracked. */
    active: boolean;
    /** Horizontal yaw delta from the active touch drag. */
    yawDelta: number;
    /** Vertical pitch delta from the active touch drag. */
    pitchDelta: number;
  };

  /** Whether the pointer is currently locked to the attached target. */
  pointerLocked: boolean;

  /**
   * Primary fire input for the current snapshot.
   *
   * Mouse clicks and keyboard fire presses are latched until consumed so quick
   * taps between render frames are not lost.
   */
  fire: boolean;

  /**
   * Dash input for the current snapshot.
   *
   * Dash presses are latched until consumed so quick taps between render frames
   * are not lost.
   */
  dash: boolean;

  /**
   * Dynamic light toggle input for the current snapshot.
   *
   * Toggle presses are latched until consumed so quick taps between render
   * frames are not lost.
   */
  lightToggle: boolean;
}

/** Detaches all event listeners installed by the input router. */
export type InputRouterDetach = () => void;

/** Public surface of a host input router. */
export interface InputRouter {
  /**
   * Attach input listeners to a DOM target.
   *
   * @param target - Element that owns pointer lock and receives mouse/touch
   *   events. Keyboard events are read from `window`.
   * @returns A detach function for this attachment.
   * @throws {Error} If the router is already attached.
   */
  attach(target: HTMLElement): InputRouterDetach;

  /** Remove all installed listeners and reset transient input state. */
  detach(): void;

  /**
   * Capture and reset the current input snapshot.
   *
   * Calling this consumes pending mouse, touch, keyboard-look, fire, and dash
   * deltas/actions.
   *
   * @returns A clone-safe input snapshot.
   */
  getSnapshot(): InputSnapshot;
}

// ---------------------------------------------------------------------------
// Renderer bridge types (previously in renderer-bridge.ts)
// ---------------------------------------------------------------------------

/**
 * Configuration needed to create a host side renderer bridge that spawns the
 * display worker and transfers the OffscreenCanvas to the selected tier.
 */
export interface NeatensteinRendererBridgeOptions {
  /** The visible canvas element on the host page. */
  canvas: HTMLCanvasElement;
  /** Absolute or relative URL to the worker bundle. */
  workerUrl?: string;
  /** Selected renderer tier. */
  tier: NeatensteinTier;
  /** Deterministic seed used to build the wall grid on the worker. */
  mapSeed: number;
}

/**
 * Public surface of the host renderer bridge that callers use to forward
 * state, send input, and consume rendered frames from the display worker.
 */
export interface NeatensteinRendererBridge {
  /** The spawned display worker. */
  worker: Worker;
  /** Latest frame request id received from the worker. */
  requestId: number;
  /** Forward a simulation state snapshot to the worker. The bridge applies
   * worker-busy backpressure so only one snapshot is in flight at a time;
   * additional calls defer the latest state until the worker acknowledges. */
  postSimState(state: NeatensteinRenderState): void;
  /** Forward a host-resized CSS-box dimension to the worker. */
  resize(width: number, height: number): void;
  /** Forward an input snapshot to the worker so it can advance the sim tick. */
  forwardWorkerInput(snapshot: InputSnapshot): void;
  /** Terminate the worker and release the bridge. */
  destroy(): void;
  /**
   * Register a callback that receives every rendered frame produced by the
   * worker on the `cpu` and `gpu` tiers.
   *
   * The consumer is called synchronously when a `frame` message arrives, before
   * the bridge updates its latest request id. This lets the host overlay or
   * capture pipeline consume the same frame payload without an extra copy.
   *
   * @param consumer - Function invoked with each incoming render frame.
   */
  setFrameConsumer(consumer: (frame: NeatensteinRenderFrame) => void): void;
  /**
   * Register a callback invoked when the worker finishes rendering a frame
   * and has no pending state to flush. The host uses this to schedule the
   * next `requestAnimationFrame`, making the render loop purely worker-paced
   * instead of running continuously at display refresh rate.
   *
   * The callback is NOT called when a deferred `pendingState` is flushed on
   * the frame ack — the flushed state's own ack will trigger it once the
   * worker finishes that render.
   *
   * Pass `null` to clear the callback.
   *
   * @param callback - Function invoked when the worker is idle and ready for
   *   the next frame, or `null` to clear.
   */
  setOnFrameReady(callback: (() => void) | null): void;
}

// ---------------------------------------------------------------------------
// Resize types (previously in resize.ts)
// ---------------------------------------------------------------------------

/** Result of a host resize operation. */
export interface NeatensteinResizeResult {
  /**
   * Number of backing-store pixels per renderer column.
   *
   * For the direct worker tier this should normally be `1`, because the worker
   * renders one ray column per backing-store pixel.
   */
  columnStride: number;

  /** Fresh render frame sized to the resolved column count. */
  frame: NeatensteinRenderFrame;
}

// ---------------------------------------------------------------------------
// Wave transition types (previously in waves.ts)
// ---------------------------------------------------------------------------

/**
 * Options accepted by {@link advanceWave}, pairing the live enemy population
 * with an optional cap on how many enemies the next wave should spawn.
 */
export interface AdvanceWaveOptions {
  /** Live enemy population whose champion snapshot is advanced one generation. */
  population: MlpEnemyPopulation;
  /**
   * Number of enemies to spawn for the new wave.
   *
   * Defaults to the max concurrent enemy cap. Values below zero are
   * treated as zero; non-finite values fall back to the default cap; values
   * above the concurrency cap are clamped to the cap.
   */
  spawnCount?: number;
}

/** Result returned by {@link advanceWave}. */
export interface AdvanceWaveResult {
  /** New game snapshot with a cleared arena and freshly spawned enemies. */
  state: GameState;
  /** Champion snapshot produced by advancing the enemy population. */
  snapshot: Snapshot;
  /** Number of enemies actually spawned (always in [0, cap]). */
  spawnedCount: number;
}

// ---------------------------------------------------------------------------
// Episode types (previously in game/episode.ts)
// ---------------------------------------------------------------------------

/** Options accepted by {@link createEpisode}. */
export interface CreateEpisodeOptions {
  /** Deterministic seed used to initialize the episode. */
  seed?: number;

  /** Target episode duration in milliseconds. */
  durationMs?: number;
}

/** In-memory handle for a single episode run. */
export interface Episode {
  /** Mutable snapshot of the running episode state. */
  state: GameState;

  /** Target duration in milliseconds used to decide episode completion. */
  durationMs: number;
}

// ---------------------------------------------------------------------------
// Cadence types (previously in game/cadence.ts)
// ---------------------------------------------------------------------------

/** Inputs accepted by cadence estimation helpers. */
export interface EstimateCadenceOptions {
  /**
   * Average episode duration in milliseconds.
   *
   * Invalid, infinite, or negative values make the estimate invalid and return
   * a cadence of `0`.
   */
  episodeDurationMs: number;

  /**
   * Per-episode evaluator overhead in milliseconds.
   *
   * This includes inference, fitness calculation, bookkeeping, and any other
   * fixed per-generation work. Invalid, infinite, or negative values make the
   * estimate invalid and return a cadence of `0`.
   */
  evaluationOverheadMs: number;
}

// ---------------------------------------------------------------------------
// Combat types (previously in game/combat.ts)
// ---------------------------------------------------------------------------

/** Result of attempting to fire the traveling plasma bolt. */
export interface FireBoltResult {
  /** Snapshot after the shot: ammo consumed, bolt appended, damage applied. */
  state: GameState;

  /** `true` when a shot was actually fired this frame. */
  fired: boolean;

  /** Spawned bolt for this frame, or `null` when the weapon did not fire. */
  bolt: BoltState | null;
}

/**
 * Input shape for spawning an enemy bolt from a hitscan event.
 *
 * Mirrors the relevant fields of a hitscan event without importing the
 * controller module, keeping the combat module dependency-free.
 */
export interface FireEnemyBoltInput {
  /** World-space origin of the enemy's hitscan ray. */
  origin: Vector2;
  /** Normalized direction toward the player at fire time. */
  direction: Vector2;
  /** Damage applied on hit. */
  damage?: number;
}

// ---------------------------------------------------------------------------
// Collision types (previously in game/collision.ts)
// ---------------------------------------------------------------------------

/** Position-like shape used by contact-distance helpers. */
export interface ContactPosition {
  /** World X coordinate in grid cells. */
  x: number;
  /** World Y coordinate in grid cells. */
  y: number;
}

// ---------------------------------------------------------------------------
// Tick input types (previously in game/tick.input.utils.ts)
// ---------------------------------------------------------------------------

/**
 * Normalized input snapshot consumed by the game tick.
 *
 * The shape is deliberately clone-safe so the same snapshot can be forwarded
 * from the host input router, across the worker boundary, and replayed for
 * deterministic simulation evaluation.
 */
export interface GameTickInputSnapshot {
  /** Player-local movement vector; `x=-1` left, `x=1` right, `y=1` forward. */
  move: Vector2;

  /** Horizontal look delta to apply this tick, in radians. */
  lookDelta: number;

  /** `true` when the fire action is requested or held this tick. */
  fire: boolean;

  /** `true` when the dash action is requested this tick. */
  dash: boolean;
}

/** Fully normalized input used internally by the tick pipeline. */
export interface NormalizedGameTickInputSnapshot {
  /** Sanitized player-local movement vector. */
  move: Vector2;

  /** Finite horizontal look delta in radians. */
  lookDelta: number;

  /** Whether fire is active this tick. */
  fire: boolean;

  /** Whether dash is active this tick. */
  dash: boolean;
}

// ---------------------------------------------------------------------------
// Tick result types (previously in various tick.*.utils.ts)
// ---------------------------------------------------------------------------

/** Result of advancing enemy bolts for one tick. */
export interface UpdateEnemyBoltsResult {
  /** Updated game state (damage applied if any bolt hit the player). */
  state: GameState;
  /** Updated enemy bolt array (inactive bolts still present for culling). */
  bolts: EnemyBoltState[];
}

/** Result of applying bolt-impact results to the game state. */
export interface BoltImpactResult {
  /** Updated game state with enemy damage and impact spots applied. */
  state: GameState;
  /** `true` when at least one bolt hit an enemy this tick. */
  boltHitEnemy: boolean;
}

/** Result of firing a bolt and applying gun recoil. */
export interface FireRecoilResult {
  /** Updated game state after firing. */
  state: GameState;
  /** `true` when a bolt was actually fired. */
  fired: boolean;
}

/** Result of a single spawn tick. */
export interface SpawnWaveTickResult {
  /** Number of enemies added this tick (always 0 or 1). */
  spawnedThisTick: number;
  /** New state snapshot with any spawned enemy included. */
  state: GameState;
}

// ---------------------------------------------------------------------------
// Mugshot types (previously in hud-mugshot.ts)
// ---------------------------------------------------------------------------

/** Head-crop result: width, height, and RGBA pixel data. */
export interface MugshotHeadCrop {
  /** Crop width in physical pixels. */
  width: number;
  /** Crop height in physical pixels. */
  height: number;
  /** RGBA pixel data buffer. */
  data: Uint8ClampedArray;
}

/**
 * Mouse-look state used to select the mugshot direction.
 *
 * `yawDelta` is the per-frame mouse-look yaw delta consumed from
 * `InputSnapshot.look.yawDelta`. A negative delta (turning left) yields the
 * `frontLeft` view; a positive delta (turning right) yields `frontRight`; a
 * zero delta (mouse still) yields the neutral `front` view.
 */
export interface MugshotLook {
  /** Per-frame mouse-look yaw delta. */
  yawDelta: number;
}

/** Supported mugshot directions. */
export type MugshotDirection = 'front' | 'frontLeft' | 'frontRight';

/** Mugshot canvas overlay instance. */
export interface MugshotOverlay {
  /** Canvas element rendering the mugshot head crop. */
  canvas: HTMLCanvasElement;
  /** Redraw the mugshot with a new direction and health ratio. */
  update: (direction: MugshotDirection, healthRatio?: number) => void;
}

// ---------------------------------------------------------------------------
// HUD types (previously in hud.ts, hud.*.utils.ts, hud-mugshot.ts)
// ---------------------------------------------------------------------------

/**
 * Available human-mode selector values.
 *
 * - `'auto'` — the demo runs in autonomous NEAT-driven mode.
 * - `'human'` — the demo switches to human-play mode with replay-buffer
 *   selection pressure.
 */
export type HumanMode = 'auto' | 'human';

/** Human-mode selector instance. */
export interface HumanModeSelector {
  /** Resolved host container element that owns the selector. */
  container: HTMLElement;
  /** The `<select>` element with auto/human options. */
  select: HTMLSelectElement;
  /** Current mode ('auto' or 'human'). */
  mode: HumanMode;
  /** Programmatically set the mode and notify registered callbacks. */
  setMode: (mode: HumanMode) => void;
  /** Register a callback invoked whenever the mode changes. */
  onToggle: (callback: (mode: HumanMode) => void) => void;
}

/** Render-state snapshot accepted by the HIVE density HUD update callback. */
export interface HiveDensityHudState {
  /** Current enemy population density, expected in [0, 1]. */
  hiveDensity: number;
}

/** HIVE density HUD instance. */
export interface HiveDensityHud {
  /** Resolved host container element that owns the HUD. */
  container: HTMLElement;
  /** Meter track element (fixed-size outer bar). */
  meter: HTMLElement;
  /** Fill bar element (width + color driven by density). */
  fill: HTMLElement;
  /** Label element showing the meter title and percentage. */
  label: HTMLElement;
  /** Refresh the HUD from a render-state snapshot. */
  update: (state: HiveDensityHudState) => void;
}

/** Adaptation signal accepted by the death feedback indicator. */
export interface DeathFeedbackSignal {
  /** Coarse direction label: `'stronger'`, `'weaker'`, or `'shifted'`. */
  direction: 'stronger' | 'weaker' | 'shifted';
  /** Change in enemy aggression between generations. */
  aggressionDelta: number;
  /** Change in enemy movement pattern between generations. */
  movementDelta: number;
  /** Change in enemy positioning between generations. */
  positioningDelta: number;
}

/** Death feedback indicator instance. */
export interface DeathFeedbackIndicator {
  /** Resolved host container element that owns the indicator. */
  container: HTMLElement;
  /** Indicator wrapper element appended to the host container. */
  indicator: HTMLElement;
  /** Label element showing the adaptation direction text. */
  label: HTMLElement;
  /** Refresh the indicator from an adaptation signal. */
  update: (signal: DeathFeedbackSignal) => void;
}

/** Render-state snapshot accepted by the health/ammo HUD update callback. */
export interface HealthAmmoHudState {
  /** Current player health. */
  health: number;
  /** Maximum player health. */
  maxHealth: number;
  /** Current player ammo. */
  ammo: number;
  /** Maximum player ammo. */
  maxAmmo: number;
}

/** Health/ammo HUD instance. */
export interface HealthAmmoHud {
  /** Resolved host container element that owns the HUD. */
  container: HTMLElement;
  /** Health meter track element (fixed-size outer bar). */
  healthTrack: HTMLElement;
  /** Health fill bar element (width + color driven by health fraction). */
  healthFill: HTMLElement;
  /** Label element showing the health percentage. */
  healthLabel: HTMLElement;
  /** Label element showing the ammo count. */
  ammoLabel: HTMLElement;
  /** Refresh the HUD from a render-state snapshot. */
  update: (state: HealthAmmoHudState) => void;
}

/**
 * Render-state snapshot accepted by the neon status bar update callback.
 *
 * The health/ammo fields are always required. `hiveDensity`, `playerKills`,
 * and `playerDeaths` are optional and default to 0 when omitted.
 */
export interface NeonStatusBarState {
  /** Current enemy population density, expected in [0, 1]. */
  hiveDensity?: number;
  /** Current player health. */
  playerHealth: number;
  /** Maximum player health. */
  playerMaxHealth: number;
  /** Current player ammo. */
  playerAmmo: number;
  /** Maximum player ammo. */
  playerMaxAmmo: number;
  /** Current player kill count (fallback 0). */
  playerKills?: number;
  /** Current player death count (fallback 0). */
  playerDeaths?: number;
  /** Current evolutionary generation (fallback 0). */
  generation?: number;
}

/** Neon status bar HUD instance. */
export interface NeonStatusBarHud {
  /** Resolved host container element that owns the status bar. */
  container: HTMLElement;
  /** Absolutely-positioned bottom overlay bar element. */
  bar: HTMLElement;
  /** Array of health segment elements lit by health fraction. */
  healthSegments: HTMLElement[];
  /** Array of ammo segment elements lit by ammo fraction. */
  ammoSegments: HTMLElement[];
  /** HIVE density fill bar element (width driven by density). */
  hiveFill: HTMLElement;
  /** Label element showing the player kill count. */
  killsLabel: HTMLElement;
  /** Label element showing the player death count. */
  deathsLabel: HTMLElement;
  /** Label element showing the evolutionary generation. */
  generationLabel: HTMLElement;
  /** Mugshot canvas overlay rendering the robot head crop with damage tint. */
  mugshot: MugshotOverlay;
  /** Refresh the status bar from a render-state snapshot. */
  update: (state: NeonStatusBarState) => void;
}

/** Wave announcement overlay state. */
export interface WaveAnnouncementHud {
  /** Resolved host container element. */
  container: HTMLElement;
  /** The overlay div element. */
  overlay: HTMLDivElement;
  /** Show the "Wave N" announcement with fade-in/hold/fade-out animation. */
  show: (waveNumber: number) => void;
}
