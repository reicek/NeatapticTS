/**
 * Packed structure-of-arrays frame produced by the simulation worker and
 * consumed by the display thread.
 *
 * All typed arrays are row-major with `agentCount` rows (one row per car).
 * The `tireState` array is `agentCount * 4` elements ordered as
 * `[FL, FR, RL, RR]` per car, with each channel clamped to `[0, 1]`.
 * The `radioField` array has 0 elements in Tier 0 (radio disabled).
 *
 * Tier 4+ packs (4 or more cars) also include a compact `pitStatus` typed
 * array laid out as `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`.
 * The value `255` in a car slot means that team's pit is currently empty;
 * the tick slot counts down from `PIT_STOP_TICKS` (4) to zero, at which
 * point the car is released and its tires are restored to full health.
 * The environment-level `PitOccupancyState` shelf uses the full six-slot
 * layout (three per team); the packed frame projects that into this compact
 * form for zero-copy transfer.
 *
 * Zero-copy transfer contract:
 * - Every `ArrayBuffer` backing a typed-array field must appear exactly once in
 *   the postMessage transfer list.
 * - A buffer that appears in the transfer list is detached after transfer;
 *   the producer must not reuse it.
 * - Consumers must reject frames whose `schemaVersion` differs from
 *   `'racing-packed-v1'`.
 */
export type RacingRenderFrame = {
  /** Schema sentinel for forward-compatibility rejection. Must be `'racing-packed-v1'`. */
  schemaVersion: 'racing-packed-v1';
  /** Monotonic fixed-timestep tick index. */
  tick: number;
  /** PRNG seed used to generate the current episode. */
  seed: number;
  /** Identifies the frozen TrackSpec this frame belongs to. */
  trackId: number;
  /** Number of agent slots (fixed for the episode). */
  agentCount: number;
  /** Feature-flag bitfield: bit 0 = tiresEnabled, bit 1 = radioEnabled, bit 2 = pitsEnabled. */
  featureFlags: number;

  // --- Per-car position and state (one element per agent) ---

  /** Car X positions in logical world units. Length = agentCount. */
  carX: Float32Array;
  /** Car Y positions in logical world units. Length = agentCount. */
  carY: Float32Array;
  /** Car headings in radians. Length = agentCount. */
  carHeading: Float32Array;
  /** Active flags: 1 = in race, 0 = inactive slot. Length = agentCount. */
  carActive: Uint8Array;
  /** Team index: 0 = Team A, 1 = Team B. Length = agentCount. */
  carTeam: Uint8Array;
  /** GatingRouter mode index (0 when routing not yet present). Length = agentCount. */
  carMode: Uint8Array;
  /** Signed forward speed in world units per second. Length = agentCount. */
  forwardSpeedWorld?: Float32Array;
  /** Signed lateral speed in world units per second. Length = agentCount. */
  lateralSpeedWorld?: Float32Array;
  /** Unsigned world speed in world units per second. Length = agentCount. */
  speedWorld?: Float32Array;

  // --- Tire state row-major [agentCount × 4] ---

  /** Tire state per corner (FL, FR, RL, RR) in [0, 1]. Length = agentCount * 4. */
  tireState: Float32Array;

  // --- Team radio field [agentCount × radioDim]; empty (length=0) in Tier 0 ---

  /** Radio field typed array. Length = 0 when radio is disabled (Tier 0). */
  radioField: Float32Array;
  /** Optional Tier 4 pit-status tuple `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`; car slots use `255` for `no car`, and tick slots count down the fixed pit stop. */
  pitStatus?: Uint8Array | Uint16Array | Int16Array;

  // --- Episode scalars per car ---

  /** Current lap index per car. Length = agentCount. */
  lap: Uint16Array;
  /** Current race position per car. Length = agentCount. */
  place: Uint8Array;

  // --- Episode-level scalars ---

  /** Elapsed race time in milliseconds. */
  raceTimeMs: number;
  /** True when the episode has ended. */
  done: boolean;
};
