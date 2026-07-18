/**
 * Tier 5 simulation-worker helpers for the smallest honest 3v3 six-car race pack.
 *
 * This module keeps the six-car seam narrow: one factory allocates the canonical
 * packed frame, and one resolver answers which teammate radio rows are readable
 * for a given car.
 */
import type { RacingRenderFrame } from './simulation-worker.types';

/** Number of teams in the Tier 5 race pack. */
const TIER_FIVE_TEAM_COUNT = 2;
/** Number of cars assigned to each team in the 3v3 slice. */
const TIER_FIVE_TEAM_SIZE = 3;
/** Number of agents in a Tier 5 3v3 race (`2 × 3 = 6`). */
const TIER_FIVE_AGENT_COUNT = TIER_FIVE_TEAM_COUNT * TIER_FIVE_TEAM_SIZE;
/** Number of tire channels packed per car (`frontLeft`, `frontRight`, `rearLeft`, `rearRight`). */
const TIRE_CHANNEL_COUNT_PER_CAR = 4;
/** Total tire-state size for the six-car pack (`6 × 4 = 24`). */
const TIER_FIVE_TIRE_STATE_SIZE =
  TIER_FIVE_AGENT_COUNT * TIRE_CHANNEL_COUNT_PER_CAR;
/** Number of radio channels packed per car. */
const RADIO_CHANNEL_COUNT_PER_CAR = 7;
/** Total packed radio-field size for the Tier 5 3v3 race (`6 × 7 = 42`). */
const TIER_FIVE_RADIO_FIELD_SIZE =
  TIER_FIVE_AGENT_COUNT * RADIO_CHANNEL_COUNT_PER_CAR;
/** Canonical Tier 5 team layout `[0, 0, 0, 1, 1, 1]` = `[A0, A1, A2, B0, B1, B2]`. */
const TIER_FIVE_TEAM_LAYOUT = [0, 0, 0, 1, 1, 1] as const;
/**
 * Packed pit-status width
 * `[teamA_carIndex, teamA_ticks, teamA_waitingCarIndex, teamB_carIndex, teamB_ticks, teamB_waitingCarIndex]`.
 */
const PIT_STATUS_CHANNEL_COUNT = 6;
/** Sentinel value meaning a team's pit slot is empty in the packed `pitStatus` tuple. */
const NO_CAR_INDEX = 255;

/**
 * Creates the canonical packed Tier 5 race frame for a 3v3 evaluation slice.
 *
 * This helper pins the six-car transport contract used by the Tier 5 worker seam:
 *
 * - `agentCount = 6` because the pack always represents two teams of three cars.
 * - `carTeam = [0, 0, 0, 1, 1, 1]`, so cars `0..2` are Team A and cars `3..5` are Team B.
 * - `radioField.length = 42` because six cars each publish one seven-channel row.
 * - `tireState.length = 24` because six cars each store four wheel channels.
 * - `pitStatus.length = 6` because the pit shelf includes wait slots for both teams.
 *
 * `NO_CAR_INDEX` marks an empty team pit slot inside that packed `pitStatus` tuple.
 *
 * @returns Six-car packed render frame with tire, radio, and pit transport enabled.
 * @example
 * ```ts
 * const expectedAgentCount = TIER_FIVE_AGENT_COUNT;
 * const expectedRadioWidth = TIER_FIVE_RADIO_FIELD_SIZE;
 * const expectedTireWidth = TIER_FIVE_TIRE_STATE_SIZE;
 * const frame = createTier5RacePack();
 *
 * frame.agentCount === expectedAgentCount; // true
 * frame.radioField.length === expectedRadioWidth; // true
 * frame.tireState.length === expectedTireWidth; // true
 * frame.pitStatus.length === PIT_STATUS_CHANNEL_COUNT; // true
 * ```
 */
export function createTier5RacePack(): RacingRenderFrame & {
  pitStatus: Int16Array;
  focusCarIndex: number;
  forwardSpeedWorld: Float32Array;
  lateralSpeedWorld: Float32Array;
  speedWorld: Float32Array;
} {
  const pitStatus = new Int16Array(PIT_STATUS_CHANNEL_COUNT);
  pitStatus[0] = NO_CAR_INDEX;
  pitStatus[2] = NO_CAR_INDEX;
  pitStatus[3] = NO_CAR_INDEX;
  pitStatus[5] = NO_CAR_INDEX;

  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed: 0,
    trackId: 0,
    agentCount: TIER_FIVE_AGENT_COUNT,
    featureFlags: 0b111,
    // Step 1: Allocate one pose/activity lane per car in the six-car roster.
    carX: new Float32Array(TIER_FIVE_AGENT_COUNT),
    carY: new Float32Array(TIER_FIVE_AGENT_COUNT),
    carHeading: new Float32Array(TIER_FIVE_AGENT_COUNT),
    carActive: new Uint8Array(TIER_FIVE_AGENT_COUNT).fill(1),
    // Step 2: Pin team ownership to `[A0, A1, A2, B0, B1, B2]` for every consumer.
    carTeam: Uint8Array.from(TIER_FIVE_TEAM_LAYOUT),
    carMode: new Uint8Array(TIER_FIVE_AGENT_COUNT),
    // Step 2b: Allocate speed lanes required by Tier 6 opponent-perception slots.
    forwardSpeedWorld: new Float32Array(TIER_FIVE_AGENT_COUNT),
    lateralSpeedWorld: new Float32Array(TIER_FIVE_AGENT_COUNT),
    speedWorld: new Float32Array(TIER_FIVE_AGENT_COUNT),
    // Step 3: Allocate the shared transport slabs: tires, radio rows, and pit shelf.
    tireState: new Float32Array(TIER_FIVE_TIRE_STATE_SIZE).fill(1),
    radioField: new Float32Array(TIER_FIVE_RADIO_FIELD_SIZE),
    pitStatus,
    focusCarIndex: 0,
    // Step 4: Allocate the remaining race-progress lanes.
    lap: new Uint16Array(TIER_FIVE_AGENT_COUNT),
    place: new Uint8Array(TIER_FIVE_AGENT_COUNT),
    raceTimeMs: 0,
    done: false,
  };
}

/**
 * Resolves which teammate radio rows the querying car may read from the shared field.
 *
 * The shared field stores one seven-channel radio row per car in the canonical
 * roster `[A0, A1, A2, B0, B1, B2]`. Visibility stays team-local and includes the
 * querying car row for parity with the Tier 3 helper contract.
 *
 * @param frame - Current packed race frame.
 * @param carIndex - Zero-based index of the querying car.
 * @returns Ordered teammate row indices visible to the querying car.
 * @example
 * ```ts
 * const teamAAnchorCarIndex = 0;
 * const teamBAnchorCarIndex = TIER_FIVE_TEAM_SIZE;
 * const frame = createTier5RacePack();
 *
 * resolveReadableRadioRows(frame, teamAAnchorCarIndex); // [0, 1, 2]
 * resolveReadableRadioRows(frame, teamBAnchorCarIndex); // [3, 4, 5]
 * ```
 */
export function resolveReadableRadioRows(
  frame: RacingRenderFrame,
  carIndex: number,
): readonly number[] {
  const teamIndex = frame.carTeam[carIndex] ?? 0;
  const readableRows: number[] = [];

  // Step 1: Collect only rows owned by the querying car's team.
  for (let agentIndex = 0; agentIndex < frame.agentCount; agentIndex++) {
    if (frame.carTeam[agentIndex] === teamIndex) {
      readableRows.push(agentIndex);
    }
  }

  return readableRows;
}
