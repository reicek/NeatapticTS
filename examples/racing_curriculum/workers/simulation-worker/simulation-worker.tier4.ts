/**
 * Tier 4 simulation-worker helpers for the smallest honest 2v2 tire-and-pit race pack.
 *
 * This keeps the worker seam narrow: one factory allocates the canonical
 * four-car packed frame, and one resolver returns the readable radio rows for a
 * given car. Tier 4 reuses the exact team-local radio visibility from Tier 3
 * while extending the packed frame with pit-status transport.
 */
import { resolveReadableRadioRows as resolveTierThreeReadableRadioRows } from './simulation-worker.tier3';
import type { RacingRenderFrame } from './simulation-worker.types';

/** Number of agents in a 2v2 Tier 4 race. */
const TIER_FOUR_AGENT_COUNT = 4;
/** Number of radio channels per agent. */
const TIER_FOUR_RADIO_CHANNEL_COUNT = 7;
/** Total radio field size for a 2v2 race. */
const TIER_FOUR_RADIO_FIELD_SIZE =
  TIER_FOUR_AGENT_COUNT * TIER_FOUR_RADIO_CHANNEL_COUNT;
/** Sentinel value used when a team's pit is currently unoccupied. */
const NO_CAR_INDEX = 255;

/**
 * Creates the canonical packed Tier 4 race frame for a 2v2 evaluation slice.
 *
 * The returned arrays are sized to the smallest honest tire-and-pit contract:
 * `tireState.length = 16` (`4 cars × 4 tires`) and `pitStatus.length = 4`
 * storing `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`. The car slots
 * start at `255` so an untouched pack always means `no car in pit`.
 *
 * @returns Four-car packed render frame with tire and pit transport enabled.
 * @example
 * ```ts
 * const frame = createTier4RacePack();
 *
 * frame.agentCount; // 4
 * frame.tireState.length; // 16
 * frame.pitStatus.length; // 4
 * ```
 */
export function createTier4RacePack(): RacingRenderFrame & {
  pitStatus: Int16Array;
} {
  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed: 0,
    trackId: 0,
    agentCount: TIER_FOUR_AGENT_COUNT,
    featureFlags: 0b111,
    carX: new Float32Array(TIER_FOUR_AGENT_COUNT),
    carY: new Float32Array(TIER_FOUR_AGENT_COUNT),
    carHeading: new Float32Array(TIER_FOUR_AGENT_COUNT),
    carActive: new Uint8Array(TIER_FOUR_AGENT_COUNT).fill(1),
    carTeam: Uint8Array.from([0, 0, 1, 1]),
    carMode: new Uint8Array(TIER_FOUR_AGENT_COUNT),
    tireState: new Float32Array(TIER_FOUR_AGENT_COUNT * 4).fill(1),
    radioField: new Float32Array(TIER_FOUR_RADIO_FIELD_SIZE),
    pitStatus: Int16Array.from([NO_CAR_INDEX, 0, NO_CAR_INDEX, 0]),
    lap: new Uint16Array(TIER_FOUR_AGENT_COUNT),
    place: new Uint8Array(TIER_FOUR_AGENT_COUNT),
    raceTimeMs: 0,
    done: false,
  };
}

/**
 * Resolves which radio rows the querying car may read from the shared field.
 *
 * Tier 4 keeps the exact Tier 3 readability contract: Team A reads rows
 * `[0, 1]`, Team B reads rows `[2, 3]`.
 *
 * @param frame - Current packed race frame.
 * @param carIndex - Zero-based index of the querying car.
 * @returns Ordered list of row indices visible to that car's team.
 */
export function resolveReadableRadioRows(
  frame: RacingRenderFrame,
  carIndex: number,
): readonly number[] {
  return resolveTierThreeReadableRadioRows(frame, carIndex);
}
