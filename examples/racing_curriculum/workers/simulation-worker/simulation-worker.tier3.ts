/**
 * Tier 3 simulation-worker helpers for the smallest honest 2v2 race pack.
 *
 * This module keeps the Tier 3 worker contract narrow: one factory allocates
 * the canonical four-car packed frame, and one resolver answers which radio
 * rows a car may read. That is enough to exercise teammate communication
 * without hard-coding later tournament logic into the worker seam.
 */
import type { RacingRenderFrame } from './simulation-worker.types';

/** Number of agents in a 2v2 Tier 3 race (two per team). */
const TIER_THREE_AGENT_COUNT = 4;
/** Number of radio channels per agent. */
const TIER_THREE_RADIO_CHANNEL_COUNT = 7;
/** Total radio field size for a 2v2 race. */
const TIER_THREE_RADIO_FIELD_SIZE =
  TIER_THREE_AGENT_COUNT * TIER_THREE_RADIO_CHANNEL_COUNT;

/**
 * Creates the canonical packed Tier 3 race frame for a 2v2 evaluation slice.
 *
 * The frame follows the shared structure-of-arrays contract used by the rest of
 * the worker pipeline: four car rows, `carTeam = [0, 0, 1, 1]`, radio enabled
 * through feature-flag bit 1, and a 28-float `radioField` laid out as
 * 4 rows × 7 channels.
 *
 * @returns Four-car packed render frame with a 2v2 team layout.
 * @example
 * ```ts
 * const frame = createTier3RacePack();
 *
 * frame.agentCount; // 4
 * frame.radioField.length; // 28
 * Array.from(frame.carTeam); // [0, 0, 1, 1]
 * ```
 */
export function createTier3RacePack(): RacingRenderFrame {
  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed: 0,
    trackId: 0,
    agentCount: TIER_THREE_AGENT_COUNT,
    featureFlags: 0b10,
    carX: new Float32Array(TIER_THREE_AGENT_COUNT),
    carY: new Float32Array(TIER_THREE_AGENT_COUNT),
    carHeading: new Float32Array(TIER_THREE_AGENT_COUNT),
    carActive: new Uint8Array(TIER_THREE_AGENT_COUNT).fill(1),
    carTeam: Uint8Array.from([0, 0, 1, 1]),
    carMode: new Uint8Array(TIER_THREE_AGENT_COUNT),
    tireState: new Float32Array(TIER_THREE_AGENT_COUNT * 4).fill(1),
    radioField: new Float32Array(TIER_THREE_RADIO_FIELD_SIZE),
    lap: new Uint16Array(TIER_THREE_AGENT_COUNT),
    place: new Uint8Array(TIER_THREE_AGENT_COUNT),
    raceTimeMs: 0,
    done: false,
  };
}

/**
 * Resolves which radio rows the querying car may read from the shared field.
 *
 * Tier 3 keeps radio visibility team-local: a car may read its own row and its
 * teammate's row, but not the opposing team's rows. The resolver therefore
 * walks the packed frame and returns only indices whose `carTeam` matches the
 * querying car.
 *
 * @param frame - Current packed race frame.
 * @param carIndex - Zero-based index of the querying car.
 * @returns Ordered list of row indices visible to that car's team.
 * @example
 * ```ts
 * const frame = createTier3RacePack();
 *
 * resolveReadableRadioRows(frame, 0); // [0, 1]
 * resolveReadableRadioRows(frame, 2); // [2, 3]
 * ```
 */
export function resolveReadableRadioRows(
  frame: RacingRenderFrame,
  carIndex: number,
): readonly number[] {
  const teamId = frame.carTeam[carIndex] ?? 0;
  const readableRows: number[] = [];

  // Step 1: Collect only the row indices owned by the querying car's team.
  for (let agentIndex = 0; agentIndex < frame.agentCount; agentIndex++) {
    if (frame.carTeam[agentIndex] === teamId) {
      readableRows.push(agentIndex);
    }
  }

  return readableRows;
}
