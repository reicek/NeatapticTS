/**
 * Unified league structure (AlphaStar-style) for the Neatenstein co-evolution
 * harness.
 *
 * Replaces the separate hall-of-fame and opponent-pool concepts with a single
 * league containing current champions, past champions (main exploiters), and
 * diverse strategy samples from MAP-Elites.
 *
 * Stub module — implementation lands in Step B4 (04-implementing).
 * Tests in `league.test.ts` define the RED contracts.
 *
 * @module
 */

/**
 * Unified league structure (AlphaStar-style) for opponent management.
 *
 * Replaces separate hall-of-fame and opponent-pool structures with a single
 * league containing current champions, past champions (main exploiters for
 * forgetting prevention), and diverse strategy samples from MAP-Elites.
 *
 * Past champions use newest-first ordering: when a new champion is added, the
 * old current champion is unshifted to the front of the past champions array.
 * When the array exceeds capacity, the most recently archived champion (at the
 * front) is evicted, preserving the oldest champions for the curriculum.
 *
 * @module
 */

/**
 * Interface for a champion entry in the league.
 */
export interface LeagueChampion {
  /** Weight vector. */
  weights: Float32Array;
  /** Generation number. */
  generation: number;
  /** Fitness score. */
  fitness: number;
}

/**
 * Interface for a diverse strategy sample.
 */
export interface DiverseSample {
  /** Weight vector. */
  weights: Float32Array;
  /** Behavior descriptors. */
  behaviorMetrics: {
    aggression: number;
    positioning: number;
    movementPattern?: number;
  };
}

/**
 * League configuration.
 */
export interface LeagueConfig {
  /** Maximum number of past champions to retain. */
  maxPastChampions: number;
  /** Maximum number of diverse samples to retain. */
  maxDiverseSamples: number;
}

/**
 * League state containing champions and diverse samples.
 */
export interface LeagueState {
  /** Current champion (most recent). */
  currentChampion: LeagueChampion | null;
  /** Past champions, newest-first. */
  pastChampions: LeagueChampion[];
  /** Diverse strategy samples from MAP-Elites. */
  diverseSamples: DiverseSample[];
  /** Maximum past champion capacity. */
  maxPastChampions: number;
  /** Maximum diverse sample capacity. */
  maxDiverseSamples: number;
}

/**
 * Creates an empty league with bounded capacity.
 *
 * @param config Configuration with maxPastChampions and maxDiverseSamples.
 * @returns A new empty league state.
 */
export function createLeague(config: LeagueConfig): LeagueState {
  return {
    currentChampion: null,
    pastChampions: [],
    diverseSamples: [],
    maxPastChampions: config.maxPastChampions,
    maxDiverseSamples: config.maxDiverseSamples,
  };
}

/**
 * Adds a new current champion. The previous current champion (if any) is
 * archived to the front of the past champions array. If the past champions
 * array exceeds capacity, the most recently archived entry is evicted.
 *
 * @param state The league state.
 * @param champion The new champion to add.
 */
export function addCurrentChampion(
  state: LeagueState,
  champion: LeagueChampion,
): void {
  if (state.currentChampion) {
    state.pastChampions.unshift(state.currentChampion);
    while (state.pastChampions.length > state.maxPastChampions) {
      state.pastChampions.shift();
    }
  }
  state.currentChampion = champion;
}

/**
 * Archives the current champion to the past champions list without adding a
 * replacement. The current champion slot is cleared.
 *
 * @param state The league state.
 */
export function archiveCurrentChampion(state: LeagueState): void {
  if (state.currentChampion) {
    state.pastChampions.unshift(state.currentChampion);
    while (state.pastChampions.length > state.maxPastChampions) {
      state.pastChampions.shift();
    }
    state.currentChampion = null;
  }
}

/**
 * Adds a diverse strategy sample to the league. If the diverse samples exceed
 * capacity, the oldest entry is evicted.
 *
 * @param state The league state.
 * @param sample The diverse sample to add.
 */
export function addDiverseSample(
  state: LeagueState,
  sample: DiverseSample,
): void {
  state.diverseSamples.push(sample);
  while (state.diverseSamples.length > state.maxDiverseSamples) {
    state.diverseSamples.shift();
  }
}

/**
 * Samples a mix of past champions and diverse samples as opponents.
 *
 * @param state The league state.
 * @param count The number of opponents to sample.
 * @param seed Optional random seed for deterministic sampling.
 * @returns Array of opponent entries (champions or diverse samples).
 */
export function sampleOpponents(
  state: LeagueState,
  count: number,
  seed?: number,
): unknown[] {
  const pool: unknown[] = [...state.pastChampions, ...state.diverseSamples];
  if (state.currentChampion) {
    pool.push(state.currentChampion);
  }
  if (pool.length === 0) {
    return [];
  }
  // Deterministic simple sampling if seed provided, otherwise take first N
  const result: unknown[] = [];
  if (typeof seed === 'number') {
    let s = seed >>> 0;
    const rng = () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const used = new Set<number>();
    while (result.length < count && result.length < pool.length) {
      const idx = Math.floor(rng() * pool.length);
      if (!used.has(idx)) {
        used.add(idx);
        result.push(pool[idx]);
      }
    }
  } else {
    for (let i = 0; i < count && i < pool.length; i++) {
      result.push(pool[i]);
    }
  }
  return result;
}

/**
 * Returns past champions for periodic re-evaluation (forgetting prevention
 * curriculum).
 *
 * @param state The league state.
 * @returns Array of past champions.
 */
export function getCurriculumOpponents(state: LeagueState): LeagueChampion[] {
  return state.pastChampions;
}
