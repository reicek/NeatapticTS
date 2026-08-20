/**
 * WeightSharedCohort (SWARM) backend for the Neatenstein enemy population.
 *
 * This module materializes a small cohort of enemies that share a single DNA
 * string and a shared weight vector. Each enemy member receives a distinct
 * coordinate injection so the swarm behaves as one genotype with many
 * phenotypic bodies. The cohort refreshes on the cadence defined by
 * {@link NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS}.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import {
  NEATENSTEIN_SWARM_MAX_SIZE,
  NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS,
} from './constants';
import { SNAPSHOT_KIND_SWARM } from '../constants';
import type {
  Snapshot,
  SwarmSnapshot,
  CreateSwarmEnemyPopulationOptions,
  SwarmVariant,
  SwarmEnemyPopulation,
} from './types';
import type { Vector2 } from '../host/game/types';

/**
 * Options accepted by {@link createSwarmEnemyPopulation}.
 *
 */
export type { CreateSwarmEnemyPopulationOptions } from './types';

/**
 * One member of the weight-shared cohort.
 *
 */
export type { SwarmVariant } from './types';

/**
 * Swarm enemy population returned by {@link createSwarmEnemyPopulation}.
 *
 */
export type { SwarmEnemyPopulation } from './types';

/** Number of coordinate points injected into each enemy member. */
const COORDINATES_PER_MEMBER = 4;

/**
 * Create a WeightSharedCohort enemy population.
 *
 * The returned population exposes the common {@link EnemyPopulation} contract
 * plus an {@link SwarmEnemyPopulation.update | update} method used by the
 * harness to gate snapshot refreshes every 3 generations.
 *
 * @param options - Population configuration. Defaults to `seed: 0` and
 *   `size: NEATENSTEIN_SWARM_MAX_SIZE` so zero-argument calls still produce a
 *   deterministic population.
 * @returns A swarm enemy population with shared DNA and per-enemy coordinates.
 *
 * @example
 * ```ts
 * const population = createSwarmEnemyPopulation({ seed: 7 });
 * const variant = population.sample(0) as SwarmVariant;
 * console.log(variant.coordinates.length); // 4
 * ```
 */
export function createSwarmEnemyPopulation(
  options: CreateSwarmEnemyPopulationOptions = {},
): SwarmEnemyPopulation {
  const seed = options.seed ?? 0;
  const size = Math.min(
    options.size ?? NEATENSTEIN_SWARM_MAX_SIZE,
    NEATENSTEIN_SWARM_MAX_SIZE,
  );
  const dna = createDna(seed);
  const weights = createSharedWeights(seed);
  const members = createMembers(seed, size, dna, weights);

  let championSnapshot: SwarmSnapshot = createSnapshot(
    dna,
    members[0].coordinates,
  );

  return {
    kind: SNAPSHOT_KIND_SWARM,
    size,
    sample(index: number): unknown {
      // Index-stable sampling: any index maps deterministically to a member.
      const safeIndex = Number.isFinite(index) && index >= 0 ? index % size : 0;
      return members[safeIndex] as unknown;
    },
    snapshot: () => championSnapshot,
    update: ({ generation }: { generation: number }): Snapshot => {
      if (generation % NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS === 0) {
        championSnapshot = createSnapshot(
          dna,
          createChampionCoordinates(seed, generation),
        );
      }
      return championSnapshot;
    },
  };
}

/**
 * Build all members of the cohort for the population seed.
 *
 * @param seed - Population seed.
 * @param size - Cohort size.
 * @param dna - Shared DNA string.
 * @param weights - Shared weight vector.
 * @returns Ordered list of enemy variants.
 */
function createMembers(
  seed: number,
  size: number,
  dna: string,
  weights: Float32Array,
): SwarmVariant[] {
  const members: SwarmVariant[] = [];
  for (let i = 0; i < size; i++) {
    members.push({
      id: i,
      dna,
      weights,
      coordinates: createMemberCoordinates(seed, i),
    });
  }
  return members;
}

/**
 * Generate the deterministic shared DNA string for the cohort.
 *
 * The same `seed` always produces the same DNA, and every member of the
 * cohort receives the same reference.
 *
 * @param seed - Population seed.
 * @returns A compact DNA identifier.
 */
function createDna(seed: number): string {
  const rng = seedrandom(`${seed}:dna`);
  return `swarm:${seed}:${rng().toString(36).slice(2, 10)}`;
}

/**
 * Generate the shared weight vector for the cohort.
 *
 * @param seed - Population seed.
 * @returns A new shared weight vector.
 */
function createSharedWeights(seed: number): Float32Array {
  const rng = seedrandom(`${seed}:weights`);
  const weights = new Float32Array(8);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = rng() * 2 - 1;
  }
  return weights;
}

/**
 * Generate distinct coordinates for one enemy member.
 *
 * Coordinates are sampled from a per-member seeded PRNG so the same `seed`
 * and `index` always produce the same injection, while different indices are
 * extremely unlikely to collide.
 *
 * @param seed - Population seed.
 * @param index - Stable member index.
 * @returns Distinct coordinates for this member.
 */
function createMemberCoordinates(seed: number, index: number): Vector2[] {
  const rng = seedrandom(`${seed}:enemy:${index}`);
  const coordinates: Vector2[] = [];
  for (let i = 0; i < COORDINATES_PER_MEMBER; i++) {
    coordinates.push({ x: rng() * 2 - 1, y: rng() * 2 - 1 });
  }
  return coordinates;
}

/**
 * Generate the champion coordinate injection for a refresh generation.
 *
 * @param seed - Population seed.
 * @param generation - Generation at which the snapshot refreshes.
 * @returns A new deterministic champion coordinate set.
 */
function createChampionCoordinates(
  seed: number,
  generation: number,
): Vector2[] {
  const rng = seedrandom(`${seed}:refresh:${generation}`);
  const coordinates: Vector2[] = [];
  for (let i = 0; i < COORDINATES_PER_MEMBER; i++) {
    coordinates.push({ x: rng() * 2 - 1, y: rng() * 2 - 1 });
  }
  return coordinates;
}

/**
 * Wrap a DNA string and coordinate set in the SWARM snapshot shape.
 *
 * @param dna - Shared DNA identifier.
 * @param coordinates - Champion coordinates.
 * @returns A serializable SWARM snapshot.
 */
function createSnapshot(dna: string, coordinates: Vector2[]): SwarmSnapshot {
  return {
    kind: SNAPSHOT_KIND_SWARM,
    dna,
    coordinates: coordinates.map((point) => ({ x: point.x, y: point.y })),
  };
}
