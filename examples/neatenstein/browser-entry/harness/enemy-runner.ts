/**
 * Headless enemy wave runner for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * The runner evaluates every variant in the enemy population against a fixed,
 * frozen seed pack in a single headless generation. It refreshes the rolling
 * enemy snapshot store through the generation barrier, runs a real bounded
 * MLP-driven rollout for each variant, scores the team-level result, and
 * selects a champion using deterministic lowest-id tie-breaking.
 *
 * @module
 */

import {
  buildEnemyDistanceMap,
  buildVisionVector,
  findBestNavigationStep,
  getDistance,
} from '../../scripts/enemy-navigation';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { buildNeatensteinMap, createCollisionMap } from '../renderer/map';
import type { CollisionMap } from '../renderer/map';

import { buildEnemyEvaluationBarrier } from './barrier';
import { NEATENSTEIN_MAX_EPISODE_TICKS } from './constants';
import { activateMlp, interpretMlpOutputs } from './enemy-mlp';
import { computeEnemyTeamFitness } from './fitness';
import { makeEnemySeedPack } from './seed-pack';
import { getEnemySnapshot } from './snapshot';
import { selectVariant } from './select';
import type {
  EnemyEpisodeTelemetry,
  EnemyPopulation,
  EnemyTeamFitnessConfig,
  FitnessScore,
  Individual,
  MlpSnapshot,
  SeedPack,
} from './types';

/**
 * Enemy movement speed in cells per tick.
 *
 * Derived from 2.5 cells/sec × 16 ms / 1000 ms ≈ 0.04 cells per fixed timestep.
 */
const ROLLOUT_SPEED_CELLS_PER_TICK = 0.04;

/**
 * Collision radius for wall checks during the rollout, in world cells.
 *
 * A quarter-cell radius ensures the enemy stays at least 0.25 cells away from
 * any wall edge, matching the runtime controller behaviour.
 */
const ROLLOUT_COLLISION_RADIUS_CELLS = 0.25;

/** Maximum BFS cell distance at which the enemy can hit the static player. */
const ROLLOUT_FIRE_RANGE_CELLS = 8;

/** Damage dealt per successful fire tick. */
const ROLLOUT_FIRE_DAMAGE = 10;

/** Fire cooldown in ticks (~1000 ms / 16 ms per tick ≈ 63). */
const ROLLOUT_FIRE_COOLDOWN_TICKS = 63;

/** Player goal position X (center of the map, guaranteed open by carveCentralArena). */
const PLAYER_GOAL_X = Math.floor(NEATENSTEIN_MAP_SIZE / 2);

/** Player goal position Y. */
const PLAYER_GOAL_Y = Math.floor(NEATENSTEIN_MAP_SIZE / 2);

/** Enemy spawn offset from the player goal (within the central arena, guaranteed open). */
const ENEMY_SPAWN_OFFSET = 2;

/**
 * Optional configuration accepted by {@link runEnemyWaveRunner}.
 */
export interface EnemyWaveRunnerConfig {
  /** Generation label used by the evaluation barrier (defaults to `seed`). */
  generation?: number;
  /** Optional weights for the team-level enemy fitness composite. */
  fitness?: EnemyTeamFitnessConfig;
}

/**
 * Result emitted by {@link runEnemyWaveRunner} for one enemy wave.
 */
export interface EnemyWaveRunnerResult {
  /** Selected enemy champion with its score and frozen snapshot. */
  champion: {
    /** Stable variant id of the champion. */
    id: number;
    /** Scalar score used for selection. */
    score: FitnessScore;
    /** Frozen snapshot backing the champion. */
    snapshot: MlpSnapshot;
  };
  /** Per-variant scores in population order. */
  scores: { id: number; score: FitnessScore }[];
  /** Generation label used for the barrier. */
  generation: number;
  /** Fixed seed pack the wave was evaluated against. */
  seedPack: SeedPack;
}

/**
 * Run a headless enemy wave evaluation.
 *
 * All 32 enemy variants are evaluated against the same deterministic seed
 * pack. The rolling snapshot store is refreshed through
 * {@link buildEnemyEvaluationBarrier} so the evaluation reads frozen snapshots,
 * never live mutable population weights. Selection uses
 * {@link selectVariant}, which resolves ties by choosing the lowest variant id.
 *
 * @param population - Live enemy population to evaluate.
 * @param seed - Deterministic seed for the seed pack and episode map generation.
 * @param config - Optional generation override and fitness weights.
 * @returns Champion, per-variant scores, generation, and the fixed seed pack.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 1 });
 * const result = runEnemyWaveRunner(population, 123);
 * console.log(result.champion.id, result.scores.length); // 0..31, 32
 * ```
 */
export function runEnemyWaveRunner(
  population: EnemyPopulation,
  seed: number,
  config: EnemyWaveRunnerConfig = {},
): EnemyWaveRunnerResult {
  const generation = config.generation ?? seed;
  const seedPack = makeEnemySeedPack(seed);

  // Refresh the frozen snapshot store and establish the generation barrier.
  buildEnemyEvaluationBarrier(generation, population, seedPack);

  const evaluated: Individual<MlpSnapshot>[] = [];
  const scores: { id: number; score: FitnessScore }[] = [];

  for (let variantId = 0; variantId < population.size; variantId++) {
    const snapshot = getEnemySnapshot(variantId);
    const episodeSeed = seedPack.seeds[variantId % seedPack.seeds.length];
    const telemetry = simulateEnemyEpisode(snapshot, episodeSeed);
    const score = computeEnemyTeamFitness(telemetry, config.fitness);

    evaluated.push({ id: variantId, variant: snapshot, fitness: score });
    scores.push({ id: variantId, score });
  }

  const champion = selectVariant(evaluated);

  return {
    champion: {
      id: champion.id,
      score: champion.fitness as FitnessScore,
      snapshot: champion.variant,
    },
    scores,
    generation,
    seedPack,
  };
}

/**
 * Check whether a circular enemy position overlaps any wall cell in the
 * collision map. Uses the same point-sampling pattern as the runtime
 * `enemy-controller.ts` `isPositionBlockedByWall` helper.
 *
 * @param collisionMap - Map collision interface.
 * @param x - Enemy center X in world cells.
 * @param y - Enemy center Y in world cells.
 * @param radius - Collision radius in cells.
 * @returns `true` when any sampled point around the enemy is inside a wall.
 */
function isPositionBlocked(
  collisionMap: CollisionMap,
  x: number,
  y: number,
  radius: number,
): boolean {
  const minX = Math.floor(x - radius);
  const maxX = Math.floor(x + radius);
  const minY = Math.floor(y - radius);
  const maxY = Math.floor(y + radius);

  for (let cy = minY; cy <= maxY; cy++) {
    for (let cx = minX; cx <= maxX; cx++) {
      if (collisionMap.isSolid(cx, cy)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Simulate one deterministic, bounded MLP-driven episode for a frozen enemy
 * variant.
 *
 * The rollout materializes the enemy's {@link MlpSnapshot} weights and runs up
 * to {@link NEATENSTEIN_MAX_EPISODE_TICKS} ticks. Each tick:
 *
 * 1. Builds a 6-input vision vector from the BFS distance map (relative
 *    distance, movement progress, previous distance).
 * 2. Calls {@link activateMlp} with the snapshot weights and vision vector.
 * 3. Interprets the 4 outputs (move, strafe, turn, fire) via
 *    {@link interpretMlpOutputs}.
 * 4. Computes a navigation step using the BFS distance map toward the static
 *    player goal and scales it by the move/strafe outputs.
 * 5. Checks wall collision and updates position, stagnation, and telemetry.
 *
 * The static player sits at the center of the map (guaranteed open by
 * `carveCentralArena`). The BFS distance map is built once before the loop
 * because the map and player position are constant across ticks, producing the
 * same result as recomputing each tick (AC-10.5c-002).
 *
 * @param snapshot - Frozen enemy MLP snapshot (weights used directly by activateMlp).
 * @param episodeSeed - Deterministic seed from the generation seed pack (selects spawn position).
 * @returns Per-episode telemetry: damage, survival, exploration, stagnation, and final distance.
 */
export function simulateEnemyEpisode(
  snapshot: MlpSnapshot,
  episodeSeed: number,
): EnemyEpisodeTelemetry {
  // Build the static map and collision interface once.
  const flatMap = buildNeatensteinMap(episodeSeed);
  const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);

  // Static player goal at map center (guaranteed open by carveCentralArena).
  const playerX = PLAYER_GOAL_X;
  const playerY = PLAYER_GOAL_Y;

  // BFS distance map from the player position — constant across ticks.
  const distanceMap = buildEnemyDistanceMap(
    collisionMap,
    playerX,
    playerY,
    NEATENSTEIN_MAP_SIZE,
  );

  // Deterministic spawn offset derived from the episode seed.
  // Both offset components are in [-ENEMY_SPAWN_OFFSET, +ENEMY_SPAWN_OFFSET].
  const seedHash =
    (((episodeSeed * 2654435761) >>> 0) % (ENEMY_SPAWN_OFFSET * 2 + 1)) -
    ENEMY_SPAWN_OFFSET;
  const spawnOffsetX = seedHash;
  const spawnOffsetY = -seedHash; // mirror to stay within the central arena

  let enemyX = playerX + spawnOffsetX;
  let enemyY = playerY + spawnOffsetY;

  // Clamp spawn to map bounds (central arena guarantees it is open).
  enemyX = Math.max(1, Math.min(NEATENSTEIN_MAP_SIZE - 2, enemyX));
  enemyY = Math.max(1, Math.min(NEATENSTEIN_MAP_SIZE - 2, enemyY));

  const visitedCells = new Set<string>();
  visitedCells.add(`${Math.floor(enemyX)},${Math.floor(enemyY)}`);

  let damageDealt = 0;
  let stagnationTicks = 0;
  let previousDistance = getDistance(
    distanceMap,
    Math.floor(enemyX),
    Math.floor(enemyY),
  );
  const bfsDistances: number[] = [];
  let fireCooldown = 0;

  for (let tick = 0; tick < NEATENSTEIN_MAX_EPISODE_TICKS; tick++) {
    const cellX = Math.floor(enemyX);
    const cellY = Math.floor(enemyY);

    // Record the BFS distance at the start of this tick.
    bfsDistances.push(previousDistance);

    // 1. Build vision vector from the BFS distance map.
    const vision = buildVisionVector(
      distanceMap,
      cellX,
      cellY,
      previousDistance,
    );

    // 2. Forward pass through the MLP.
    const outputs = activateMlp(snapshot.weights, vision);

    // 3. Interpret outputs.
    const { move, strafe, fire } = interpretMlpOutputs(outputs);

    // 4. Navigation step via BFS.
    const step = findBestNavigationStep(distanceMap, cellX, cellY);
    if (step !== null) {
      // Scale the BFS step by the MLP move/strafe outputs (both in [-1, 1]).
      const dx = step.dx * move + -step.dy * strafe;
      const dy = step.dy * move + step.dx * strafe;
      const newX = enemyX + dx * ROLLOUT_SPEED_CELLS_PER_TICK;
      const newY = enemyY + dy * ROLLOUT_SPEED_CELLS_PER_TICK;

      if (
        !isPositionBlocked(
          collisionMap,
          newX,
          newY,
          ROLLOUT_COLLISION_RADIUS_CELLS,
        ) &&
        (Math.abs(newX - enemyX) > 1e-9 || Math.abs(newY - enemyY) > 1e-9)
      ) {
        enemyX = newX;
        enemyY = newY;
        const newCellX = Math.floor(enemyX);
        const newCellY = Math.floor(enemyY);
        visitedCells.add(`${newCellX},${newCellY}`);
        previousDistance = getDistance(distanceMap, newCellX, newCellY);
      } else {
        stagnationTicks++;
      }
    } else {
      stagnationTicks++;
    }

    // 5. Fire model: if fire output > 0 and within range and cooldown elapsed.
    if (fire > 0 && fireCooldown <= 0) {
      const bfsDistance = getDistance(
        distanceMap,
        Math.floor(enemyX),
        Math.floor(enemyY),
      );
      if (bfsDistance >= 0 && bfsDistance <= ROLLOUT_FIRE_RANGE_CELLS) {
        damageDealt += ROLLOUT_FIRE_DAMAGE;
      }
      fireCooldown = ROLLOUT_FIRE_COOLDOWN_TICKS;
    }
    if (fireCooldown > 0) {
      fireCooldown--;
    }
  }

  const finalCellX = Math.floor(enemyX);
  const finalCellY = Math.floor(enemyY);
  const finalDistance = getDistance(distanceMap, finalCellX, finalCellY);

  return {
    position: { x: finalCellX, y: finalCellY },
    bfsDistances,
    damageDealt,
    enemiesSurvived: 1,
    cellsVisited: visitedCells.size,
    stagnationTicks,
    finalDistance: finalDistance < 0 ? -1 : finalDistance,
  };
}
