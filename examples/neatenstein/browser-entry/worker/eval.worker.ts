/**
 * Dedicated evaluation worker for the Neatenstein NEAT population evaluation.
 *
 * Offloads the CPU-heavy NEAT population evaluation (episode-based fitness
 * scoring) from the display worker's render loop so the render loop never
 * stalls during generation evaluation (500–2000 ms stalls without this
 * worker).
 *
 * Message protocol:
 *
 * - **Inbound** (from display worker):
 *   `{ type: 'evaluate', seed, generation, enemySnapshot, humanMode }`
 * - **Outbound** (to display worker):
 *   `{ type: 'evalComplete', generation, championNetworkJSON }`
 *
 * The champion network is serialized via `Network.toJSON()` so it can cross
 * the `postMessage` structured-clone boundary. The display worker
 * deserializes it with `Network.fromJSON()`.
 *
 * @module
 */

/// <reference lib="webworker" />

import { buildNeatensteinMap, createCollisionMap } from '../renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { gameTick } from '../host/game/tick';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../host/game/constants';
import type { EpisodeTelemetry } from '../host/game/types';
import { createEpisode, endEpisode } from '../host/game/episode';
import { extractSensors } from '../../scripts/enemy-navigation';
import {
  createFireGateState,
  ENEMY_VISIBLE_SENSOR_INDEX,
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_MAIN_NEAT_OUTPUTS,
  networkOutputToTickInput,
} from '../harness/neat-io-config';
import type { Network } from 'neataptic';
import {
  computeCombatQualitySignal,
  extractCombatQualitySignal,
} from '../harness/fitness';
import { NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS } from '../harness/constants';
import type { CombatQualitySignal, Snapshot } from '../harness/types';
import { runArmsRaceGeneration } from '../harness/arms-race';
import { hashSeed } from '../harness/hash-seed';

/**
 * Population size for the hoisted Neat evaluation.
 *
 * @see AC-039
 */
const NEATENSTEIN_MAIN_NEAT_POPSIZE = 4;

/**
 * Run a deterministic fitness evaluation episode for a NEAT network.
 *
 * Creates a fresh game episode from the evaluation seed, drives the network
 * through the full game tick pipeline for
 * {@link NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS} ticks, and extracts the
 * combat-quality signal from the final game state and telemetry.
 *
 * @param network - NEAT network to evaluate.
 * @param episodeSeed - Deterministic seed for the episode.
 * @returns Combat-quality signal from the completed episode.
 */
/**
 * Run a deterministic fitness evaluation episode for a NEAT network.
 *
 * Creates a fresh game episode from the evaluation seed, drives the network
 * through the full game tick pipeline for
 * {@link NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS} ticks, and extracts the
 * combat-quality signal from the final game state and telemetry.
 *
 * Exported for testability of the fire-gate wiring; not part of the public
 * worker API.
 *
 * @param network - NEAT network to evaluate.
 * @param episodeSeed - Deterministic seed for the episode.
 * @returns Combat-quality signal from the completed episode.
 */
export function runFitnessEpisode(
  network: Network,
  episodeSeed: number,
): CombatQualitySignal {
  const episode = createEpisode({ seed: episodeSeed });
  let state = episode.state;
  const flatMap = buildNeatensteinMap(state.seed);
  const epCollisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);

  // P5S1: Episode-local fire-gate state so hysteresis persists across ticks,
  // matching the display worker's module-level fireGateState.
  const fireGateState = createFireGateState();

  for (let tick = 0; tick < NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS; tick += 1) {
    const sensors = extractSensors(
      state,
      flatMap,
      NEATENSTEIN_MAP_SIZE,
      epCollisionMap,
    );
    const raw = network.activate(sensors);
    const out: number[] = Array.isArray(raw)
      ? raw.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0))
      : new Array<number>(NEATENSTEIN_MAIN_NEAT_OUTPUTS).fill(0);
    const tickInput = networkOutputToTickInput(out, {
      state: fireGateState,
      enemyVisible: sensors[ENEMY_VISIBLE_SENSOR_INDEX] ?? 0,
    });
    state = gameTick(
      state,
      tickInput,
      epCollisionMap,
      NEATENSTEIN_FIXED_TIMESTEP_MS,
    );
  }

  const finalState = endEpisode(state);
  const telemetry: EpisodeTelemetry = finalState.telemetry ?? {
    damageDealt: 0,
    shotsFired: 0,
    shotsHit: 0,
    aimMissRate: 0,
    shotsWallHit: 0,
    shotsRangeExpired: 0,
    shotsBlindFire: 0,
    shotsNearMiss: 0,
  };

  return extractCombatQualitySignal(finalState, telemetry);
}

/**
 * Inbound evaluation request payload received from the display worker.
 */
interface EvalRequestPayload {
  type: 'evaluate';
  seed: number;
  generation: number;
  enemySnapshot: Snapshot;
  humanMode: boolean;
}

/**
 * Lazy-load the Neat constructor from the neataptic entry point.
 *
 * Kept as a mutable getter so tests can inject a lightweight mock Neat
 * without pulling the full library into the worker test environment.
 */
let getNeatConstructor: () => Promise<
  Pick<typeof import('neataptic'), 'Neat'>
> = async () => {
  const { Neat } = await import('neataptic');
  return { Neat };
};

/**
 * Test-only hook to inject a mock Neat constructor.
 *
 * @param getter - Function returning a `{ Neat }` object for the worker.
 */
export function __testOnlySetNeatConstructor(
  getter: () => Promise<Pick<typeof import('neataptic'), 'Neat'>>,
): void {
  getNeatConstructor = getter;
}

/**
 * Test-only hook to read the current Neat constructor loader.
 *
 * Used to cover the lazy default loader path without instantiating Neat in
 * the worker test environment.
 *
 * @returns The current Neat constructor getter.
 */
export function __testOnlyGetNeatConstructor(): () => Promise<
  Pick<typeof import('neataptic'), 'Neat'>
> {
  return getNeatConstructor;
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data as EvalRequestPayload | null;
  if (!data || typeof data !== 'object' || data.type !== 'evaluate') {
    return;
  }

  const { seed, generation, enemySnapshot } = data;
  const humanModeBool = data.humanMode === true;

  const popSeed = hashSeed(seed, generation);

  // Lazy-load Neat to avoid pulling the full neataptic entry point into the
  // static import chain. The getter is overridable for tests.
  const { Neat } = await getNeatConstructor();

  // Episode-based fitness function: run a full deterministic game episode
  // per network and score it with the combat-quality composite.
  const fitnessFn = (network: Network): number => {
    const signal = runFitnessEpisode(network, popSeed);
    return computeCombatQualitySignal(signal);
  };

  const neatPop = new Neat(
    NEATENSTEIN_MAIN_NEAT_INPUTS,
    NEATENSTEIN_MAIN_NEAT_OUTPUTS,
    fitnessFn,
    {
      popsize: NEATENSTEIN_MAIN_NEAT_POPSIZE,
      seed: popSeed,
      maxNodes: 64,
      maxConns: 256,
    },
  );

  // Evaluate and evolve the population.
  await neatPop.evaluate();
  await neatPop.evolve();

  // Extract the champion network.
  const championNetwork = neatPop.getFittest();

  // Extract the real combat-quality signal from the champion's episode.
  const championQuality = runFitnessEpisode(championNetwork, popSeed);

  // Call the synchronous arms-race runner with the champion network.
  const result = runArmsRaceGeneration({
    seed,
    generation,
    enemySnapshot,
    humanMode: humanModeBool,
    championNetwork,
    championQuality,
  });

  // Serialize the champion network so it can cross the postMessage boundary.
  const championNetworkJSON = championNetwork.toJSON();

  self.postMessage({
    type: 'evalComplete',
    generation: result.generation,
    championNetworkJSON,
  });
};
