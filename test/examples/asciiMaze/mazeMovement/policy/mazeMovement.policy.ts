/**
 * Action policy helpers for the dedicated mazeMovement module.
 *
 * This file owns direction selection, epsilon handling, short-horizon policy
 * overrides, and saturation-driven bias control.
 */

import type { INetwork, INodeStruct } from '../../interfaces';
import { MazeUtils } from '../../mazeUtils';
import { MAZE_MOVEMENT_CONSTANTS } from '../mazeMovement.constants';
import {
  getMazeMovementRunServiceState,
  randomMazeMovementUnit,
  readMazeMovementOutputHistory,
  writeMazeMovementOutputHistory,
} from '../mazeMovement.services';
import type {
  DirectionSelectionStats,
  SimulationState,
} from '../mazeMovement.types';
import {
  getMazeMovementDistance,
  isMazeMovementCellOpen,
} from '../runtime/mazeMovement.runtime';

const C = MAZE_MOVEMENT_CONSTANTS;
const POLICY_SCRATCH_CENTERED = new Float64Array(4);
const POLICY_SCRATCH_EXPS = new Float64Array(4);
const POLICY_SOFTMAX = new Float64Array(4);

/**
 * Compute the adaptive epsilon used for policy exploration.
 *
 * @param stepNumber - Global step number inside the active simulation.
 * @param stepsSinceImprovement - Number of steps without improvement.
 * @param distHere - Current distance to goal for the active position.
 * @param saturations - Rolling saturation count from the shared run state.
 * @returns Exploration epsilon in the range `[0, 1]`.
 */
export function computeMazeMovementEpsilon(
  stepNumber: number,
  stepsSinceImprovement: number,
  distHere: number,
  saturations: number,
): number {
  const isWarmup = stepNumber < C.EPSILON_WARMUP_STEPS;
  const isHighlyStagnant =
    stepsSinceImprovement > C.EPSILON_STAGNANT_HIGH_THRESHOLD;
  const isModeratelyStagnant =
    stepsSinceImprovement > C.EPSILON_STAGNANT_MED_THRESHOLD;
  const isSaturationTriggered = saturations > C.EPSILON_SATURATION_TRIGGER;

  let chosenEpsilon = 0;
  switch (true) {
    case isWarmup:
      chosenEpsilon = C.EPSILON_INITIAL;
      break;
    case isHighlyStagnant:
      chosenEpsilon = C.EPSILON_STAGNANT_HIGH;
      break;
    case isModeratelyStagnant:
      chosenEpsilon = C.EPSILON_STAGNANT_MED;
      break;
    case isSaturationTriggered:
      chosenEpsilon = C.EPSILON_SATURATIONS;
      break;
    default:
      break;
  }

  if (distHere <= C.PROXIMITY_SUPPRESS_EXPLOR_DIST) {
    chosenEpsilon = Math.min(chosenEpsilon, C.EPSILON_MIN_NEAR_GOAL);
  }

  return chosenEpsilon;
}

/**
 * Convert raw network outputs into a chosen direction plus diagnostics.
 *
 * @param outputs - Raw action logits for the four maze directions.
 * @returns Chosen direction plus softmax and entropy diagnostics.
 */
export function selectMazeMovementDirection(
  outputs: number[],
): DirectionSelectionStats {
  const actionCount = C.ACTION_DIM;
  if (!Array.isArray(outputs) || outputs.length !== actionCount) {
    return {
      direction: C.NO_MOVE,
      softmax: Array.from(POLICY_SOFTMAX),
      entropy: 0,
      maxProb: 0,
      secondProb: 0,
    };
  }

  let outputSum = 0;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    outputSum += outputs[actionIndex];
  }
  const meanOutput = outputSum / actionCount;

  let varianceAccumulator = 0;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    const delta = outputs[actionIndex] - meanOutput;
    POLICY_SCRATCH_CENTERED[actionIndex] = delta;
    varianceAccumulator += delta * delta;
  }
  varianceAccumulator /= actionCount;
  let standardDeviation = Math.sqrt(varianceAccumulator);
  if (!Number.isFinite(standardDeviation) || standardDeviation < C.STD_MIN) {
    standardDeviation = C.STD_MIN;
  }

  const collapseRatio =
    standardDeviation < C.COLLAPSE_STD_THRESHOLD
      ? C.COLLAPSE_RATIO_FULL
      : standardDeviation < C.COLLAPSE_STD_MED
        ? C.COLLAPSE_RATIO_HALF
        : 0;
  const temperature = C.TEMPERATURE_BASE + C.TEMPERATURE_SCALE * collapseRatio;

  let maxCentered = -Infinity;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    const centeredValue = POLICY_SCRATCH_CENTERED[actionIndex];
    if (centeredValue > maxCentered) maxCentered = centeredValue;
  }

  let expSum = 0;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    const expValue = Math.exp(
      (POLICY_SCRATCH_CENTERED[actionIndex] - maxCentered) / temperature,
    );
    POLICY_SCRATCH_EXPS[actionIndex] = expValue;
    expSum += expValue;
  }
  if (expSum === 0) expSum = 1;

  let chosenDirection = 0;
  let bestProbability = -Infinity;
  let runnerUpProbability = 0;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    const probability = POLICY_SCRATCH_EXPS[actionIndex] / expSum;
    POLICY_SOFTMAX[actionIndex] = probability;
    if (probability > bestProbability) {
      runnerUpProbability = bestProbability;
      bestProbability = probability;
      chosenDirection = actionIndex;
    } else if (probability > runnerUpProbability) {
      runnerUpProbability = probability;
    }
  }

  let entropy = 0;
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex++) {
    const probability = POLICY_SOFTMAX[actionIndex];
    if (probability > 0) entropy += -probability * Math.log(probability);
  }
  entropy /= C.LOG_ACTIONS;

  return {
    direction: chosenDirection,
    softmax: Array.from(POLICY_SOFTMAX),
    entropy,
    maxProb: bestProbability,
    secondProb: runnerUpProbability,
  };
}

/**
 * Activate the network, record output history, and choose the next direction.
 *
 * @param state - Mutable simulation state for the active run.
 * @param network - Policy network used for the current step.
 */
export function decideMazeMovementDirection(
  state: SimulationState,
  network: INetwork,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  try {
    const networkOutputs = network.activate(state.vision);
    const outputsLength = networkOutputs.length | 0;
    const outputsHistoryCopy: number[] = new Array(outputsLength);
    for (let copyIndex = 0; copyIndex < outputsLength; copyIndex++) {
      outputsHistoryCopy[copyIndex] = networkOutputs[copyIndex];
    }

    const previousHistory = readMazeMovementOutputHistory(network);
    const updatedHistory = MazeUtils.pushHistory(
      previousHistory,
      outputsHistoryCopy,
      C.OUTPUT_HISTORY_LENGTH,
    );
    writeMazeMovementOutputHistory(network, updatedHistory);

    const selectedActionStats = selectMazeMovementDirection(networkOutputs);
    state.actionStats = selectedActionStats;
    applyMazeMovementSaturationAndBiasAdjust(
      state,
      networkOutputs,
      network,
      coordinateScratch,
    );
    state.direction = selectedActionStats.direction;
  } catch (error: unknown) {
    console.error('Error activating network:', error);
    state.direction = C.NO_MOVE;
  }
}

/**
 * Apply the short-horizon proximity-greedy override near the maze exit.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for move validity checks.
 * @param distanceMap - Optional precomputed distance map.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementProximityGreedy(
  state: SimulationState,
  encodedMaze: number[][],
  distanceMap: number[][] | undefined,
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate || state.distHere > C.PROXIMITY_GREEDY_DISTANCE) {
    return;
  }

  let chosenDirection = state.direction;
  let minimalNeighborDistance = Infinity;

  for (
    let directionIndex = 0;
    directionIndex < C.ACTION_DIM;
    directionIndex++
  ) {
    const [deltaX, deltaY] = C.DIRECTION_DELTAS[directionIndex];
    const neighbourX = (state.position[0] + deltaX) | 0;
    const neighbourY = (state.position[1] + deltaY) | 0;

    coordinateScratch[0] = neighbourX;
    coordinateScratch[1] = neighbourY;

    if (
      !isMazeMovementCellOpen(
        encodedMaze,
        neighbourX,
        neighbourY,
        coordinateScratch,
      )
    ) {
      continue;
    }

    const neighbourDistance = getMazeMovementDistance(
      encodedMaze,
      [neighbourX, neighbourY],
      distanceMap,
    );
    if (neighbourDistance < minimalNeighborDistance) {
      minimalNeighborDistance = neighbourDistance;
      chosenDirection = directionIndex;
    }
  }

  if (chosenDirection !== undefined && chosenDirection !== state.direction) {
    state.direction = chosenDirection;
  }
}

/**
 * Apply epsilon-greedy exploration to the current action choice.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for move validity checks.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementEpsilonExploration(
  state: SimulationState,
  encodedMaze: number[][],
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  const epsilon = computeMazeMovementEpsilon(
    state.steps,
    state.stepsSinceImprovement,
    state.distHere,
    getMazeMovementRunServiceState().saturations,
  );
  if (!(randomMazeMovementUnit() < epsilon)) return;

  const actionCount = C.ACTION_DIM;
  const previousAction = state.prevAction;
  const currentPositionX = state.position[0] | 0;
  const currentPositionY = state.position[1] | 0;

  for (let attemptIndex = 0; attemptIndex < actionCount; attemptIndex++) {
    const randomDirection = (randomMazeMovementUnit() * actionCount) | 0;
    if (randomDirection === previousAction) continue;

    const [directionDeltaX, directionDeltaY] =
      C.DIRECTION_DELTAS[randomDirection];
    const candidateX = (currentPositionX + directionDeltaX) | 0;
    const candidateY = (currentPositionY + directionDeltaY) | 0;
    coordinateScratch[0] = candidateX;
    coordinateScratch[1] = candidateY;

    if (
      isMazeMovementCellOpen(
        encodedMaze,
        candidateX,
        candidateY,
        coordinateScratch,
      )
    ) {
      state.direction = randomDirection;
      break;
    }
  }
}

/**
 * Force a random valid move when the policy has stalled with repeated no-move outputs.
 *
 * @param state - Mutable simulation state for the active run.
 * @param encodedMaze - Maze grid used for move validity checks.
 * @param coordinateScratch - Reused coordinate scratch buffer.
 */
export function applyMazeMovementForcedExploration(
  state: SimulationState,
  encodedMaze: number[][],
  coordinateScratch: Int32Array,
): void {
  if (state.earlyTerminate) return;

  const runServices = getMazeMovementRunServiceState();
  if (state.direction === C.NO_MOVE) {
    runServices.noMoveStreak++;
  } else {
    runServices.noMoveStreak = 0;
  }

  if (runServices.noMoveStreak < C.NO_MOVE_STREAK_THRESHOLD) return;

  const actionCount = C.ACTION_DIM;
  const currentPositionX = state.position[0] | 0;
  const currentPositionY = state.position[1] | 0;

  for (let attemptIndex = 0; attemptIndex < actionCount; attemptIndex++) {
    const candidateDirection = (randomMazeMovementUnit() * actionCount) | 0;
    const [deltaX, deltaY] = C.DIRECTION_DELTAS[candidateDirection];
    const candidateX = (currentPositionX + deltaX) | 0;
    const candidateY = (currentPositionY + deltaY) | 0;
    coordinateScratch[0] = candidateX;
    coordinateScratch[1] = candidateY;

    if (
      isMazeMovementCellOpen(
        encodedMaze,
        candidateX,
        candidateY,
        coordinateScratch,
      )
    ) {
      state.direction = candidateDirection;
      break;
    }
  }

  runServices.noMoveStreak = 0;
}

/**
 * Detect saturation and optionally damp output-node biases.
 *
 * @param state - Mutable simulation state for the active run.
 * @param outputs - Raw network logits for the current step.
 * @param network - Policy network that produced the logits.
 * @param coordinateScratch - Reused scratch buffer for temporary penalties.
 */
export function applyMazeMovementSaturationAndBiasAdjust(
  state: SimulationState,
  outputs: number[],
  network: INetwork,
  coordinateScratch: Int32Array,
): void {
  const rewardScale = C.REWARD_SCALE;
  const actionStats = state.actionStats;
  if (!actionStats) return;

  const maxProbability = actionStats.maxProb ?? 0;
  const secondProbability = actionStats.secondProb ?? 0;
  const isOverConfident =
    maxProbability > C.OVERCONFIDENT_PROB &&
    secondProbability < C.SECOND_PROB_LOW;

  const actionCount = C.ACTION_DIM;
  let logitsSum = 0;
  for (let outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
    logitsSum += outputs[outputIndex];
  }
  const meanLogit = logitsSum / actionCount;

  let varianceAccumulator = 0;
  for (let outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
    const delta = outputs[outputIndex] - meanLogit;
    varianceAccumulator += delta * delta;
  }
  const variance = varianceAccumulator / actionCount;
  const standardDeviation = Math.sqrt(variance);
  const isFlatCollapsed = standardDeviation < C.LOGSTD_FLAT_THRESHOLD;

  const runServices = getMazeMovementRunServiceState();
  let saturationCounter = runServices.saturations;
  if (isOverConfident || isFlatCollapsed) {
    saturationCounter++;
    state.saturatedSteps++;
  } else if (saturationCounter > 0) {
    saturationCounter--;
  }
  runServices.saturations = saturationCounter;

  if (isOverConfident) {
    coordinateScratch[0] = -C.OVERCONFIDENT_PENALTY * rewardScale;
    state.invalidMovePenalty += coordinateScratch[0];
  }
  if (isFlatCollapsed) {
    coordinateScratch[0] = -C.FLAT_COLLAPSE_PENALTY * rewardScale;
    state.invalidMovePenalty += coordinateScratch[0];
  }

  const shouldAdjustBiases =
    runServices.saturations > C.SATURATION_ADJUST_MIN &&
    state.steps % C.SATURATION_ADJUST_INTERVAL === 0;
  if (!shouldAdjustBiases) return;

  try {
    const outputNodes = network.nodes?.filter(
      (node: INodeStruct): node is INodeStruct & { bias: number } =>
        node.type === C.NODE_TYPE_OUTPUT && typeof node.bias === 'number',
    );
    if (!outputNodes || outputNodes.length === 0) return;

    let biasSum = 0;
    for (let outputIndex = 0; outputIndex < outputNodes.length; outputIndex++) {
      biasSum += outputNodes[outputIndex].bias;
    }
    const meanBias = biasSum / outputNodes.length;
    for (let outputIndex = 0; outputIndex < outputNodes.length; outputIndex++) {
      const outputNode = outputNodes[outputIndex];
      const adjustedBias = outputNode.bias - meanBias * C.BIAS_ADJUST_FACTOR;
      outputNode.bias = Math.max(
        -C.BIAS_CLAMP,
        Math.min(C.BIAS_CLAMP, adjustedBias),
      );
    }
  } catch {
    // Best-effort only; network shapes vary across tests and demos.
  }
}

export {};
