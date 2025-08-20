/**
 * Maze Movement - Handles agent movement and simulation logic (Simplified)
 *
 * This module contains functions for agent movement and simulation in the maze environment,
 * focusing on simple navigation based primarily on neural network decisions.
 *
 * The agent movement system demonstrates:
 * - Decision making based on neural network outputs
 * - Basic reward calculations for reinforcement learning
 * - Simple goal-seeking behavior
 * - Simulation of movement with collision detection
 */

import { INetwork } from './interfaces';
import { MazeUtils } from './mazeUtils';
import { MazeVision } from './mazeVision';

/**
 * MazeMovement provides static methods for agent movement and simulation.
 */
export class MazeMovement {
  /**
   * Checks if a move is valid (within bounds and not a wall).
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param [x, y] - Coordinates to check.
   * @returns Boolean indicating if the position is valid for movement.
   */
  /**
   * Checks if a move is valid (within maze bounds and not a wall cell).
   *
   * @param encodedMaze - 2D array representation of the maze (cells: -1=wall, 0+=open).
   * @param coords - [x, y] coordinates to check for validity.
   * @returns {boolean} True if the position is within bounds and not a wall.
   */
  static isValidMove(
    encodedMaze: number[][],
    [x, y]: [number, number]
  ): boolean {
    // Check boundaries and wall status
    return (
      x >= 0 &&
      y >= 0 &&
      y < encodedMaze.length &&
      x < encodedMaze[0].length &&
      encodedMaze[y][x] !== -1
    );
  }

  /**
   * Moves the agent in the given direction if possible, otherwise stays in place.
   *
   * Handles collision detection with walls and maze boundaries,
   * preventing the agent from making invalid moves.
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param position - Current [x,y] position of the agent.
   * @param direction - Direction index (0=North, 1=East, 2=South, 3=West).
   * @returns New position after movement, or original position if move was invalid.
   */
  /**
   * Moves the agent in the specified direction if the move is valid.
   *
   * Handles collision detection with walls and maze boundaries,
   * preventing the agent from making invalid moves.
   *
   * @param encodedMaze - 2D array representation of the maze.
   * @param position - Current [x, y] position of the agent.
   * @param direction - Direction index (0=North, 1=East, 2=South, 3=West, -1=No move).
   * @returns { [number, number] } New position after movement, or original position if move was invalid.
   */
  static moveAgent(
    encodedMaze: number[][],
    position: [number, number],
    direction: number
  ): [number, number] {
    // If direction is -1, do not move
    if (direction === -1) {
      return [...position] as [number, number];
    }
    // Copy current position
    /**
     * Next position candidate for the agent after moving
     */
    const nextPosition: [number, number] = [...position] as [number, number];
    // Update position based on direction
    switch (direction) {
      case 0: // North
        nextPosition[1] -= 1;
        break;
      case 1: // East
        nextPosition[0] += 1;
        break;
      case 2: // South
        nextPosition[1] += 1;
        break;
      case 3: // West
        nextPosition[0] -= 1;
        break;
    }
    // Check if the new position is valid
    if (MazeMovement.isValidMove(encodedMaze, nextPosition)) {
      return nextPosition;
    } else {
      // If invalid, stay in place
      return position;
    }
  }

  /**
   * Selects the direction with the highest output value from the neural network.
   * Applies softmax to interpret outputs as probabilities, then uses argmax.
   *
   * @param outputs - Array of output values from the neural network (length 4).
   * @returns Index of the highest output value (0=N, 1=E, 2=S, 3=W), or -1 for no movement.
   */
  /**
   * Selects the direction with the highest output value from the neural network.
   * Applies softmax to interpret outputs as probabilities, then uses argmax.
   * Also computes entropy and confidence statistics for analysis.
   *
   * @param outputs - Array of output values from the neural network (length 4).
   * @returns {object} Direction index, softmax probabilities, entropy, and confidence stats.
   */
  static selectDirection(
    outputs: number[]
  ): {
    direction: number;
    softmax: number[];
    entropy: number;
    maxProb: number;
    secondProb: number;
  } {
    // Handle invalid or missing outputs
    if (!outputs || outputs.length !== 4) {
      return {
        direction: -1,
        softmax: [0, 0, 0, 0],
        entropy: 0,
        maxProb: 0,
        secondProb: 0,
      };
    }
    // Center logits to prevent mean bias drift
    /**
     * Mean of the output logits
     */
    const mean = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4;
    /**
     * Variance of the outputs (for adaptive temperature)
     * @type {number}
     */
    let variance = 0;
    for (const o of outputs) variance += (o - mean) * (o - mean);
    variance /= 4;
    /**
     * Standard deviation of the outputs
     * @type {number}
     */
    let std = Math.sqrt(variance);
    if (!Number.isFinite(std) || std < 1e-6) std = 1e-6;
    // Centered logits (preserve scale for evolutionary signals)
    /**
     * Centered logits (mean subtracted)
     */
    const centered = outputs.map((o) => o - mean);
    // Adaptive temperature: higher if variance is tiny
    /**
     * Ratio for adaptive temperature (higher if variance is tiny)
     */
    const collapseRatio = std < 0.01 ? 1 : std < 0.03 ? 0.5 : 0;
    /**
     * Softmax temperature (adaptive)
     */
    const temperature = 1 + 1.2 * collapseRatio; // max 2.2
    // Softmax calculation
    /**
     * Maximum centered logit value
     */
    const max = Math.max(...centered);
    /**
     * Exponentiated logits for softmax
     */
    const exps = centered.map((v) => Math.exp((v - max) / temperature));
    /**
     * Sum of exponentiated logits (softmax denominator)
     */
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    /**
     * Softmax probability vector
     */
    const softmax = exps.map((e) => e / sum);
    // Find direction with highest probability
    let direction = 0;
    let maxProb = -Infinity;
    let secondProb = 0;
    softmax.forEach((p, i) => {
      if (p > maxProb) {
        secondProb = maxProb;
        maxProb = p;
        direction = i;
      } else if (p > secondProb) secondProb = p;
    });
    // Compute entropy (uncertainty measure)
    let entropy = 0;
    softmax.forEach((p) => {
      if (p > 0) entropy += -p * Math.log(p);
    });
    entropy /= Math.log(4); // Normalize to [0,1]
    return { direction, softmax, entropy, maxProb, secondProb };
  }

  /**
   * Simulates the agent navigating the maze using its neural network.
   *
   * Runs a complete simulation of an agent traversing a maze,
   * using its neural network for decision making. This implementation focuses
   * on a minimalist approach, putting more responsibility on the neural network.
   *
   * @param network - Neural network controlling the agent.
   * @param encodedMaze - 2D array representation of the maze.
   * @param startPos - Starting position [x,y] of the agent.
   * @param exitPos - Exit/goal position [x,y] of the maze.
   * @param maxSteps - Maximum steps allowed before terminating (default 3000).
   * @returns Object containing:
   *   - success: Boolean indicating if exit was reached.
   *   - steps: Number of steps taken.
   *   - path: Array of positions visited.
   *   - fitness: Calculated fitness score for evolution.
   *   - progress: Percentage progress toward exit (0-100).
   */
  static simulateAgent(
    network: INetwork,
    encodedMaze: number[][],
    startPos: [number, number],
    exitPos: [number, number],
    distanceMap?: number[][],
    maxSteps = 3000
  ): {
    success: boolean;
    steps: number;
    path: [number, number][];
    fitness: number;
    progress: number;
    saturationFraction?: number;
    actionEntropy?: number;
  } {
    /**
     * Current position of the agent [x, y]
     * @type {[number, number]}
     */
    let position = [...startPos] as [number, number];
    /**
     * Number of steps taken so far
     * @type {number}
     */
    let steps = 0;
    /**
     * Path of positions visited by the agent
     * @type {Array<[number, number]>}
     */
    let path = [position.slice() as [number, number]];
    /**
     * Set of visited positions (as string keys)
     * @type {Set<string>}
     */
    let visitedPositions = new Set<string>();
    /**
     * Map of visit counts per cell
     * @type {Map<string, number>}
     */
    let visitCounts = new Map<string, number>();
    /**
     * Short-term memory for last N positions (for loop/oscillation detection)
     * @type {string[]}
     */
    let moveHistory: string[] = [];
    /**
     * Number of positions to keep in moveHistory
     * @type {number}
     */
    const MOVE_HISTORY_LENGTH = 6;
    /**
     * Closest distance to exit found so far
     * @type {number}
     */
    let minDistanceToExit = distanceMap
      ? distanceMap[position[1]]?.[position[0]] ?? Infinity
      : MazeUtils.bfsDistance(encodedMaze, position, exitPos);

    /**
     * Reward scaling factor for all reward/penalty calculations
     * @type {number}
     */
    const rewardScale = 0.5;

    // Reward tracking variables
    /**
     * Accumulated reward for progress toward exit
     * @type {number}
     */
    let progressReward = 0;
    /**
     * Bonus for exploring new cells
     * @type {number}
     */
    let newCellExplorationBonus = 0;
    /**
     * Penalty for invalid moves or stagnation
     * @type {number}
     */
    let invalidMovePenalty = 0;

    // Memory and stagnation tracking
    /**
     * Last direction taken (0-3 or -1)
     * @type {number}
     */
    let prevAction = -1;
    /**
     * Steps since last improvement in distance to exit
     * @type {number}
     */
    let stepsSinceImprovement = 0;
    /**
     * Initial global distance to exit
     * @type {number}
     */
    const startDistanceGlobal = distanceMap
      ? distanceMap[position[1]]?.[position[0]] ?? Infinity
      : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
    /**
     * Last global distance to exit
     * @type {number}
     */
    let lastDistanceGlobal = startDistanceGlobal;
    /**
     * Number of saturated steps (output collapse or overconfidence)
     * @type {number}
     */
    let saturatedSteps = 0;
    /**
     * Window size for local area stagnation detection
     * @type {number}
     */
    const LOCAL_WINDOW = 30;
    /**
     * Recent positions for local area stagnation
     * @type {Array<[number, number]>}
     */
    const recentPositions: [number, number][] = [];
    /**
     * Penalty for local area stagnation
     * @type {number}
     */
    let localAreaPenalty = 0;

    // Main simulation loop: agent moves until maxSteps or exit is reached
    let lastProgressRatio = 0;
    while (steps < maxSteps) {
      steps++;

      // --- Step 1: Record current position as visited ---
      /**
       * String key for the agent's current position
       */
      const currentPosKey = `${position[0]},${position[1]}`;
      visitedPositions.add(currentPosKey);
      visitCounts.set(currentPosKey, (visitCounts.get(currentPosKey) || 0) + 1);
      moveHistory.push(currentPosKey);
      if (moveHistory.length > MOVE_HISTORY_LENGTH) moveHistory.shift();

      // --- Step 2: Calculate percent of maze explored so far ---
      /**
       * Percent of maze explored so far
       */
      const percentExplored =
        visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);

      // --- Step 3: Oscillation/loop detection (A->B->A->B) ---
      /**
       * Penalty for oscillation/looping
       */
      let loopPenalty = 0;
      if (
        moveHistory.length >= 4 &&
        moveHistory[moveHistory.length - 1] ===
          moveHistory[moveHistory.length - 3] &&
        moveHistory[moveHistory.length - 2] ===
          moveHistory[moveHistory.length - 4]
      ) {
        loopPenalty -= 10 * rewardScale; // Strong penalty for 2-step loop
      }
      /**
       * Memory loop indicator input (1 if loop detected, else 0)
       */
      const loopFlag = loopPenalty < 0 ? 1 : 0;

      // --- Step 4: Penalty for returning to any cell in recent history ---
      /**
       * Penalty for returning to a cell in recent history
       */
      let memoryPenalty = 0;
      if (
        moveHistory.length > 1 &&
        moveHistory.slice(0, -1).includes(currentPosKey)
      ) {
        memoryPenalty -= 2 * rewardScale;
      }

      // --- Step 5: Dynamic penalty for multiple visits ---
      /**
       * Penalty for revisiting cells
       */
      let revisitPenalty = 0;
      /**
       * Number of times the current cell has been visited
       */
      const visits = visitCounts.get(currentPosKey) || 1;
      if (visits > 1) {
        revisitPenalty -= 0.2 * (visits - 1) * rewardScale; // Penalty increases with each revisit
      }

      // --- Step 6: Early termination if a cell is visited too many times ---
      if (visits > 10) {
        invalidMovePenalty -= 1000 * rewardScale;
        break;
      }

      // --- Step 7: Build agent's perception (vision inputs) ---
      /**
       * Previous local distance to exit
       */
      const prevDistLocal = distanceMap
        ? distanceMap[position[1]]?.[position[0]] ?? undefined
        : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
      /**
       * Current local distance to exit (same as prev before move)
       */
      const distCurrentLocal = prevDistLocal;
      /**
       * Vision input vector for the agent
       */
      const vision = MazeVision.buildInputs6(
        encodedMaze,
        position,
        exitPos,
        distanceMap,
        (MazeMovement as any)._prevDistanceStep,
        distCurrentLocal,
        prevAction
      );
      (MazeMovement as any)._prevDistanceStep = distCurrentLocal;

      // --- Step 8: Get distance at current location (for proximity logic) ---
      /**
       * Distance at current location (pre-action) for proximity exploitation logic
       */
      const distHere = distanceMap
        ? distanceMap[position[1]]?.[position[0]] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, position, exitPos);

      // --- Step 9: Neural network decision making ---
      /**
       * Chosen direction for this step
       */
      let direction;
      /**
       * Action statistics (softmax, entropy, etc.)
       */
      let actionStats: any = null;
      try {
        // Activate the network with vision inputs
        /**
         * Output vector from the neural network
         */
        const outputs = network.activate(vision) as number[];
        // Track outputs for variance diagnostics (sliding window)
        (network as any)._lastStepOutputs =
          (network as any)._lastStepOutputs || [];
        /**
         * Sliding window of last step outputs for variance diagnostics
         */
        const _ls = (network as any)._lastStepOutputs;
        _ls.push(outputs.slice());
        if (_ls.length > 80) _ls.shift();
        // Select direction and compute stats
        actionStats = MazeMovement.selectDirection(outputs);
        // Detect output saturation (overconfidence or flat collapse)
        (MazeMovement as any)._saturations =
          (MazeMovement as any)._saturations || 0;
        const overConfident =
          actionStats.maxProb > 0.985 && actionStats.secondProb < 0.01;
        // Recompute std on centered logits
        const logitsMean =
          (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4;
        let logVar = 0;
        for (const o of outputs) logVar += Math.pow(o - logitsMean, 2);
        logVar /= 4;
        const logStd = Math.sqrt(logVar);
        const flatCollapsed = logStd < 0.01;
        const saturatedNow = overConfident || flatCollapsed;
        if (saturatedNow) {
          (MazeMovement as any)._saturations++;
          saturatedSteps++;
        } else {
          (MazeMovement as any)._saturations = Math.max(
            0,
            (MazeMovement as any)._saturations - 1
          );
        }
        // Penalties for saturation
        if (overConfident) invalidMovePenalty -= 0.25 * rewardScale;
        if (flatCollapsed) invalidMovePenalty -= 0.35 * rewardScale;
        // Adaptive bias anti-saturation: gently reduce output biases if chronic
        try {
          if ((MazeMovement as any)._saturations > 6 && steps % 5 === 0) {
            const outs = (network as any).nodes?.filter(
              (n: any) => n.type === 'output'
            );
            if (outs?.length) {
              const mean =
                outs.reduce((a: number, n: any) => a + n.bias, 0) / outs.length;
              outs.forEach((n: any) => {
                n.bias = Math.max(-5, Math.min(5, n.bias - mean * 0.5));
              });
            }
          }
        } catch {
          /* ignore */
        }
        direction = actionStats.direction;
      } catch (error) {
        console.error('Error activating network:', error);
        direction = -1; // Fallback: don't move
      }

      // --- Step 10: Proximity exploitation (greedy move if near exit) ---
      if (distHere <= 2) {
        /**
         * Best direction found (minimizing distance to exit)
         */
        let bestDir = direction;
        /**
         * Best distance found
         */
        let bestDist = Infinity;
        for (let d = 0; d < 4; d++) {
          const testPos = MazeMovement.moveAgent(encodedMaze, position, d);
          if (testPos[0] === position[0] && testPos[1] === position[1])
            continue; // invalid
          /**
           * Distance value for candidate direction
           */
          const dVal = distanceMap
            ? distanceMap[testPos[1]]?.[testPos[0]] ?? Infinity
            : MazeUtils.bfsDistance(encodedMaze, testPos, exitPos);
          if (dVal < bestDist) {
            bestDist = dVal;
            bestDir = d;
          }
        }
        if (bestDir != null) direction = bestDir;
      }

      // Epsilon-greedy exploration: encourage divergence early & when stagnant
      const stepsStagnant = stepsSinceImprovement;
      let epsilon = 0;
      if (steps < 10) epsilon = 0.35;
      else if (stepsStagnant > 12) epsilon = 0.5;
      else if (stepsStagnant > 6) epsilon = 0.25;
      else if ((MazeMovement as any)._saturations > 3) epsilon = 0.3;
      // Suppress exploration when near goal to encourage completion
      if (distHere <= 5) epsilon = Math.min(epsilon, 0.05);
      if (Math.random() < epsilon) {
        // pick a random valid direction differing from previous when possible
        /**
         * Candidate directions for random exploration
         */
        const candidates = [0, 1, 2, 3].filter((d) => d !== prevAction);
        while (candidates.length) {
          /**
           * Index of candidate direction
           */
          const idx = Math.floor(Math.random() * candidates.length);
          /**
           * Candidate direction value
           */
          const cand = candidates.splice(idx, 1)[0];
          /**
           * Test position for candidate direction
           */
          const testPos = MazeMovement.moveAgent(encodedMaze, position, cand);
          if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
            direction = cand;
            break;
          }
        }
      }

      // --- Anti-stagnation: if we haven't moved in several attempts, force exploratory move
      // Track consecutive failed moves
      (MazeMovement as any)._noMoveStreak =
        (MazeMovement as any)._noMoveStreak || 0;
      if (direction === -1) (MazeMovement as any)._noMoveStreak++;
      if ((MazeMovement as any)._noMoveStreak >= 5) {
        // pick a random cardinal direction until a valid move found (epsilon-greedy style)
        for (let tries = 0; tries < 4; tries++) {
          /**
           * Candidate direction for forced exploration
           */
          const cand = Math.floor(Math.random() * 4);
          /**
           * Test position for candidate direction
           */
          const testPos = MazeMovement.moveAgent(encodedMaze, position, cand);
          if (testPos[0] !== position[0] || testPos[1] !== position[1]) {
            direction = cand;
            break;
          }
        }
        (MazeMovement as any)._noMoveStreak = 0;
      }

      // Save previous state for reward calculation
      /**
       * Previous position before move
       */
      const prevPosition = [...position] as [number, number];
      /**
       * Previous distance to exit before move
       */
      const prevDistance = distanceMap
        ? distanceMap[position[1]]?.[position[0]] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, position, exitPos);

      // --- ACTION: Move based on network decision
      position = MazeMovement.moveAgent(encodedMaze, position, direction);
      /**
       * Whether the agent actually moved this step
       */
      const moved =
        prevPosition[0] !== position[0] || prevPosition[1] !== position[1];

      // Record movement and update rewards/penalties
      if (moved) {
        path.push(position.slice() as [number, number]);
        recentPositions.push(position.slice() as [number, number]);
        if (recentPositions.length > LOCAL_WINDOW) recentPositions.shift();
        if (recentPositions.length === LOCAL_WINDOW) {
          /**
           * Minimum and maximum X/Y in recent positions
           */
          let minX = Infinity,
            maxX = -Infinity,
            minY = Infinity,
            maxY = -Infinity;
          for (const [rx, ry] of recentPositions) {
            if (rx < minX) minX = rx;
            if (rx > maxX) maxX = rx;
            if (ry < minY) minY = ry;
            if (ry > maxY) maxY = ry;
          }
          /**
           * Span of recent positions (for local oscillation detection)
           */
          const span = maxX - minX + (maxY - minY);
          // Penalize tight oscillation in a small neighborhood when no improvements recently
          if (span <= 5 && stepsSinceImprovement > 8) {
            localAreaPenalty -= 0.05 * rewardScale; // accumulate gradually
          }
        }

        // Calculate current distance to exit
        /**
         * Current distance to exit after move
         */
        const currentDistance = distanceMap
          ? distanceMap[position[1]]?.[position[0]] ?? Infinity
          : MazeUtils.bfsDistance(encodedMaze, position, exitPos);

        // Reward for getting closer to exit, penalty for moving away
        /**
         * Change in distance to exit (positive if improved)
         */
        const distanceDelta = prevDistance - currentDistance; // positive if improved
        if (distanceDelta > 0) {
          // Confidence shaping if available
          const conf = actionStats?.maxProb ?? 1;
          progressReward += (0.3 + 0.7 * conf) * rewardScale;
          if (stepsSinceImprovement > 0)
            progressReward += Math.min(
              stepsSinceImprovement * 0.02 * rewardScale,
              0.5 * rewardScale
            );
          stepsSinceImprovement = 0;
          // Additional proportional reward to create gradient
          progressReward += distanceDelta * 2.0 * (0.4 + 0.6 * conf); // scale by confidence
        } else if (currentDistance > prevDistance) {
          const conf = actionStats?.maxProb ?? 0.5;
          progressReward -= (0.05 + 0.15 * conf) * rewardScale;
          stepsSinceImprovement++;
        } else {
          stepsSinceImprovement++;
        }

        // Bonus for exploring new cells, penalty for revisiting
        if (visits === 1) {
          newCellExplorationBonus += 0.3 * rewardScale;
        } else {
          newCellExplorationBonus -= 0.5 * rewardScale; // Stronger penalty for revisiting
        }

        // Track closest approach to exit
        minDistanceToExit = Math.min(minDistanceToExit, currentDistance);
      } else {
        // Penalty for invalid move (collision or out of bounds)
        // Previously this was extremely punitive (-1000 * scale) causing all genomes to bottom-out at the clamp
        // which destroyed selection pressure. Keep it mild so progress/exploration dominate.
        invalidMovePenalty -= 10 * rewardScale; // mild penalty now
        // No tolerance for invalid moves; break if needed
        steps === maxSteps;
      }
      // Update global distance improvement memory
      /**
       * Current global distance to exit
       */
      const currentDistanceGlobal = distanceMap
        ? distanceMap[position[1]]?.[position[0]] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, position, exitPos);
      if (currentDistanceGlobal < lastDistanceGlobal) {
        // bonus for breaking a long stagnation globally
        if (stepsSinceImprovement > 10)
          progressReward += Math.min(
            (stepsSinceImprovement - 10) * 0.01 * rewardScale,
            0.5 * rewardScale
          );
        stepsSinceImprovement = 0;
      }
      lastDistanceGlobal = currentDistanceGlobal;
      // Repetition penalty: if repeating same action without improvement
      if (prevAction === direction && stepsSinceImprovement > 4) {
        invalidMovePenalty -= 0.05 * (stepsSinceImprovement - 4) * rewardScale;
      }
      // Penalize backward (opposite) moves strongly if they do not improve
      if (prevAction >= 0 && direction >= 0) {
        /**
         * Opposite direction to previous action
         */
        const opposite = (prevAction + 2) % 4;
        if (direction === opposite && stepsSinceImprovement > 0) {
          invalidMovePenalty -= 0.2 * rewardScale;
        }
      }
      // Only record previous action if movement succeeded to avoid mismatches
      if (moved) {
        prevAction = direction; // record last successful move for back-direction suppression
        prevAction = direction;
      }

      // Encourage decisiveness: slight penalty for very high entropy (uniform outputs),
      // slight bonus for confident low-entropy when some guidance signal (gradient or LOS) exists.
      if (actionStats) {
        const { entropy, maxProb, secondProb } = actionStats;
        // Compute presence of directional guidance (any non-zero gradient or LOS)
        const hasGuidance =
          vision[8] + vision[9] + vision[10] + vision[11] > 0 || // LOS group
          vision[12] + vision[13] + vision[14] + vision[15] > 0; // Gradient group
        if (entropy > 0.95) {
          invalidMovePenalty -= 0.03 * rewardScale; // discourage persistent ambiguity
        } else if (
          hasGuidance &&
          entropy < 0.55 &&
          maxProb - secondProb > 0.25
        ) {
          newCellExplorationBonus += 0.015 * rewardScale; // tiny shaping bonus for clear decision
        }
        // Penalty for prolonged saturation (uninformative all-ones behavior)
        if ((MazeMovement as any)._saturations >= 5) {
          invalidMovePenalty -= 0.05 * rewardScale;
          if ((MazeMovement as any)._saturations % 10 === 0) {
            invalidMovePenalty -= 0.1 * rewardScale; // escalating every 10 steps saturated
          }
        }
      }

      // Early termination on deep stagnation (disabled for browser demo to allow full exploration)
      if (stepsSinceImprovement > 40) {
        try {
          if (typeof window === 'undefined') {
            invalidMovePenalty -= 2 * rewardScale;
            break; // keep for non-browser environments (tests / Node)
          }
        } catch {
          // if window check failed, proceed with default behavior
          invalidMovePenalty -= 2 * rewardScale;
          break;
        }
      }

      // Apply oscillation/loop/memory/revisit penalties
      invalidMovePenalty += loopPenalty + memoryPenalty + revisitPenalty;

      // --- SUCCESS CHECK: Exit reached
      if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
        // Calculate fitness for successful completion
        /**
         * Step efficiency (remaining steps)
         */
        const stepEfficiency = maxSteps - steps;
        // Action entropy bonus for successful runs (promote balanced yet decisive policies)
        /**
         * Action entropy for the successful path
         */
        const { actionEntropy } = MazeMovement.computeActionEntropy(path);
        /**
         * Fitness score for successful completion
         */
        const fitness =
          650 +
          stepEfficiency * 0.2 +
          progressReward +
          newCellExplorationBonus +
          invalidMovePenalty +
          actionEntropy * 5;

        return {
          success: true,
          steps,
          path,
          fitness: Math.max(150, fitness),
          progress: 100,
          saturationFraction: steps ? saturatedSteps / steps : 0,
          actionEntropy,
        };
      }
    }

    // --- FAILURE CASE: Did not reach exit
    /**
     * Progress percentage toward exit (0-100)
     */
    const progress = distanceMap
      ? MazeUtils.calculateProgressFromDistanceMap(
          distanceMap,
          path[path.length - 1],
          startPos
        )
      : MazeUtils.calculateProgress(
          encodedMaze,
          path[path.length - 1],
          startPos,
          exitPos
        );

    // Fitness for unsuccessful attempts: emphasize progress & exploration with moderated penalties
    /**
     * Fractional progress toward exit (0..1)
     */
    const progressFrac = progress / 100;
    /**
     * Shaped progress score (concave for early gradient)
     */
    const shapedProgress = Math.pow(progressFrac, 1.3) * 500;
    /**
     * Exploration score (number of unique cells visited)
     */
    const explorationScore = visitedPositions.size * 1.0; // increase weight so exploration differentiates genomes
    /**
     * Aggregated penalty
     */
    const penalty = invalidMovePenalty; // already aggregated
    // Action entropy based on path
    /**
     * Action entropy for the failed path
     */
    const { actionEntropy } = MazeMovement.computeActionEntropy(path);
    /**
     * Bonus for action entropy
     */
    const entropyBonus = actionEntropy * 4; // weight
    /**
     * Fraction of steps that were saturated
     */
    const satFrac = steps ? saturatedSteps / steps : 0;
    /**
     * Penalty for output saturation
     */
    const saturationPenalty =
      satFrac > 0.35
        ? -(satFrac - 0.35) * 40 // linear scale beyond threshold
        : 0;
    // Penalize persistently near-constant output vectors across steps (low std)
    /**
     * Penalty for low output variance
     */
    let outputVarPenalty = 0;
    try {
      /**
       * History of last step outputs
       */
      const hist: number[][] = (network as any)._lastStepOutputs || [];
      if (hist.length >= 15) {
        const recent = hist.slice(-30);
        let lowVar = 0;
        for (const o of recent) {
          const m = (o[0] + o[1] + o[2] + o[3]) / 4;
          let v = 0;
          for (const x of o) v += (x - m) * (x - m);
          v /= 4;
          if (Math.sqrt(v) < 0.01) lowVar++;
        }
        if (lowVar > 4) outputVarPenalty -= (lowVar - 4) * 0.3; // escalate with count beyond small tolerance
      }
    } catch {}
    // Near-miss penalty: strongly encourage finishing if within 1 step at any point
    /**
     * Penalty for being within 1 step of exit but not finishing
     */
    let nearMissPenalty = 0;
    if (minDistanceToExit === 1) nearMissPenalty -= 30 * rewardScale;
    /**
     * Base fitness score before final adjustment
     */
    const base =
      shapedProgress +
      explorationScore +
      progressReward +
      newCellExplorationBonus +
      penalty +
      entropyBonus +
      localAreaPenalty +
      saturationPenalty +
      outputVarPenalty +
      nearMissPenalty;
    // Remove tight clamp that caused saturation; apply gentle floor far lower so relative differences remain
    // Add tiny stochastic tie-breaker noise so identical behaviors diverge slightly for selection pressure
    // Replace static floor with softer nonlinear squash so early differentials aren't erased
    // and prevent population collapse at a shared floor.
    /**
     * Raw fitness score (with noise)
     */
    const raw = base + Math.random() * 0.01;
    /**
     * Final fitness score (nonlinear squash for negatives)
     */
    const fitness = raw >= 0 ? raw : -Math.log1p(1 - raw);
    return {
      success: false,
      steps,
      path,
      fitness,
      progress,
      saturationFraction: satFrac,
      actionEntropy,
    };
  }
}
/**
 * Computes the entropy of the agent's action distribution from its path.
 * Higher entropy means more diverse movement; lower means repetitive.
 *
 * @param path - Array of [x, y] positions visited by the agent.
 * @returns {object} actionEntropy (0..1)
 */
export namespace MazeMovement {
  export function computeActionEntropy(path: [number, number][]) {
    if (!path || path.length < 2) return { actionEntropy: 0 };
    /**
     * Counts of each direction taken (N, E, S, W)
     * @type {number[]}
     */
    const counts = [0, 0, 0, 0];
    for (let i = 1; i < path.length; i++) {
      const dx = path[i][0] - path[i - 1][0];
      const dy = path[i][1] - path[i - 1][1];
      if (dx === 0 && dy === -1) counts[0]++;
      // North
      else if (dx === 1 && dy === 0) counts[1]++;
      // East
      else if (dx === 0 && dy === 1) counts[2]++;
      // South
      else if (dx === -1 && dy === 0) counts[3]++; // West
    }
    /**
     * Total number of actions taken
     */
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    let ent = 0;
    counts.forEach((c) => {
      if (c > 0) {
        const p = c / total;
        ent += -p * Math.log(p);
      }
    });
    /**
     * Normalized entropy of the action distribution (0=deterministic, 1=uniform)
     * @type {number}
     */
    const actionEntropy = ent / Math.log(4);
    return { actionEntropy };
  }
}
