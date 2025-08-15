/**
 * @file Implements the agent's "vision" system, which processes the maze environment into numerical inputs for the neural network.
 *
 * @description
 * This module defines the `MazeVision` class, responsible for creating the sensory input vector that the agent's neural network
 * uses to make decisions. The vision system is sophisticated, providing not just immediate surroundings but also long-range,
 * goal-oriented information. This rich sensory data is crucial for enabling the agent to learn complex navigation strategies.
 *
 * The vision system generates a 6-dimensional input vector:
 * 1.  **Compass Scalar**: A value in `{0, 0.25, 0.5, 0.75}` indicating the general direction of the exit (N, E, S, W).
 *     This acts as a global guide, always pointing towards the goal.
 * 2.  **Openness North**: A value from 0 to 1 representing the quality of the path starting with a move to the North.
 * 3.  **Openness East**: Path quality for East.
 * 4.  **Openness South**: Path quality for South.
 * 5.  **Openness West**: Path quality for West.
 * 6.  **Progress Delta**: A value indicating recent progress towards or away from the exit.
 *
 * The "Openness" values are calculated using a long-range lookahead based on a pre-computed `distanceMap`. A value of `1`
 * indicates the best possible path from the current location, while values less than `1` represent suboptimal but still viable paths.
 * A value of `0` signifies a wall, a dead end, or a path that moves away from the exit. This allows the network to make
 * informed decisions based on the long-term consequences of a move, rather than just immediate obstacles.
 */
export class MazeVision {
  /**
   * Constructs the 6-dimensional input vector for the neural network based on the agent's current state.
   *
   * @param encodedMaze - The 2D numerical representation of the maze.
   * @param position - The agent's current `[x, y]` coordinates.
   * @param exitPos - The coordinates of the maze exit.
   * @param distanceMap - A pre-calculated map of distances from each cell to the exit.
   * @param prevDistance - The agent's distance to the exit from the previous step.
   * @param currentDistance - The agent's current distance to the exit.
   * @param prevAction - The last action taken by the agent (0:N, 1:E, 2:S, 3:W).
   * @returns A 6-element array of numbers representing the network inputs.
   */
  static buildInputs6(
    encodedMaze: number[][],
    agentPosition: [number, number],
    exitPosition: [number, number],
    distanceToExitMap: number[][] | undefined,
    previousStepDistance: number | undefined,
    currentStepDistance: number,
    previousAction: number | undefined
  ): number[] {
    // --- Initialization ---
    /**
     * Agent's current X and Y coordinates
     */
    const [agentX, agentY] = agentPosition;
    /**
     * Number of rows in the maze
     */
    const mazeHeight = encodedMaze.length;
    /**
     * Number of columns in the maze
     */
    const mazeWidth = encodedMaze[0].length;
    /**
     * Checks if a coordinate is within maze bounds
     */
    const isWithinBounds = (col: number, row: number) =>
      row >= 0 && row < mazeHeight && col >= 0 && col < mazeWidth;
    /**
     * Checks if a cell is not a wall and within bounds
     */
    const isCellOpen = (col: number, row: number) =>
      isWithinBounds(col, row) && encodedMaze[row][col] !== -1;

    /**
     * Maximum path length considered viable for openness calculation
     */
    const opennessHorizon = 1000;
    /**
     * Maximum path length for compass guidance
     */
    const compassHorizon = 5000;

    // --- Neighbor Analysis ---
    /**
     * Array to store detailed information about each of the four adjacent cells
     */
    const neighborCells: {
      directionIndex: number;
      neighborX: number;
      neighborY: number;
      pathLength: number;
      isReachable: boolean;
      opennessValue: number;
    }[] = [];
    /**
     * Direction vectors and their indices: [dx, dy, dirIndex]
     * 0: North, 1: East, 2: South, 3: West
     */
    const DIRECTION_VECTORS: [number, number, number][] = [
      [0, -1, 0], // North
      [1, 0, 1], // East
      [0, 1, 2], // South
      [-1, 0, 3], // West
    ];

    /**
     * Current cell's distance to exit (from distance map, if available)
     */
    const currentCellDistanceToExit =
      distanceToExitMap && Number.isFinite(distanceToExitMap[agentY]?.[agentX])
        ? distanceToExitMap[agentY][agentX]
        : undefined;

    // Step 1: Gather information about each neighboring cell.
    for (const [dx, dy, directionIndex] of DIRECTION_VECTORS) {
      /**
       * Neighbor's coordinates
       */
      const neighborX = agentX + dx;
      const neighborY = agentY + dy;

      // If the neighbor is a wall, it's unreachable with a value of 0.
      if (!isCellOpen(neighborX, neighborY)) {
        neighborCells.push({
          directionIndex,
          neighborX,
          neighborY,
          pathLength: Infinity,
          isReachable: false,
          opennessValue: 0,
        });
        continue;
      }

      /**
       * Neighbor's distance to exit (from distance map, if available)
       */
      const neighborDistanceToExit = distanceToExitMap
        ? distanceToExitMap[neighborY]?.[neighborX]
        : undefined;

      // If the neighbor's distance to the exit is known and it's an improvement...
      if (
        neighborDistanceToExit != null &&
        Number.isFinite(neighborDistanceToExit) &&
        currentCellDistanceToExit != null &&
        Number.isFinite(currentCellDistanceToExit)
      ) {
        if (neighborDistanceToExit < currentCellDistanceToExit) {
          /**
           * Path length to exit if moving in this direction
           */
          const pathLength = 1 + neighborDistanceToExit;
          // If it's within the horizon, record its distance.
          if (pathLength <= opennessHorizon)
            neighborCells.push({
              directionIndex,
              neighborX,
              neighborY,
              pathLength,
              isReachable: true,
              opennessValue: 0,
            });
          // Otherwise, treat it as unreachable.
          else
            neighborCells.push({
              directionIndex,
              neighborX,
              neighborY,
              pathLength: Infinity,
              isReachable: true,
              opennessValue: 0,
            });
        } else {
          // Non-improving moves are treated as dead ends (value 0).
          neighborCells.push({
            directionIndex,
            neighborX,
            neighborY,
            pathLength: Infinity,
            isReachable: true,
            opennessValue: 0,
          });
        }
      } else {
        // If distances are unknown, treat as unreachable for now.
        neighborCells.push({
          directionIndex,
          neighborX,
          neighborY,
          pathLength: Infinity,
          isReachable: true,
          opennessValue: 0,
        });
      }
    }

    // Step 2: Calculate the "openness" values based on the best path.
    /**
     * All reachable neighbors with finite distance
     */
    const reachableNeighbors = neighborCells.filter(
      (neighbor) => neighbor.isReachable && Number.isFinite(neighbor.pathLength)
    );
    /**
     * Minimum path length among all neighbors
     */
    let minPathLength = Infinity;
    for (const neighbor of reachableNeighbors)
      if (neighbor.pathLength < minPathLength)
        minPathLength = neighbor.pathLength;

    // If there's at least one viable path forward...
    if (reachableNeighbors.length && minPathLength < Infinity) {
      for (const neighbor of reachableNeighbors) {
        // The best path(s) get a value of 1.
        if (neighbor.pathLength === minPathLength) neighbor.opennessValue = 1;
        // Other viable paths get a value proportional to how good they are.
        else neighbor.opennessValue = minPathLength / neighbor.pathLength;
      }
    }

    /**
     * Openness values for each direction (N, E, S, W)
     */
    let opennessNorth = neighborCells.find((n) => n.directionIndex === 0)!
      .opennessValue;
    let opennessEast = neighborCells.find((n) => n.directionIndex === 1)!
      .opennessValue;
    let opennessSouth = neighborCells.find((n) => n.directionIndex === 2)!
      .opennessValue;
    let opennessWest = neighborCells.find((n) => n.directionIndex === 3)!
      .opennessValue;

    // Step 3: Handle the "dead end" scenario.
    // If all forward paths are blocked, provide a small signal for the reverse direction
    // to encourage the agent to backtrack.
    if (
      opennessNorth === 0 &&
      opennessEast === 0 &&
      opennessSouth === 0 &&
      opennessWest === 0 &&
      previousAction != null &&
      previousAction >= 0
    ) {
      /**
       * Opposite direction to previous action
       */
      const oppositeDirection = (previousAction + 2) % 4;
      switch (oppositeDirection) {
        case 0:
          if (isCellOpen(agentX, agentY - 1)) opennessNorth = 0.001;
          break;
        case 1:
          if (isCellOpen(agentX + 1, agentY)) opennessEast = 0.001;
          break;
        case 2:
          if (isCellOpen(agentX, agentY + 1)) opennessSouth = 0.001;
          break;
        case 3:
          if (isCellOpen(agentX - 1, agentY)) opennessWest = 0.001;
          break;
      }
    }

    // Step 4: Calculate the compass scalar.
    // This points in the direction of the cell with the absolute shortest path to the exit,
    // even if it's very far away (using the extended compass horizon).
    /**
     * Best direction to the exit (0=N, 1=E, 2=S, 3=W)
     */
    let bestDirectionToExit = 0;
    if (distanceToExitMap) {
      /**
       * Minimum path length found for compass
       */
      let minCompassPathLength = Infinity;
      /**
       * Whether a valid path was found for compass
       */
      let foundCompassPath = false;
      for (const neighbor of neighborCells) {
        /**
         * Raw distance to exit for neighbor
         */
        const neighborRawDistance =
          distanceToExitMap[neighbor.neighborY]?.[neighbor.neighborX];
        if (
          neighborRawDistance != null &&
          Number.isFinite(neighborRawDistance)
        ) {
          const pathLength = neighborRawDistance + 1;
          if (
            pathLength < minCompassPathLength &&
            pathLength <= compassHorizon
          ) {
            minCompassPathLength = pathLength;
            bestDirectionToExit = neighbor.directionIndex;
            foundCompassPath = true;
          }
        }
      }
      // If no path is found via distance map, fall back to a simple geometric heuristic.
      if (!foundCompassPath) {
        /**
         * X and Y deltas to goal
         */
        const deltaXToGoal = exitPosition[0] - agentX;
        const deltaYToGoal = exitPosition[1] - agentY;
        if (Math.abs(deltaXToGoal) > Math.abs(deltaYToGoal))
          bestDirectionToExit = deltaXToGoal > 0 ? 1 : 3;
        else bestDirectionToExit = deltaYToGoal > 0 ? 2 : 0;
      }
    } else {
      // Fallback if no distance map is available.
      /**
       * X and Y deltas to goal
       */
      const deltaXToGoal = exitPosition[0] - agentX;
      const deltaYToGoal = exitPosition[1] - agentY;
      if (Math.abs(deltaXToGoal) > Math.abs(deltaYToGoal))
        bestDirectionToExit = deltaXToGoal > 0 ? 1 : 3;
      else bestDirectionToExit = deltaYToGoal > 0 ? 2 : 0;
    }
    /**
     * Compass scalar (0=N, 0.25=E, 0.5=S, 0.75=W)
     */
    const compassScalar = bestDirectionToExit * 0.25;

    // Step 5: Calculate the progress delta.
    // This value is > 0.5 if the agent moved closer to the exit, < 0.5 if it moved further away,
    // and 0.5 for no change.
    /**
     * Progress delta (recent progress toward/away from exit)
     */
    let progressDelta = 0.5;
    if (previousStepDistance != null && Number.isFinite(previousStepDistance)) {
      /**
       * Change in distance to exit since last step (clipped)
       */
      const distanceDelta = previousStepDistance - currentStepDistance;
      const clippedDelta = Math.max(-2, Math.min(2, distanceDelta)); // Clip to prevent extreme values.
      progressDelta = 0.5 + clippedDelta / 4;
    }

    // Step 6: Assemble and return the final input vector.
    /**
     * Final input vector for the neural network
     * [compassScalar, openN, openE, openS, openW, progressDelta]
     */
    const inputVector = [
      compassScalar,
      opennessNorth,
      opennessEast,
      opennessSouth,
      opennessWest,
      progressDelta,
    ];

    // Optional debug logging for educational/diagnostic purposes.
    // Prints a summary of the agent's vision and neighbor analysis every 5 calls if the environment variable is set.
    if (
      typeof process !== 'undefined' &&
      typeof process.env !== 'undefined' &&
      process.env.ASCII_VISION_DEBUG === '1'
    ) {
      try {
        /**
         * String summary of neighbor info for debugging, showing direction, coordinates, path length, and openness value.
         */
        const neighborSummary = neighborCells
          .map(
            (neighbor) =>
              `{dir:${neighbor.directionIndex} x:${neighbor.neighborX} y:${
                neighbor.neighborY
              } path:${
                Number.isFinite(neighbor.pathLength)
                  ? neighbor.pathLength.toFixed(2)
                  : 'Inf'
              } open:${neighbor.opennessValue.toFixed(4)}}`
          )
          .join(' ');
        // Internal debug counter to throttle log output
        (MazeVision as any)._dbgCounter =
          ((MazeVision as any)._dbgCounter || 0) + 1;
        if ((MazeVision as any)._dbgCounter % 5 === 0) {
          // Print a detailed summary of the agent's current vision state
          console.log(
            `[VISION] pos=${agentX},${agentY} comp=${compassScalar.toFixed(
              2
            )} inputs=${JSON.stringify(
              inputVector.map((v) => +v.toFixed(6))
            )} neighbors=${neighborSummary}`
          );
        }
      } catch {
        // Fail silently if any debug logic throws
      }
    }
    /**
     * Returns the 6-dimensional input vector for the agent's neural network:
     * [compassScalar, opennessNorth, opennessEast, opennessSouth, opennessWest, progressDelta]
     */
    return inputVector;
  }
}
export default MazeVision;
