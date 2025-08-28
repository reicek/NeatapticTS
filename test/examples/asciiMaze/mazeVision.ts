/**
 * MazeVision — agent sensory preprocessing for the ASCII maze examples.
 *
 * Produces a 6-element vector consumed by the agent's neural network:
 * [compassScalar, openN, openE, openS, openW, progressDelta]
 *
 * Implementation notes:
 * - Uses private static constants and helpers per STYLEGUIDE.
 * - Uses small typed arrays internally to minimize allocations on hot paths.
 */
export class MazeVision {
  /**
   * Direction table mapping cardinal direction to [dx, dy, index].
   *
   * Each entry is a tuple where the first two values are the X/Y delta to
   * reach the neighbor and the third value is the canonical direction index
   * used across the class (0 = N, 1 = E, 2 = S, 3 = W).
   * @type {readonly [number, number, number][]}
   */
  static #DIRECTION_DELTAS: readonly [number, number, number][] = [
    [0, -1, 0], // N
    [1, 0, 1], // E
    [0, 1, 2], // S
    [-1, 0, 3], // W
  ];

  // Tunable private constants
  /** Number of cardinal directions (N, E, S, W). */
  static #DIRECTION_COUNT = 4;
  /**
   * Scalar step per compass direction used to encode the compass into a single
   * continuous value: 0 = N, 0.25 = E, 0.5 = S, 0.75 = W.
   */
  static #COMPASS_STEP = 0.25;
  /** Offset to compute the opposite direction (half of direction count). */
  static #OPPOSITE_OFFSET = 2;
  /** Horizon (max path length) at which a neighbor is considered "open". */
  static #OPENNESS_HORIZON = 1000;
  /** Horizon used when selecting compass preference from a distance map. */
  static #COMPASS_HORIZON = 5000;
  /** Small scalar used to encourage backtracking from dead-ends. */
  static #BACKTRACK_SIGNAL = 0.001;
  /** Absolute clip applied to step-delta used when computing progress signal. */
  static #PROGRESS_CLIP = 2;
  /** Scale applied to clipped progress delta before adding to neutral baseline. */
  static #PROGRESS_SCALE = 4;
  /** Neutral progress value returned when progress cannot be computed reliably. */
  static #PROGRESS_NEUTRAL = 0.5;

  /**
   * Pooled scratch buffers reused across `buildInputs6` invocations to avoid
   * per-call allocations in hot paths. These are class-private and must be
   * initialized (via `fill`) at the start of each call.
   *
   * IMPORTANT: Because these buffers are reused, `buildInputs6` is not
   * reentrant and callers should not rely on the buffers' contents after
   * calling the method. See `buildInputs6` JSDoc `@remarks` for details.
   */
  static #SCRATCH_NEIGHBOR_X = new Int32Array(MazeVision.#DIRECTION_COUNT);
  static #SCRATCH_NEIGHBOR_Y = new Int32Array(MazeVision.#DIRECTION_COUNT);
  static #SCRATCH_NEIGHBOR_PATH = new Float32Array(MazeVision.#DIRECTION_COUNT);
  static #SCRATCH_NEIGHBOR_REACH = new Uint8Array(MazeVision.#DIRECTION_COUNT);
  static #SCRATCH_NEIGHBOR_OPEN = new Float32Array(MazeVision.#DIRECTION_COUNT);
  // Raw distance to exit per neighbor (NaN when not present). Pooled to avoid
  // per-call allocation; follows same non-reentrancy warning as other buffers.
  /**
   * Pooled buffer holding the raw distance-to-exit for each neighbor when a
   * `distanceToExitMap` is provided. Values are `NaN` when missing.
   */
  static #SCRATCH_NEIGHBOR_RAWDIST = new Float32Array(
    MazeVision.#DIRECTION_COUNT
  );

  // Small helpers
  /**
   * Return a neutral 6-element input vector used when inputs are invalid or
   * incomplete. The progress element is set to the neutral baseline.
   * @returns {[number, number, number, number, number, number]}
   *  [compassScalar, openN, openE, openS, openW, progressDelta]
   */
  static #neutralInput(): number[] {
    return [0, 0, 0, 0, 0, MazeVision.#PROGRESS_NEUTRAL];
  }

  /**
   * Fast bounds check for a 2D grid.
   * @param grid - 2D numeric grid where each row is an array.
   * @param col - X coordinate (column index).
   * @param row - Y coordinate (row index).
   * @returns True when the coordinate is within the grid bounds.
   */
  static #isWithinBounds(grid: number[][], col: number, row: number): boolean {
    return (
      Array.isArray(grid) &&
      row >= 0 &&
      row < grid.length &&
      Array.isArray(grid[row]) &&
      col >= 0 &&
      col < grid[row].length
    );
  }

  /**
   * Return true if the cell at [col,row] is within bounds and not a wall.
   * @param grid - Encoded maze where `-1` represents a wall.
   * @param col - Column (x) coordinate.
   * @param row - Row (y) coordinate.
   * @returns True when the cell exists and is open (not -1).
   */
  static #isCellOpen(grid: number[][], col: number, row: number): boolean {
    return MazeVision.#isWithinBounds(grid, col, row) && grid[row][col] !== -1;
  }

  /**
   * Compute the opposite cardinal direction index.
   * @param direction - Direction index in range [0, #DIRECTION_COUNT).
   * @returns Opposite direction index (e.g. 0 -> 2, 1 -> 3).
   */
  static #opposite(direction: number) {
    return (
      (direction + MazeVision.#OPPOSITE_OFFSET) % MazeVision.#DIRECTION_COUNT
    );
  }

  /**
   * Build the 6-element input vector consumed by the agent's network.
   *
   * The returned array has the shape: [compassScalar, openN, openE, openS, openW, progressDelta].
   *
   * @param encodedMaze - 2D grid where `-1` is a wall and `0+` is free space.
   * @param agentPosition - Agent coordinates as `[x, y]`.
   * @param exitPosition - Exit coordinates as `[x, y]` used as geometric fallback for compass.
   * @param distanceToExitMap - Optional distance-to-exit 2D map (same shape as `encodedMaze`).
   * @param previousStepDistance - Optional scalar distance-to-exit from the previous step.
   * @param currentStepDistance - Scalar distance-to-exit for the current step.
   * @param previousAction - Optional previous action index (0=N,1=E,2=S,3=W) to encourage backtracking.
   * @returns A 6-element number array with the vision inputs described above.
   *
   * @remarks
   * - This method reuses internal pooled scratch buffers (`#SCRATCH_*`) to avoid
   *   allocations on hot paths. Because the buffers are reused, the method is
   *   not reentrant and should not be called concurrently from multiple
   *   contexts if they share the same process/thread.
   * - Scratch buffers are initialized (via `.fill`) on each call so no state is
   *   leaked between invocations.
   */
  static buildInputs6(
    encodedMaze: number[][],
    agentPosition: readonly [number, number],
    exitPosition: readonly [number, number],
    distanceToExitMap: number[][] | undefined,
    previousStepDistance: number | undefined,
    currentStepDistance: number,
    previousAction: number | undefined
  ): number[] {
    // Step 0: Basic validation of inputs. Return a neutral vector when inputs
    // cannot be interpreted reliably (this keeps callers simple and avoids
    // throwing in tutorial/demo code paths).
    if (!Array.isArray(encodedMaze) || encodedMaze.length === 0)
      return MazeVision.#neutralInput();
    const [agentX, agentY] = agentPosition;
    if (!Number.isFinite(agentX) || !Number.isFinite(agentY))
      return MazeVision.#neutralInput();
    if (!MazeVision.#isWithinBounds(encodedMaze, agentX, agentY))
      return MazeVision.#neutralInput();

    // Step 1: Local symbolic constants. These provide readable names for
    // direction indices used throughout the function and a local alias for the
    // number of directions to avoid repeated private-field access inside loops.
    /** Direction index: North (0). */
    const DIR_N = 0;
    /** Direction index: East (1). */
    const DIR_E = 1;
    /** Direction index: South (2). */
    const DIR_S = 2;
    /** Direction index: West (3). */
    const DIR_W = 3;
    /** Local alias for `#DIRECTION_COUNT` to clarify intent and avoid repeated private-field access. */
    const D_COUNT = MazeVision.#DIRECTION_COUNT;

    // Step 2: Acquire pooled scratch buffers. These are class-private
    // Float32/Int32/Uint8 arrays that are reused to eliminate per-call heap
    // allocations. They are zeroed/initialized below before use.
    const neighborX = MazeVision.#SCRATCH_NEIGHBOR_X;
    const neighborY = MazeVision.#SCRATCH_NEIGHBOR_Y;
    const neighborPath = MazeVision.#SCRATCH_NEIGHBOR_PATH;
    const neighborReach = MazeVision.#SCRATCH_NEIGHBOR_REACH;
    const neighborOpen = MazeVision.#SCRATCH_NEIGHBOR_OPEN;

    // Cache hot lookups locally to avoid repeated private-field/property access
    const directionDeltas = MazeVision.#DIRECTION_DELTAS;
    const distMap = distanceToExitMap;

    // Step 3: Initialize scratch buffers for this invocation. We use
    // `.fill` to ensure no state from previous calls leaks into this call.
    neighborPath.fill(Infinity);
    neighborReach.fill(0);
    neighborOpen.fill(0);

    const currentCellDist = Number.isFinite(distMap?.[agentY]?.[agentX])
      ? distMap![agentY][agentX]
      : undefined;
    const hasCurrentCellDist =
      currentCellDist != null && Number.isFinite(currentCellDist);

    // Step 4: Gather neighbor info for each cardinal direction.
    // For each neighbor we compute:
    //  - neighborX/Y: coordinates
    //  - neighborReach: whether the neighbor cell is traversable (0/1)
    //  - neighborRaw: raw distance-to-exit from the optional dist map (NaN when missing)
    //  - neighborPath: finite path length when neighbor improves toward the exit
    //  - neighborOpen: preliminary openness metric (filled later)
    const neighborRaw = MazeVision.#SCRATCH_NEIGHBOR_RAWDIST;
    // initialize raw-dist buffer to NaN (represents missing)
    neighborRaw.fill(NaN);
    for (let d = 0; d < D_COUNT; d++) {
      const delta = directionDeltas[d];
      const deltaX = delta[0];
      const deltaY = delta[1];
      const directionIndex = delta[2];

      const neighborCol = agentX + deltaX;
      const neighborRow = agentY + deltaY;
      neighborX[directionIndex] = neighborCol;
      neighborY[directionIndex] = neighborRow;

      // Fast inline bounds/open check: reading the row reference directly is
      // much cheaper than calling helper functions repeatedly on hot paths.
      const neighborRowRef = encodedMaze[neighborRow];
      if (!neighborRowRef || neighborRowRef[neighborCol] === -1) {
        neighborPath[directionIndex] = Infinity;
        neighborReach[directionIndex] = 0;
        neighborOpen[directionIndex] = 0;
        neighborRaw[directionIndex] = NaN;
        continue;
      }

      // Cache the raw distance row reference if present to avoid chained lookups
      const distRow = distMap && distMap[neighborRow];
      const rawDistance = distRow ? distRow[neighborCol] : undefined;
      neighborRaw[directionIndex] = Number.isFinite(rawDistance)
        ? (rawDistance as number)
        : NaN;

      const hasValidDistance =
        rawDistance != null && Number.isFinite(rawDistance);
      neighborReach[directionIndex] = 1;
      if (
        hasValidDistance &&
        hasCurrentCellDist &&
        rawDistance! < currentCellDist!
      ) {
        const pathLength = 1 + (rawDistance as number);
        if (pathLength <= MazeVision.#OPENNESS_HORIZON) {
          // Path is within openness horizon. If it still exceeds the
          // compass horizon, mark openness as neutral so very long but
          // improving paths aren't treated as closed.
          neighborPath[directionIndex] = pathLength;
          neighborOpen[directionIndex] =
            pathLength > MazeVision.#COMPASS_HORIZON
              ? MazeVision.#PROGRESS_NEUTRAL
              : 0;
        } else {
          // Path beyond openness horizon: don't mark as closed (0);
          // use neutral signal so the agent doesn't incorrectly assume
          // no information for very long paths.
          neighborPath[directionIndex] = Infinity;
          neighborOpen[directionIndex] = MazeVision.#PROGRESS_NEUTRAL;
        }
      } else {
        neighborPath[directionIndex] = Infinity;
        neighborOpen[directionIndex] = 0;
      }
    }

    // Step 5: Compute openness values. We treat the smallest finite path as
    // the most "open" (1.0) and scale other finite paths down relative to it.
    // Cells without a finite path remain 0.
    let minPath = Infinity;
    for (let directionIndex = 0; directionIndex < D_COUNT; directionIndex++) {
      if (
        neighborReach[directionIndex] &&
        Number.isFinite(neighborPath[directionIndex]) &&
        neighborPath[directionIndex] < minPath
      )
        minPath = neighborPath[directionIndex];
    }
    if (minPath < Infinity) {
      for (let directionIndex = 0; directionIndex < D_COUNT; directionIndex++) {
        if (
          neighborReach[directionIndex] &&
          Number.isFinite(neighborPath[directionIndex])
        ) {
          neighborOpen[directionIndex] =
            neighborPath[directionIndex] === minPath
              ? 1
              : minPath / neighborPath[directionIndex];
        }
      }
    }

    // Expose named open values for clarity below.
    let openN = neighborOpen[DIR_N];
    let openE = neighborOpen[DIR_E];
    let openS = neighborOpen[DIR_S];
    let openW = neighborOpen[DIR_W];

    // Step 6: Dead-end backtrack encouragement. If all four openness values
    // are zero we encourage the agent to return the way it came by exposing a
    // tiny BACKTRACK_SIGNAL value on the opposite direction of the previous
    // action. This nudges learning away from getting stuck.
    if (
      openN === 0 &&
      openE === 0 &&
      openS === 0 &&
      openW === 0 &&
      previousAction != null
    ) {
      const oppositeDirection = MazeVision.#opposite(previousAction);
      switch (oppositeDirection) {
        case DIR_N:
          if (MazeVision.#isCellOpen(encodedMaze, agentX, agentY - 1))
            openN = MazeVision.#BACKTRACK_SIGNAL;
          break;
        case DIR_E:
          if (MazeVision.#isCellOpen(encodedMaze, agentX + 1, agentY))
            openE = MazeVision.#BACKTRACK_SIGNAL;
          break;
        case DIR_S:
          if (MazeVision.#isCellOpen(encodedMaze, agentX, agentY + 1))
            openS = MazeVision.#BACKTRACK_SIGNAL;
          break;
        case DIR_W:
          if (MazeVision.#isCellOpen(encodedMaze, agentX - 1, agentY))
            openW = MazeVision.#BACKTRACK_SIGNAL;
          break;
      }
    }

    // Step 7: Compass selection. Prefer the neighbor with the shortest cached
    // raw path to exit (if the `distanceToExitMap` is available). If no
    // suitable neighbor is found we fall back to a geometric heuristic that
    // points roughly toward the exit.
    let bestDirection = 0;
    if (distanceToExitMap) {
      let minCompassPathLength = Infinity;
      let found = false;
      for (let directionIndex = 0; directionIndex < D_COUNT; directionIndex++) {
        const neighborCachedRaw = neighborRaw[directionIndex];
        if (Number.isFinite(neighborCachedRaw)) {
          const pathLength = neighborCachedRaw + 1;
          if (
            pathLength < minCompassPathLength &&
            pathLength <= MazeVision.#COMPASS_HORIZON
          ) {
            minCompassPathLength = pathLength;
            bestDirection = directionIndex;
            found = true;
          }
        }
      }
      if (!found) {
        const deltaToExitX = exitPosition[0] - agentX;
        const deltaToExitY = exitPosition[1] - agentY;
        bestDirection =
          Math.abs(deltaToExitX) > Math.abs(deltaToExitY)
            ? deltaToExitX > 0
              ? 1
              : 3
            : deltaToExitY > 0
            ? 2
            : 0;
      }
    } else {
      const deltaToExitX = exitPosition[0] - agentX;
      const deltaToExitY = exitPosition[1] - agentY;
      bestDirection =
        Math.abs(deltaToExitX) > Math.abs(deltaToExitY)
          ? deltaToExitX > 0
            ? 1
            : 3
          : deltaToExitY > 0
          ? 2
          : 0;
    }
    const compassScalar = bestDirection * MazeVision.#COMPASS_STEP;

    // Step 8: Progress delta mapping. Map the change in distance-to-exit from
    // the previous step to a bounded progress signal centered at PROGRESS_NEUTRAL.
    let progress = MazeVision.#PROGRESS_NEUTRAL;
    if (
      previousStepDistance != null &&
      Number.isFinite(previousStepDistance) &&
      Number.isFinite(currentStepDistance)
    ) {
      const delta = previousStepDistance - currentStepDistance;
      const clipped = Math.max(
        -MazeVision.#PROGRESS_CLIP,
        Math.min(MazeVision.#PROGRESS_CLIP, delta)
      );
      progress =
        MazeVision.#PROGRESS_NEUTRAL + clipped / MazeVision.#PROGRESS_SCALE;
    }

    // Step 9: Return the canonical 6-element vision vector consumed by the
    // agent's network. We return plain numbers (not shared buffers) to avoid
    // exposing pooled buffers to callers.
    return [compassScalar, openN, openE, openS, openW, progress];
  }
}

export default MazeVision;
