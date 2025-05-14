/**
 * Maze Vision - Handles agent perception and vision-related functions (Minimalist)
 * 
 * This module contains functions for agent perception in the maze environment,
 * implementing a minimal set of sensory inputs for the neural network-controlled agent.
 * 
 * The minimalist approach focuses on:
 * - Detection of open paths in cardinal directions using DFS
 * - Basic exploration tracking
 * 
 * The DFS-based vision system is the main feature, allowing the agent to sense
 * open paths and make intelligent decisions about navigation.
 */


/**
 * Combines agent perception features into a single input vector for the neural network.
 *
 * This implementation now uses:
 * - Best direction encoding (0=N, 0.25=E, 0.5=S, 0.75=W) based on which vision input is highest
 * - Open paths in cardinal directions (N, E, S, W) using DFS
 *
 * Total: 1 + 4 = 5 inputs to the neural network
 *
 * The DFS-based vision system is the key feature that allows the agent to sense
 * open paths and make intelligent decisions about navigation.
 *
 * @param encodedMaze - 2D array representation of the maze
 * @param [agentX, agentY] - Current position of the agent
 * @param visitedPositions - Set of positions the agent has already visited
 * @param exitPos - Position of the exit/goal
 * @param globalProgress - (Unused) Kept for compatibility
 * @param visionRange - Optional parameter to specify the DFS range (default: 99999)
 * @returns Array of sensory inputs for the neural network
 */
export function getEnhancedVision(
  encodedMaze: number[][],
  [agentX, agentY]: [number, number],
  visitedPositions: Set<string>,
  exitPos?: [number, number],
  globalProgress?: number,
  visionRange?: number,
): number[] {
  const width = encodedMaze[0].length;
  const height = encodedMaze.length;

  // Cardinal directions: North, East, South, West
  const dirs: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];

  function isExit(x: number, y: number): boolean {
    if (!exitPos) return false;
    return x === exitPos[0] && y === exitPos[1];
  }

  function scanDirection(dx: number, dy: number): number {
    const visited = new Set<string>();
    let foundExit = false;
    let exitSteps = 0;
    const range = visionRange ?? 99999;
    function dfs(x: number, y: number, px: number, py: number, steps: number): number {
      if (steps > range) return 0;
      const key = `${x},${y}`;
      if (visited.has(key)) return 0;
      visited.add(key);
      if (isExit(x, y)) {
        foundExit = true;
        exitSteps = steps;
        return 1;
      }
      const isVisitedByAgent = visitedPositions.has(key);
      const open: [number, number][] = [];
      for (const [ndx, ndy] of dirs) {
        const nx = x + ndx;
        const ny = y + ndy;
        if (nx === px && ny === py) continue;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        if (encodedMaze[ny][nx] === -1) continue;
        open.push([ndx, ndy]);
      }
      if (open.length === 0) return 0;
      if (open.length === 1) {
        const [ndx, ndy] = open[0];
        const baseValue = dfs(x + ndx, y + ndy, x, y, steps + 1);
        return isVisitedByAgent ? baseValue * 0.95 : baseValue;
      }
      let splitValue = 0;
      for (const [ndx, ndy] of open) {
        splitValue += dfs(x + ndx, y + ndy, x, y, steps + 1);
      }
      if (!foundExit && splitValue > 0) {
        const junctionBonus = open.length * 0.001;
        const explorationFactor = isVisitedByAgent ? 0.98 : 1.02;
        return Math.min(0.49999, (splitValue + junctionBonus) * explorationFactor);
      }
      return splitValue;
    }
    const sx = agentX + dx;
    const sy = agentY + dy;
    if (sx < 0 || sx >= width || sy < 0 || sy >= height || encodedMaze[sy][sx] === -1) {
      return 0;
    }
    const value = dfs(sx, sy, agentX, agentY, 1);
    if (foundExit) {
      // Map steps to [0.50000, 0.99999], closer exits get higher value
      const norm = Math.max(1, Math.min(range, exitSteps));
      const mapped = 0.99999 - ((norm - 1) / (range - 1)) * 0.49999;
      return Math.max(0.5, Math.min(0.99999, mapped));
    }

    return Math.min(0.49999, value);
  }

  // Get openness in all directions using DFS
  const openDistances = dirs.map(([dx, dy]) => scanDirection(dx, dy));

  // Find the best direction (highest value among openDistances)
  let bestIdx = 0;
  let bestVal = openDistances[0];
  for (let i = 1; i < openDistances.length; i++) {
    if (openDistances[i] > bestVal) {
      bestVal = openDistances[i];
      bestIdx = i;
    }
  }
  // Encode as 0 (N), 0.25 (E), 0.5 (S), 0.75 (W)
  const bestDirectionEncoding = bestIdx * 0.25;

  // Assemble complete vision vector: [bestDirectionEncoding, openN, openE, openS, openW]
  return [
    bestDirectionEncoding,
    ...openDistances
  ];
}