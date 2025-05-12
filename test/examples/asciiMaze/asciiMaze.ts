// asciiMaze.ts - Refactored for SOLID principles, maintainability, and readability

import { Network } from '../../../src/neataptic';
import { colors } from './colors';
import { encodeMaze, findPosition, manhattanDistance, calculateProgress } from './mazeUtils';

export { encodeMaze, findPosition, manhattanDistance, calculateProgress } from './mazeUtils';

// --- Constants ---
const MOVE_DIRECTIONS = 4;
const LOCAL_GRID_SIZE = 10; // Now 10x10 grid

// --- Encoding & Perception ---

/**
 * Encodes the normalized direction and distance from the agent to the exit.
 */
export function encodeDirectionToExit(
  agentX: number,
  agentY: number,
  exitPos: [number, number] | undefined,
  width: number,
  height: number
): number[] {
  if (!exitPos) return [0, 0, 0];
  const [exitX, exitY] = exitPos;
  const dx = (exitX - agentX) / width;
  const dy = (exitY - agentY) / height;
  const maxDistance = width + height;
  const distance = manhattanDistance([agentX, agentY], exitPos);
  return [dx, dy, distance / maxDistance];
}

/**
 * Combines all agent perception features into a single input vector for the neural network.
 *
 * Input vector layout (Simplified):
 * - Direction to exit: 3
 * - Percent explored: 1
 * - Last reward: 1
 *
 * Total: 3 + 1 + 1 = 5
 */
export function getEnhancedVision(
  encodedMaze: number[][],
  [agentX, agentY]: [number, number],
  visitedPositions: Set<string>,
  exitPos?: [number, number],
  lastReward?: number,
  globalProgress?: number
): number[] {
  const width = encodedMaze[0].length;
  const height = encodedMaze.length;
  // Calculate percent explored
  const percentExplored = globalProgress !== undefined ? globalProgress : visitedPositions.size / (width * height);
  return [
    ...encodeDirectionToExit(agentX, agentY, exitPos, width, height),
    percentExplored,
    lastReward || 0
  ];
}

// --- Movement & Simulation ---

/** Checks if a move is valid (within bounds and not a wall). */
export function isValidMove(encodedMaze: number[][], [x, y]: [number, number]): boolean {
  const height = encodedMaze.length;
  const width = encodedMaze[0].length;
  return x >= 0 && x < width && y >= 0 && y < height && encodedMaze[y][x] !== -1;
}

/** Moves the agent in the given direction if possible, otherwise stays in place. */
export function moveAgent(encodedMaze: number[][], position: [number, number], direction: number): [number, number] {
  const [x, y] = position;
  const moves = [ [0, -1], [1, 0], [0, 1], [-1, 0] ];
  const validDirection = Number.isInteger(direction) && direction >= 0 && direction < moves.length 
    ? direction 
    : Math.floor(Math.random() * moves.length);
  const [dx, dy] = moves[validDirection];
  const newX = x + dx;
  const newY = y + dy;
  return isValidMove(encodedMaze, [newX, newY]) ? [newX, newY] : [x, y];
}

/** Selects the direction with the highest output value from the neural network. */
export function selectDirection(outputs: number[]): number {
  let maxVal = -Infinity, maxIdx = 0;
  for (let i = 0; i < outputs.length; i++) {
    if (!isNaN(outputs[i]) && outputs[i] > maxVal) {
      maxVal = outputs[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/** Detects which side of the agent is next to a wall (if any). */
function detectWallSide(encodedMaze: number[][], position: [number, number]): string | null {
  const [x, y] = position;
  if (!isValidMove(encodedMaze, [x - 1, y])) return 'L';
  if (!isValidMove(encodedMaze, [x + 1, y])) return 'R';
  if (!isValidMove(encodedMaze, [x, y - 1])) return 'U';
  if (!isValidMove(encodedMaze, [x, y + 1])) return 'D';
  return null;
}

/**
 * Simulates the agent navigating the maze using its neural network.
 */
export function simulateAgent(
  network: Network,
  encodedMaze: number[][],
  startPos: [number, number],
  exitPos: [number, number],
  maxSteps = 3000
): {
  success: boolean;
  steps: number;
  path: [number, number][];
  fitness: number;
  progress: number;
} {
  let position = [...startPos] as [number, number];
  let steps = 0;
  let path = [position.slice() as [number, number]];
  let visitedPositions = new Set<string>();
  let revisitCount = 0;
  let minDistanceToExit = manhattanDistance(position, exitPos);
  let invalidMoves = 0;
  let progressReward = 0;
  let consecutiveStays = 0;
  let wallHugStreak = 0;
  let lastWallSide: string | null = null;
  let noProgressSteps = 0;
  let bestDistance = manhattanDistance(position, exitPos);
  let revisitPenalty = 0;
  let ditheringPenalty = 0;
  let noProgressPenalty = 0;
  let revisitMap = new Map<string, number>();
  let oscillationPenalty = 0;
  let lastReward = 0;
  let newCellExplorationBonus = 0; // Added for explicit exploration reward

  while (steps < maxSteps) {
    steps++;

    // Vision input calculation
    const percentExplored = visitedPositions.size / (encodedMaze.length * encodedMaze[0].length);
    const vision = getEnhancedVision(
      encodedMaze,
      position,
      visitedPositions,
      exitPos,
      lastReward, // Use reward from *previous* step's outcome
      percentExplored
    );
    const expectedInputSize = 3 + 1 + 1;
    if (vision.length !== expectedInputSize) {
      throw new Error(`simulateAgent: Input vector length is ${vision.length}, expected ${expectedInputSize}`);
    }
    const outputs = network.activate(vision);
    let direction = (outputs && outputs.length > 0) ? selectDirection(outputs) : Math.floor(Math.random() * 4);

    const prevPosition = [...position] as [number, number];
    const prevDistance = manhattanDistance(position, exitPos);
    lastReward = 0; // Reset for current step's outcome evaluation

    // Agent moves
    position = moveAgent(encodedMaze, position, direction);
    const moved = prevPosition[0] !== position[0] || prevPosition[1] !== position[1];
    const currentPosKey = `${position[0]},${position[1]}`;

    // --- Update state and calculate penalties/rewards for the current step ---

    if (moved) {
      path.push(position.slice() as [number, number]);
      consecutiveStays = 0;
      const currentDistance = manhattanDistance(position, exitPos);

      // Reward for moving to a new cell
      if (!visitedPositions.has(currentPosKey)) {
        newCellExplorationBonus += 0.3; // Small direct bonus to fitness for new cells
        lastReward = Math.max(lastReward, 0.15); // Positive signal for next input
      }

      if (currentDistance < prevDistance) {
        progressReward += 0.2; // Reduced from 0.5
        lastReward = Math.max(lastReward, 0.1); // Reduced from 0.5
        noProgressSteps = 0;
      } else {
        // Small penalty for not making progress if moved
        progressReward -= 0.05; // Reduced from 0.1
        lastReward = Math.min(lastReward, -0.05); // Reduced from -0.1
        noProgressSteps++;
      }

      if (currentDistance < bestDistance) {
        bestDistance = currentDistance;
        noProgressSteps = 0; // Reset noProgressSteps if new best distance is found
        lastReward = Math.max(lastReward, 0.05); // Reduced from 0.2
      }

      // Oscillation Penalty (back and forth between two cells)
      if (path.length >= 3) {
        const prev2PathIndex = path.length - 3;
        const prev2Pos = path[prev2PathIndex];
        if (prev2Pos[0] === position[0] && prev2Pos[1] === position[1]) {
          oscillationPenalty -= 2.5; // Increased from 2
          lastReward = Math.min(lastReward, -1.5); // Increased from -1
        }
      }

      // Wall Hugging Penalty
      const currentWallSide = detectWallSide(encodedMaze, position);
      if (currentWallSide && currentWallSide === lastWallSide) {
        wallHugStreak++;
      } else {
        wallHugStreak = 0;
      }
      lastWallSide = currentWallSide;
      if (wallHugStreak >= 20) { // Increased threshold from 15
        progressReward -= 2.5; // Increased from 2
        lastReward = Math.min(lastReward, -0.75); // Increased from -0.5
      }

    } else { // Agent did not move (dithering)
      invalidMoves++;
      consecutiveStays++;
      if (consecutiveStays >= 5) {
        ditheringPenalty -= 1.5 * consecutiveStays; // Reduced from 2 * consecutiveStays
        lastReward = Math.min(lastReward, -0.4); // Reduced from -0.5
      } else {
        ditheringPenalty -= 0.5; // Reduced from 1
        lastReward = Math.min(lastReward, -0.1); // Reduced from -0.2
      }
      noProgressSteps++; // Count dithering as no progress
    }

    // No Progress Penalty (applied regardless of whether agent moved or not, if no progress to exit)
    if (noProgressSteps >= 8) { // Increased threshold from 5
      noProgressPenalty -= noProgressSteps * 0.1; // Reduced from 0.2
      lastReward = Math.min(lastReward, -0.15 * noProgressSteps); // Reduced from -0.2
    }

    // Revisit Penalty (only if moved)
    if (moved) {
      if (visitedPositions.has(currentPosKey)) {
        revisitCount++;
        const cellRevisits = revisitMap.get(currentPosKey) || 0;
        // Scaled penalty, but cap it to avoid extreme punishment for revisiting useful spots
        revisitPenalty -= 0.1 * Math.min(3, cellRevisits); // Reduced from 0.5 * Math.min(5, cellRevisits)
        lastReward = Math.min(lastReward, -0.05 * Math.min(3, cellRevisits)); // Reduced
      }
      visitedPositions.add(currentPosKey);
      revisitMap.set(currentPosKey, (revisitMap.get(currentPosKey) || 0) + 1);
    }

    // Update minDistanceToExit
    const currentDistanceToExit = manhattanDistance(position, exitPos);
    minDistanceToExit = Math.min(minDistanceToExit, currentDistanceToExit);

    // Add incentive: negative reward proportional to distance to exit
    progressReward -= 0.02 * currentDistanceToExit;

    // Early termination if stuck
    if (revisitCount > 150 || noProgressSteps > 150) { // Increased thresholds
      break;
    }

    if (position[0] === exitPos[0] && position[1] === exitPos[1]) {
      const efficiencyBonus = Math.max(0, 50 - (steps * 0.05)); // Reduced from 100 - (steps * 0.1)
      const stepPenalty = steps > (minDistanceToExit * 1.5) ? (steps - minDistanceToExit * 1.5) * 0.1 : 0; // Reduced from 0.2
      // Fitness emphasizes success, then exploration (visited unique cells), then efficiency
      const fitness = 500 + efficiencyBonus - stepPenalty + (maxSteps - steps) * 0.2 + progressReward + revisitPenalty + ditheringPenalty + noProgressPenalty + oscillationPenalty + (visitedPositions.size * 0.5); // Added stronger exploration bonus
      const progress = 100;
      return { success: true, steps, path, fitness, progress };
    }
  }
  const progress = calculateProgress(path[path.length - 1], startPos, exitPos);
  // Base score primarily on progress and exploration
  const baseScore = progress * 1.0 + (visitedPositions.size * 0.75); // Was progress * 2. Exploration more heavily weighted.
  const fitness = baseScore + progressReward + revisitPenalty + ditheringPenalty + noProgressPenalty + oscillationPenalty + newCellExplorationBonus;
  return { success: false, steps, path, fitness, progress };
}

// --- Visualization ---

function renderCell(cell: string, x: number, y: number, agentX: number, agentY: number, path: Set<string> | undefined): string {
  if (x === agentX && y === agentY) {
    if (cell === 'S') return `${colors.bgWhite}${colors.pureGreen}S${colors.reset}`;
    if (cell === 'E') return `${colors.bgWhite}${colors.pureRed}E${colors.reset}`;
    return `${colors.bgGreen}${colors.bright}A${colors.reset}`;
  }
  switch (cell) {
    case 'S': return `${colors.bgWhite}${colors.pureGreen}S${colors.reset}`;
    case 'E': return `${colors.bgWhite}${colors.pureRed}E${colors.reset}`;
    case '#': return `${colors.darkWallBg}${colors.darkWallText}#${colors.reset}`;
    case '.':
      if (path && path.has(`${x},${y}`)) return `${colors.lightBrownBg}${colors.pureGreen}•${colors.reset}`;
      return `${colors.lightBrownBg}${colors.lightBrownText}.${colors.reset}`;
    default: return cell;
  }
}

/** Renders the entire maze as a colored ASCII string, showing the agent and its path. */
export function visualizeMaze(asciiMaze: string[], [agentX, agentY]: [number, number], path?: [number, number][]): string {
  const visitedPositions = path ? new Set(path.map(pos => `${pos[0]},${pos[1]}`)) : undefined;
  return asciiMaze
    .map((row, y) =>
      [...row].map((cell, x) => renderCell(cell, x, y, agentX, agentY, visitedPositions)).join('')
    )
    .join('\n');
}

/** Prints a legend explaining the maze symbols and colors. */
export function displayMazeLegend(forceLog: (...args: any[]) => void): void {
  forceLog(`\n${centerLine('MAZE LEGEND')}`);
  forceLog(`${colors.darkWallBg}${colors.darkWallText}#${colors.reset} - Wall (obstacle the agent cannot pass through)`);
  forceLog(`${colors.lightBrownBg}${colors.lightBrownText}.${colors.reset} - Open path`);
  forceLog(`${colors.bgWhite}${colors.pureGreen}S${colors.reset} - Start position`);
  forceLog(`${colors.bgWhite}${colors.pureRed}E${colors.reset} - Exit/goal position`);
  forceLog(`${colors.bgGreen}${colors.bright}A${colors.reset} - Current agent position`);
  forceLog(`${colors.lightBrownBg}${colors.pureGreen}•${colors.reset} - Path taken by the agent`);
  forceLog(`\nThe agent must find a path from S to E while avoiding walls.\n`);
}

/** Prints a summary of the agent's attempt, including success, steps, and efficiency. */
export function printMazeStats(result: any, maze: string[], forceLog: (...args: any[]) => void): void {
  const successColor = result.success ? colors.pureGreen : colors.red;
  forceLog(`\n${colors.bright}${colors.cyan}===== MAZE SOLUTION SUMMARY =====${colors.reset}`);
  forceLog(`${colors.bright}Success:${colors.reset} ${successColor}${result.success ? 'YES' : 'NO'}${colors.reset}`);
  forceLog(`${colors.bright}Steps taken:${colors.reset} ${result.steps}`);
  forceLog(`${colors.bright}Path length:${colors.reset} ${result.path.length}`);
  const startPos = findPosition(maze, 'S');
  const exitPos = findPosition(maze, 'E');
  if (result.success) {
    const optimalLength = manhattanDistance(startPos, exitPos);
    const efficiency = ((optimalLength / (result.path.length - 1)) * 100).toFixed(1);
    forceLog(`${colors.bright}Path efficiency:${colors.reset} ${optimalLength}/${result.path.length - 1} (${efficiency}%)`);
    forceLog(`${colors.bright}${colors.pureGreen}Agent successfully navigated the maze!${colors.reset}`);
  } else {
    const bestProgress = calculateProgress(result.path[result.path.length - 1], startPos, exitPos);
    forceLog(`${colors.bright}Best progress toward exit:${colors.reset} ${bestProgress}%`);
    forceLog(`${colors.bright}${colors.red}Agent failed to reach the exit.${colors.reset}`);
  }
}

/** Prints a summary of the evolution process (generations, time, best fitness). */
export function printEvolutionSummary(generations: number, timeMs: number, bestFitness: number, forceLog: (...args: any[]) => void): void {
  forceLog(`\n${colors.bright}${colors.cyan}===== EVOLUTION SUMMARY =====${colors.reset}`);
  forceLog(`${colors.bright}Total generations:${colors.reset} ${generations}`);
  forceLog(`${colors.bright}Training time:${colors.reset} ${(timeMs/1000).toFixed(1)} seconds (${(timeMs/60000).toFixed(2)} minutes)`);
  forceLog(`${colors.bright}Best fitness:${colors.reset} ${bestFitness.toFixed(2)}`);
}

/** Displays a colored progress bar for agent progress. */
export function displayProgressBar(progress: number, length: number = 20): string {
  const filledLength = Math.max(0, Math.min(length, Math.floor(length * progress / 100)));
  const startChar = '|';
  const endChar = '|';
  const fillChar = '=';
  const emptyChar = '-';
  const pointerChar = '>';
  let bar = '';
  bar += startChar;
  if (filledLength > 0) {
    bar += fillChar.repeat(filledLength - 1);
    bar += pointerChar;
  }
  const emptyLength = length - filledLength;
  if (emptyLength > 0) {
    bar += emptyChar.repeat(emptyLength);
  }
  bar += endChar;
  const color = progress < 30 ? colors.red : progress < 70 ? colors.yellow : colors.green;
  return `${color}${bar}${colors.reset} ${progress}%`;
}

/** Formats elapsed time in a human-readable way. */
export function formatElapsedTime(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

/** Evolves the agent until it solves the maze or reaches a progress threshold. */
export async function evolveUntilSolved(
  evolveFn: () => Promise<{ finalResult: { success: boolean; progress: number } }>,
  minProgressToPass: number = 60,
  maxTries: number = 10
): Promise<{ finalResult: { success: boolean; progress: number }, tries: number }> {
  let tries = 0;
  let lastResult: { success: boolean; progress: number } = { success: false, progress: 0 };
  while (tries < maxTries) {
    tries++;
    const { finalResult } = await evolveFn();
    lastResult = finalResult;
    if (finalResult.success || finalResult.progress >= minProgressToPass) {
      return { finalResult, tries };
    }
  }
  return { finalResult: lastResult, tries };
}

/** Helper to center a message with padding (for pretty console output). */
export function centerLine(text: string, width = 100, padChar = '=') {
  const pad = Math.max(0, width - text.length);
  const left = Math.floor(pad / 2);
  const right = pad - left;
  return padChar.repeat(left) + text + padChar.repeat(right);
}

// --- Network Visualization ---

function getNodeValue(node: any): number {
  if (typeof node.activation === 'number' && isFinite(node.activation) && !isNaN(node.activation)) {
    return Math.max(-99, Math.min(99, node.activation));
  }
  return 0;
}

function fmtValue(v: number): string {
  if (typeof v !== 'number' || isNaN(v) || !isFinite(v)) return ' --';
  return (v >= 0 ? ' ' : '') + v.toFixed(2);
}

function groupedInputLayer(nodes: any[], maxOtherRows: number): { str: string, count: number }[] {
  const maxRows = Math.max(1, maxOtherRows + 2);
  const n = nodes.length;
  if (n <= maxRows) {
    return nodes.map(n => ({
      str: `${colors.green}●${colors.reset}${fmtValue(getNodeValue(n))}`,
      count: 1
    }));
  }
  const groupSize = Math.ceil(n / maxRows);
  const groups: { str: string, count: number }[] = [];
  for (let i = 0; n > i; i += groupSize) {
    const group = nodes.slice(i, i + groupSize);
    const avg = group.reduce((sum, node) => sum + getNodeValue(node), 0) / group.length;
    groups.push({
      str: `${colors.green}●${colors.reset}${fmtValue(avg)} (avg of ${group.length})`,
      count: group.length
    });
  }
  return groups;
}

function verticalLayer(symbol: string, color: string, nodes: any[], targetRows: number): string[] {
  const n = nodes.length;
  if (n === 0) return Array(targetRows).fill('');
  const maxShow = Math.max(8, targetRows);
  const shown = nodes.slice(0, maxShow);
  const lines = shown.map(n => `${color}${symbol}${colors.reset}${fmtValue(getNodeValue(n))}`);
  if (nodes.length > maxShow) lines.push(`${color}...${colors.reset} (${nodes.length} total)`);
  return lines;
}

function pad(str: string, width: number, align: 'left'|'center'|'right' = 'left'): string {
  const len = str.replace(/\x1b\[[0-9;]*m/g, '').length;
  if (len >= width) return str;
  const padLen = width - len;
  if (align === 'left') return str + ' '.repeat(padLen);
  if (align === 'right') return ' '.repeat(padLen) + str;
  const left = Math.floor(padLen / 2);
  const right = padLen - left;
  return ' '.repeat(left) + str + ' '.repeat(right);
}

/**
 * Visualizes the winner network in a compact, horizontal ASCII way.
 */
export function visualizeNetworkSummary(network: Network): string {
  const inputCount = network.input;
  const outputCount = network.output;
  const nodes = network.nodes || [];
  const inputNodes = nodes.filter(n => n.type === 'input');
  const outputNodes = nodes.filter(n => n.type === 'output');
  const hiddenNodes = nodes.filter(n => n.type !== 'input' && n.type !== 'output');

  // --- Layer height logic ---
  const minInputRows = 1;
  const maxOtherRows = Math.max(hiddenNodes.length, outputNodes.length, 1);
  const inputRows = Math.max(minInputRows, maxOtherRows + 2);
  const maxRows = Math.max(inputRows, hiddenNodes.length, outputNodes.length, 1);

  const inputCol = groupedInputLayer(inputNodes.length ? inputNodes : Array(inputCount).fill({activation: 0}), maxOtherRows);
  const hiddenCol = verticalLayer('■', colors.yellow, hiddenNodes, maxRows);

  // Output column: always top-aligned and always 4 outputs (hard coded)
  const outputColRaw: string[] = [];
  for (let i = 0; i < 4; ++i) {
    const n = outputNodes[i];
    let displayVal: string;
    if (n && typeof n.activation === 'number' && isFinite(n.activation) && !isNaN(n.activation)) {
      displayVal = fmtValue(getNodeValue(n));
    } else {
      displayVal = ' --';
    }
    outputColRaw.push(`${colors.red}▲${colors.reset}${displayVal}`);
  }
  // Top-align output nodes in output column, pad to maxRows
  const outputColStrs: string[] = [];
  outputColStrs.push(...outputColRaw);
  while (outputColStrs.length < maxRows) outputColStrs.push('');

  // Pad input/hidden columns to maxRows (top-aligned)
  const inputColStrs = inputCol.map(g => g.str);
  while (inputColStrs.length < maxRows) inputColStrs.push('');
  while (hiddenCol.length < maxRows) hiddenCol.push('');

  // Even column width calculation for 100-char output
  const docWidth = 100;
  const arrow1Width = 5; // " ──▶ "
  const arrow2Width = 5;
  const colSpace = docWidth - (arrow1Width + arrow2Width);
  const inputSegWidth = Math.floor(colSpace / 3);
  const hiddenSegWidth = Math.floor(colSpace / 3);
  const outputSegWidth = colSpace - inputSegWidth - hiddenSegWidth;

  const arrow1 = colors.cyan + ' ──▶ ' + colors.reset;
  const arrow2 = colors.cyan + ' ──▶ ' + colors.reset;

  // Header: align with columns, always show all layer counts
  let header = '';
  header += pad(`${colors.green}Input Layer [${inputCount}]${colors.reset}`, inputSegWidth, 'center');
  header += pad(arrow1, arrow1Width, 'center');
  header += pad(`${colors.yellow}Hidden Layer [${hiddenNodes.length}]${colors.reset}`, hiddenSegWidth, 'center');
  header += pad(arrow2, arrow2Width, 'center');
  header += pad(`${colors.red}Output Layer [${outputCount}]${colors.reset}`, outputSegWidth, 'center');

  const lines: string[] = [];
  lines.push(header);

  for (let i = 0; i < maxRows; ++i) {
    const inputStr = inputColStrs[i] || '';
    const hiddenStr = hiddenCol[i] || '';
    let outputStr = '';
    let arrow2Seg = ' '.repeat(arrow2Width);
    if (i < 4) {
      outputStr = outputColStrs[i];
      arrow2Seg = pad(arrow2, arrow2Width, 'center');
    }
    if (!outputStr) outputStr = '';
    const inputSeg  = pad(inputStr, inputSegWidth, 'left');
    const arrow1Seg = (inputStr.trim().length > 0 && hiddenStr.trim().length > 0)
      ? pad(arrow1, arrow1Width, 'center') : ' '.repeat(arrow1Width);
    const hiddenSeg = pad(hiddenStr, hiddenSegWidth, 'left');
    const outputSeg = pad(outputStr, outputSegWidth, 'left');
    const row = inputSeg + arrow1Seg + hiddenSeg + arrow2Seg + outputSeg;
    lines.push(row.length > docWidth ? row.slice(0, docWidth) : pad(row, docWidth));
  }

  lines.push('');
  lines.push(
    `${colors.green}Inputs:${inputCount}${colors.reset}   ` +
    `${colors.yellow}Hidden:${hiddenNodes.length}${colors.reset}   ` +
    `${colors.red}Outputs:${outputCount}${colors.reset}`
  );
  lines.push('');
  lines.push(`${colors.cyan}Arrows indicate feed-forward flow.${colors.reset}`);
  lines.push('');
  lines.push(
    `Legend:  ${colors.green}●${colors.reset}=Input  ` +
    `${colors.yellow}■${colors.reset}=Hidden  ` +
    `${colors.red}▲${colors.reset}=Output  value: last activation`
  );
  return lines.join('\n');
}
