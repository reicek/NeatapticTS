/**
 * Maze Visualization - Handles rendering and visualization of mazes
 * 
 * This module contains functions for visualizing mazes in the terminal,
 * including colored cell rendering, path visualization, and progress indicators.
 * It provides an intuitive way to observe the agent's behavior and solution paths.
 * 
 * The visualization uses ANSI color codes to create a rich terminal interface
 * showing different maze elements (walls, paths, start/exit) and the agent's
 * current position and traversal history.
 */

import { findPosition, bfsDistance, calculateProgress, encodeMaze } from './mazeUtils';
import { colors } from './colors';

/**
 * Renders a single maze cell with proper coloring based on its content and agent location.
 * 
 * This function applies appropriate colors and styling to each cell in the maze:
 * - Different colors for walls, open paths, start and exit positions
 * - Highlights the agent's current position
 * - Marks cells that are part of the agent's path
 * - Renders box drawing characters as walls with proper styling
 * 
 * @param cell - The character representing the cell ('S', 'E', '#', '.' etc.)
 * @param x - X-coordinate of the cell
 * @param y - Y-coordinate of the cell
 * @param agentX - X-coordinate of the agent's current position
 * @param agentY - Y-coordinate of the agent's current position
 * @param path - Optional set of visited coordinates in "x,y" format
 * @returns Colorized string representing the cell
 */
function renderCell(cell: string, x: number, y: number, agentX: number, agentY: number, path: Set<string> | undefined): string {
  // Unicode box drawing characters that should be treated as walls
  const wallChars = new Set(['#', '═', '║', '╔', '╗', '╚', '╝', '╠', '╣', '╦', '╩', '╬']);
  
  // Agent's current position takes precedence in visualization
  if (x === agentX && y === agentY) {
    if (cell === 'S') return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;
    if (cell === 'E') return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;
    return `${colors.bgBlack}${colors.orangeNeon}A${colors.reset}`; // 'A' for Agent - TRON cyan
  }
  
  // Render other cell types
  switch (cell) {
    case 'S': return `${colors.bgBlack}${colors.orangeNeon}S${colors.reset}`;    // Start position
    case 'E': return `${colors.bgBlack}${colors.orangeNeon}E${colors.reset}`;    // Exit position - TRON orange
    case '.':
      // Show path breadcrumbs if this cell was visited
      if (path && path.has(`${x},${y}`)) return `${colors.floorBg}${colors.orangeNeon}•${colors.reset}`;
      return `${colors.floorBg}${colors.gridLineText}.${colors.reset}`; // Open path - dark floor with subtle grid
    default:
      // For box drawing characters and # - render as wall
      if (wallChars.has(cell)) {
        return `${colors.bgBlack}${colors.blueNeon}${cell}${colors.reset}`;
      }
      return cell; // Any other character
  }
}

/** 
 * Renders the entire maze as a colored ASCII string, showing the agent and its path.
 * 
 * This is the main visualization function that converts the maze data structure
 * into a human-readable, colorized representation showing:
 * - The maze layout with walls and open paths
 * - The start and exit positions
 * - The agent's current position
 * - The path the agent has taken (if provided)
 *
 * @param asciiMaze - Array of strings representing the maze layout
 * @param [agentX, agentY] - Current position of the agent
 * @param path - Optional array of positions representing the agent's path
 * @returns A multi-line string with the visualized maze
 */
export function visualizeMaze(asciiMaze: string[], [agentX, agentY]: [number, number], path?: [number, number][]): string {
  // Convert path array to a set of "x,y" strings for quick lookup
  const visitedPositions = path ? new Set(path.map(pos => `${pos[0]},${pos[1]}`)) : undefined;
  
  // Process each row and cell
  return asciiMaze
    .map((row, y) =>
      [...row].map((cell, x) => renderCell(cell, x, y, agentX, agentY, visitedPositions)).join('')
    )
    .join('\n');
}

/** 
 * Prints a summary of the agent's attempt, including success, steps, and efficiency.
 * 
 * This function provides performance metrics about the agent's solution attempt:
 * - Whether it successfully reached the exit
 * - How many steps it took
 * - How efficient the path was compared to the optimal BFS distance
 * 
 * @param result - Object containing the simulation results (success, steps, path)
 * @param maze - Array of strings representing the maze layout
 * @param forceLog - Function used for logging output
 */
export function printMazeStats(result: any, maze: string[], forceLog: (...args: any[]) => void): void {
  const successColor = result.success ? colors.cyanNeon : colors.neonRed;

  forceLog(`${colors.blueCore}║${centerLine(' ', 148, ' ')}║${colors.reset}`);
  forceLog(`${colors.blueCore}║${centerLine(' ', 148, ' ')}║${colors.reset}`);

  forceLog(`${colors.blueCore}║ ${colors.neonSilver}Success:${colors.reset} ${successColor}${result.success ? 'YES' : 'NO'}${colors.reset}`);
  forceLog(`${colors.blueCore}║ ${colors.neonSilver}Steps taken:${colors.reset} ${result.steps}`);
  forceLog(`${colors.blueCore}║ ${colors.neonSilver}Path length:${colors.reset} ${result.path.length}`);
  
  // Find maze start and end positions
  const startPos = findPosition(maze, 'S');
  const exitPos = findPosition(maze, 'E');
  const optimalLength = bfsDistance(encodeMaze(maze), startPos, exitPos);
  forceLog(`${colors.blueCore}║ ${colors.bright}Optimal distance to exit:${colors.reset} ${optimalLength}`);
  
  if (result.success) {
    // Calculate path efficiency - optimal vs actual
    const pathLength = result.path.length - 1;
    const efficiency = Math.min(100, Math.round((optimalLength / pathLength) * 100)).toFixed(1);
    const overhead = ((pathLength / optimalLength) * 100 - 100).toFixed(1);
    
    // Calculate unique cells and revisits
    const uniqueCells = new Set<string>();
    let revisitedCells = 0;
    let directionChanges = 0;
    let lastDirection: string | null = null;
    
    // Track path metrics
    for (let i = 0; i < result.path.length; i++) {
      const [x, y] = result.path[i];
      const cellKey = `${x},${y}`;
      
      // Count revisits
      if (uniqueCells.has(cellKey)) {
        revisitedCells++;
      } else {
        uniqueCells.add(cellKey);
      }
      
      // Count direction changes (if not the first step)
      if (i > 0) {
        const [prevX, prevY] = result.path[i-1];
        const dx = x - prevX;
        const dy = y - prevY;
        
        // Get current direction (N, S, E, W)
        let currentDirection = "";
        if (dx > 0) currentDirection = "E";
        else if (dx < 0) currentDirection = "W";
        else if (dy > 0) currentDirection = "S";
        else if (dy < 0) currentDirection = "N";
        
        // Check if direction changed
        if (lastDirection !== null && currentDirection !== lastDirection) {
          directionChanges++;
        }
        
        lastDirection = currentDirection;
      }
    }
    
    // Calculate exploration coverage (unique cells compared to walkable cells)
    const mazeWidth = maze[0].length;
    const mazeHeight = maze.length;
    const encodedMaze = encodeMaze(maze);
    let walkableCells = 0;
    
    for (let y = 0; y < mazeHeight; y++) {
      for (let x = 0; x < mazeWidth; x++) {
        if (encodedMaze[y][x] !== -1) { // Not a wall
          walkableCells++;
        }
      }
    }
    
    const coveragePercent = ((uniqueCells.size / walkableCells) * 100).toFixed(1);
    
    // Display stats
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Path efficiency:${colors.reset} ${optimalLength}/${pathLength} (${efficiency}%)`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Optimal steps:${colors.reset} ${optimalLength} times`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Path overhead:${colors.reset} ${overhead}% longer than optimal`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Direction changes:${colors.reset} ${directionChanges}`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Unique cells visited:${colors.reset} ${uniqueCells.size} (${coveragePercent}% of maze)`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Cells revisited:${colors.reset} ${revisitedCells} times`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Decisions per cell:${colors.reset} ${(directionChanges / uniqueCells.size).toFixed(2)}`);
    forceLog(`${colors.blueCore}║ ${colors.blueCore}Agent successfully navigated the maze!${colors.reset}`);
  } else {
    // Calculate progress made toward the exit
    const bestProgress = calculateProgress(encodeMaze(maze), result.path[result.path.length - 1], startPos, exitPos);
    
    // Calculate unique cells visited
    const uniqueCells = new Set<string>();
    for (const [x, y] of result.path) {
      uniqueCells.add(`${x},${y}`);
    }
    
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Best progress toward exit:${colors.reset} ${bestProgress}%`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Optimal steps:${colors.reset} ${optimalLength} times`);
    forceLog(`${colors.blueCore}║ ${colors.neonSilver}Unique cells visited:${colors.reset} ${uniqueCells.size}`);
    forceLog(`${colors.blueCore}║ ${colors.neonOrange}Agent trying to reach the exit.${colors.reset}`);
  }
}

/** 
 * Displays a colored progress bar for agent progress.
 * 
 * Creates a visual representation of the agent's progress toward the exit
 * as a horizontal bar with appropriate coloring based on percentage.
 * 
 * @param progress - Progress percentage (0-100)
 * @param length - Length of the progress bar in characters (default: 60)
 * @returns A string containing the formatted progress bar
 */
export function displayProgressBar(progress: number, length: number = 60): string {
  // Calculate the number of filled positions
  const filledLength = Math.max(0, Math.min(length, Math.floor(length * progress / 100)));
  
  // Define the characters for the progress bar
  const startChar = '|';
  const endChar = '|';
  const fillChar = '═';
  const emptyChar = '-';
  const pointerChar = '>'; // Indicates the current progress point
  
  // Construct the progress bar
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
  
  // Add color based on progress level using our TRON palette
  const color = progress < 30 ? colors.red : progress < 70 ? colors.blueNeon : colors.cyanNeon;
  return `${color}${bar}${colors.reset} ${progress}%`;
}

/** 
 * Formats elapsed time in a human-readable way.
 * 
 * Converts seconds into appropriate units (seconds, minutes, hours)
 * for more intuitive display of time durations.
 * 
 * @param seconds - Time in seconds
 * @returns Formatted string (e.g., "5.3s", "2m 30s", "1h 15m")
 */
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

/** 
 * Helper to center a message with padding (for pretty console output).
 * 
 * This utility function creates a centered text line surrounded by
 * padding characters, useful for creating headers and section dividers
 * in the console output.
 * 
 * @param text - The text to center
 * @param width - Total width of the line (default: 150)
 * @param padChar - Character to use for padding (default: '═')
 * @returns The formatted centered line
 */
export function centerLine(text: string, width = 150, padChar: string = '═'): string {
  const pad = Math.max(0, width - text.length);
  const left = Math.floor(pad / 2);
  const right = pad - left;
  return padChar.repeat(left) + text + padChar.repeat(right);
}
