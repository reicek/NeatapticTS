/**
 * Maze Enhancements - Advanced agent perception and navigation capabilities
 * 
 * This module extends the basic maze-solving capabilities with more sophisticated
 * perception, memory, and navigation abilities. These enhancements make the agent
 * more effective at solving complex mazes without changing its I/O structure.
 * 
 * The enhancements include:
 * - Memory-enhanced vision
 * - Dead-end detection
 * - Momentum-based movement
 * - Junction recognition
 * - Pattern recognition
 * - Dynamic reward scaling
 * - Exit scent gradient
 * - Uncertainty-based exploration
 * - Temporal integration
 * - Enhanced oscillation detection
 * - Goal-directed backtracking
 * - Dynamic vision range adjustment
 */

import { manhattanDistance } from './mazeUtils';

/**
 * Tracks visited positions with additional metadata to enhance agent memory
 */
export class EnhancedMemory {
  // Core memory components
  visitedPositions: Set<string> = new Set<string>();
  visitCounts: Map<string, number> = new Map<string, number>();
  
  // Special tracking for important features
  junctions: Map<string, {exits: number, visited: number}> = new Map();
  deadEnds: Set<string> = new Set<string>();
  backtrackPoints: Array<[number, number]> = [];

  // Movement history for pattern detection
  positionHistory: Array<[number, number]> = [];
  directionHistory: number[] = [];
  lastDirection: number = -1;
  
  // Exit position tracking for navigation
  lastKnownExitPos: [number, number] | null = null;
  
  // Maximum history lengths to prevent memory overflow
  readonly MAX_HISTORY = 50;
  readonly MAX_BACKTRACK_POINTS = 10;

  /**
   * Records a new position and updates all memory structures
   */
  recordPosition(position: [number, number], direction: number, encodedMaze: number[][]): void {
    const posKey = `${position[0]},${position[1]}`;
    
    // Basic visit tracking
    this.visitedPositions.add(posKey);
    this.visitCounts.set(posKey, (this.visitCounts.get(posKey) || 0) + 1);
    
    // Record in history
    this.positionHistory.push([...position]);
    this.directionHistory.push(direction);
    this.lastDirection = direction;
    
    // Trim history if needed
    if (this.positionHistory.length > this.MAX_HISTORY) {
      this.positionHistory.shift();
      this.directionHistory.shift();
    }
    
    // Check if this position is a junction or dead end
    this.analyzePosition(position, encodedMaze);
  }
  
  /**
   * Analyzes a position to determine if it's a junction or dead end
   */
  private analyzePosition(position: [number, number], encodedMaze: number[][]): void {
    const [x, y] = position;
    const posKey = `${x},${y}`;
    
    // Count open paths in cardinal directions
    let openPaths = 0;
    const dirs: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    
    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      
      // Check if position is valid (not a wall or out of bounds)
      if (
        nx >= 0 && nx < encodedMaze[0].length &&
        ny >= 0 && ny < encodedMaze.length &&
        encodedMaze[ny][nx] !== -1
      ) {
        openPaths++;
      }
    }
    
    // Record junction if there are 3+ exits
    if (openPaths >= 3) {
      if (!this.junctions.has(posKey)) {
        this.junctions.set(posKey, {exits: openPaths, visited: 1});
      } else {
        const junction = this.junctions.get(posKey)!;
        junction.visited++;
      }
      
      // Add as potential backtrack point if not already in list
      if (!this.backtrackPoints.some(p => p[0] === x && p[1] === y)) {
        this.backtrackPoints.push([x, y]);
        if (this.backtrackPoints.length > this.MAX_BACKTRACK_POINTS) {
          this.backtrackPoints.shift();
        }
      }
    }
    
    // Record dead end if there's only one exit
    if (openPaths === 1) {
      this.deadEnds.add(posKey);
    }
  }
  
  /**
   * Detects if the agent is oscillating between positions
   */
  detectOscillation(): boolean {
    if (this.positionHistory.length < 6) return false;
    
    // Look for A-B-A-B pattern (length 4)
    const end = this.positionHistory.length - 1;
    
    // Check simple 2-position oscillation (ABABA)
    const posA = this.positionHistory[end];
    const posB = this.positionHistory[end-1];
    const posC = this.positionHistory[end-2];
    const posD = this.positionHistory[end-3];
    
    const isABOscillation = 
      posA[0] === posC[0] && posA[1] === posC[1] && 
      posB[0] === posD[0] && posB[1] === posD[1];

    // Check for 3-position oscillation (ABCABC)
    const posE = this.positionHistory[end-4];
    const posF = this.positionHistory[end-5];
    
    const isABCOscillation = 
      posA[0] === posD[0] && posA[1] === posD[1] && 
      posB[0] === posE[0] && posB[1] === posE[1] &&
      posC[0] === posF[0] && posC[1] === posF[1];
      
    return isABOscillation || isABCOscillation;
  }
  
  /**
   * Find the nearest junction for backtracking
   */
  getNearestBacktrackPoint(currentPosition: [number, number]): [number, number] | null {
    if (this.backtrackPoints.length === 0) return null;
    
    // Find the nearest junction that's not the current position
    let nearestPoint: [number, number] | null = null;
    let nearestDistance = Infinity;
    
    for (const point of this.backtrackPoints) {
      // Skip current position
      if (point[0] === currentPosition[0] && point[1] === currentPosition[1]) continue;
      
      const distance = manhattanDistance(currentPosition, point);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestPoint = point;
      }
    }
    
    return nearestPoint;
  }
  
  /**
   * Get a list of the least visited neighboring positions
   */
  getLeastVisitedNeighbors(position: [number, number], encodedMaze: number[][]): [number, number][] {
    const [x, y] = position;
    const neighbors: Array<{pos: [number, number], visits: number}> = [];
    const dirs: [number, number][] = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    
    for (const [dx, dy] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      
      // Check if position is valid (not a wall or out of bounds)
      if (
        nx >= 0 && nx < encodedMaze[0].length &&
        ny >= 0 && ny < encodedMaze.length &&
        encodedMaze[ny][nx] !== -1
      ) {
        const posKey = `${nx},${ny}`;
        const visits = this.visitCounts.get(posKey) || 0;
        neighbors.push({pos: [nx, ny], visits});
      }
    }
    
    // Sort by visit count (ascending)
    neighbors.sort((a, b) => a.visits - b.visits);
    
    return neighbors.map(n => n.pos);
  }
  
  /**
   * Check if a position is a dead end
   */
  isDeadEnd(posKey: string): boolean {
    return this.deadEnds.has(posKey);
  }
}

/**
 * Object to store and calculate the temporal history of vision inputs
 */
export class VisionMemory {
  history: number[][] = [];
  readonly maxHistory: number = 5;
  
  /**
   * Add a new vision input to the history
   */
  addVision(vision: number[]): void {
    this.history.push([...vision]);
    if (this.history.length > this.maxHistory) {
      this.history.shift(); // Remove oldest
    }
  }
  
  /**
   * Calculate the delta between current and previous vision
   * Helps identify changes in the environment
   */
  getVisionDelta(): number[] {
    if (this.history.length < 2) {
      return Array(this.history[0]?.length || 0).fill(0);
    }
    
    const current = this.history[this.history.length - 1];
    const previous = this.history[this.history.length - 2];
    
    return current.map((val, idx) => val - previous[idx]);
  }
  
  /**
   * Calculate moving average of vision to reduce noise
   */
  getSmoothedVision(): number[] {
    if (this.history.length === 0) return [];
    if (this.history.length === 1) return [...this.history[0]];
    
    const latest = this.history[this.history.length - 1];
    const result = Array(latest.length).fill(0);
    
    // Calculate weighted average, with more recent visions weighted higher
    let totalWeight = 0;
    for (let i = 0; i < this.history.length; i++) {
      const weight = i + 1; // Higher weight for newer items
      totalWeight += weight;
      
      for (let j = 0; j < this.history[i].length; j++) {
        result[j] += this.history[i][j] * weight;
      }
    }
    
    // Normalize by total weight
    for (let i = 0; i < result.length; i++) {
      result[i] /= totalWeight;
    }
    
    return result;
  }
}

/**
 * Enhanced direction sensing with adaptive exploration modes
 */
export class EnhancedDirectionSense {
  /**
   * Create an exit scent gradient that gets stronger near the exit
   * This provides a more nuanced alternative to the simple direction vector
   */
  static calculateExitScent(
    position: [number, number], 
    exitPos: [number, number], 
    width: number, 
    height: number
  ): number {
    if (!exitPos) return 0;
    
    const maxDistance = width + height;
    const distance = manhattanDistance(position, exitPos);
    
    // Create a non-linear scent that gets stronger closer to the exit
    // Using a quadratic function to make the gradient more pronounced at closer distances
    return Math.pow((maxDistance - distance) / maxDistance, 2);
  }
  
  /**
   * Calculate adaptive directional influence based on maze complexity
   */
  static getAdaptiveDirection(
    dx: number, 
    dy: number, 
    distance: number,
    complexity: number // 0-1 scale of estimated maze complexity
  ): number[] {
    // Determine how much to rely on directional cues based on maze complexity
    // In complex mazes, directional information is less trustworthy
    const directionWeight = Math.max(0.2, 1 - complexity);
    
    // Scale direction components
    const scaledDx = dx * directionWeight;
    const scaledDy = dy * directionWeight;
    const scaledDist = distance * directionWeight;
    
    return [scaledDx, scaledDy, scaledDist];
  }
  
  /**
   * Calculate maze complexity based on wall density and path complexity
   */
  static calculateMazeComplexity(encodedMaze: number[][]): number {
    const height = encodedMaze.length;
    const width = encodedMaze[0].length;
    
    // Count walls
    let wallCount = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (encodedMaze[y][x] === -1) wallCount++;
      }
    }
    
    // Calculate wall density as a complexity measure (0-1)
    const totalCells = width * height;
    const wallDensity = wallCount / totalCells;
    
    // A more complex maze has a wall density of around 0.3-0.5
    // Too many walls (>0.5) or too few (<0.3) are actually simpler
    const complexityFactor = 1 - Math.abs(wallDensity - 0.4) * 2;
    
    return Math.min(1, Math.max(0, complexityFactor));
  }
}

/**
 * Enhanced vision system with memory and pattern recognition
 */
export class EnhancedVision {
  /**
   * Enhances the scanDirection function with memory of visited positions
   */
  static enhanceScanWithMemory(
    scanValue: number,
    dx: number, 
    dy: number,
    visitedPositions: Set<string>,
    agentX: number,
    agentY: number,
    encodedMaze: number[][],
    memory: EnhancedMemory
  ): number {
    // Base scanValue from the original implementation
    let enhancedValue = scanValue;
    
    // Add subtle random factor to break symmetry and prevent getting stuck
    const randomFactor = Math.random() * 0.001;
    enhancedValue += randomFactor;
    
    // Check if this direction leads to a known dead end
    const nx = agentX + dx;
    const ny = agentY + dy;
    const posKey = `${nx},${ny}`;
    
    if (memory.isDeadEnd(posKey)) {
      // Heavily penalize known dead ends
      enhancedValue *= 0.5;
    }
    
    // Check visit counts - encourage exploring less-visited paths
    const visits = memory.visitCounts.get(posKey) || 0;
    if (visits > 0) {
      // Apply a discount based on visit count
      const visitDiscount = Math.min(0.5, visits * 0.1); // Cap at 50% reduction
      enhancedValue *= (1 - visitDiscount);
    }
    
    // Check if this direction leads to a junction
    if (memory.junctions.has(posKey)) {
      // Junctions are valuable exploration points - boost them slightly
      const junction = memory.junctions.get(posKey)!;
      
      // Junctions with more exits are better
      const exitBonus = 0.02 * junction.exits;
      
      // But we want to explore less visited junctions
      const visitPenalty = 0.01 * Math.min(5, junction.visited);
      
      enhancedValue += exitBonus - visitPenalty;
    }
    
    // Ensure value stays within valid range (0-1)
    return Math.min(0.999, Math.max(0, enhancedValue));
  }
  
  /**
   * Get an adaptive vision range based on local maze complexity
   */
  static getAdaptiveVisionRange(encodedMaze: number[][], position: [number, number]): number {
    // Default range
    let baseRange = 100;
    
    // Calculate local complexity (wall density in nearby area)
    let wallCount = 0;
    const searchRadius = 10;
    
    for (let y = Math.max(0, position[1] - searchRadius); 
         y < Math.min(encodedMaze.length, position[1] + searchRadius); 
         y++) {
      for (let x = Math.max(0, position[0] - searchRadius); 
           x < Math.min(encodedMaze[0].length, position[0] + searchRadius); 
           x++) {
        if (encodedMaze[y][x] === -1) wallCount++;
      }
    }
    
    // Area being examined
    const totalArea = Math.min(searchRadius*2, encodedMaze[0].length) * 
                     Math.min(searchRadius*2, encodedMaze.length);
    const complexity = wallCount / totalArea;
    
    // Adjust vision range based on complexity
    // More complex areas (more walls) get shorter but more detailed vision
    return Math.max(50, Math.min(999, baseRange * (1.5 - complexity)));
  }
  
  /**
   * Detect wall patterns to help with navigation
   */
  static detectWallPattern(encodedMaze: number[][], position: [number, number]): string {
    const [x, y] = position;
    
    // Check in all four directions
    const hasWallNorth = y <= 0 || encodedMaze[y-1][x] === -1;
    const hasWallEast = x >= encodedMaze[0].length-1 || encodedMaze[y][x+1] === -1;
    const hasWallSouth = y >= encodedMaze.length-1 || encodedMaze[y+1][x] === -1;
    const hasWallWest = x <= 0 || encodedMaze[y][x-1] === -1;
    
    // Count walls
    const wallCount = (hasWallNorth ? 1 : 0) + 
                     (hasWallEast ? 1 : 0) + 
                     (hasWallSouth ? 1 : 0) + 
                     (hasWallWest ? 1 : 0);
    
    // Identify pattern
    if (wallCount === 0) return "open";
    if (wallCount === 3) return "dead-end";
    if (wallCount === 1) return "corridor-3way";
    if (wallCount === 2) {
      if ((hasWallNorth && hasWallSouth) || (hasWallEast && hasWallWest)) {
        return "corridor";
      }
      return "corner";
    }
    
    return "unknown";
  }
}

/**
 * Enhanced movement strategies for better navigation
 */
export class EnhancedMovement {
  /**
   * Apply momentum to encourage consistent direction
   */
  static applyMomentum(
    outputs: number[], 
    previousDirection: number,
    momentum: number = 0.05
  ): number[] {
    const momentumOutputs = [...outputs];
    
    // Add a small bonus for continuing in same direction
    if (previousDirection >= 0 && previousDirection < outputs.length) {
      momentumOutputs[previousDirection] *= (1 + momentum);
    }
    
    return momentumOutputs;
  }
  
  /**
   * Apply uncertainty to encourage exploration of unfamiliar areas
   */
  static addExplorationUncertainty(
    directionValues: number[], 
    visitCounts: Map<string, number>,
    directionToKey: (dir: number) => string
  ): number[] {
    return directionValues.map((value, idx) => {
      const dirKey = directionToKey(idx);
      const visits = visitCounts.get(dirKey) || 0;
      
      // Add uncertainty bonus inversely proportional to visits
      const uncertaintyBonus = Math.max(0, 0.1 - (visits * 0.01));
      return Math.min(0.999, value + uncertaintyBonus);
    });
  }
  
  /**
   * Find a backtracking route when stuck or oscillating
   */
  static findBacktrackRoute(
    position: [number, number],
    path: [number, number][],
    encodedMaze: number[][],
    memory: EnhancedMemory
  ): [number, number] | null {
    // First try to find nearest junction
    const nearestJunction = memory.getNearestBacktrackPoint(position);
    if (nearestJunction) return nearestJunction;
    
    // If no junction found, try to find a position on the path that has unexplored neighbors
    for (let i = path.length - 1; i >= 0; i--) {
      const pos = path[i];
      const neighbors = memory.getLeastVisitedNeighbors(pos, encodedMaze);
      
      // Look for a position with unvisited neighbors
      for (const neighbor of neighbors) {
        const neighborKey = `${neighbor[0]},${neighbor[1]}`;
        if (!memory.visitedPositions.has(neighborKey)) {
          return pos;
        }
      }
    }
    
    return null;
  }
}

/**
 * Enhanced reward system with more sophisticated behaviors
 */
export class EnhancedReward {
  /**
   * Calculate dynamic reward scale based on maze complexity
   */
  static calculateDynamicRewardScale(encodedMaze: number[][]): number {
    const complexity = EnhancedDirectionSense.calculateMazeComplexity(encodedMaze);
    
    // More complex mazes get higher exploration rewards
    return 0.3 + (complexity * 0.7); // Scale between 0.3 and 1.0
  }
  
  /**
   * Calculate exploration bonus based on global exploration rate
   */
  static calculateExplorationBonus(
    visitedPositions: Set<string>,
    encodedMaze: number[][],
    currentPos: [number, number]
  ): number {
    const totalCells = encodedMaze.length * encodedMaze[0].length;
    const exploredCells = visitedPositions.size;
    const explorationRate = exploredCells / totalCells;
    
    // In early exploration (<30%), focus on covering ground
    // In mid exploration (30-70%), balance between depth and breadth
    // In late exploration (>70%), focus on filling gaps
    
    if (explorationRate < 0.3) {
      return 0.5; // High bonus for any new cell
    } else if (explorationRate < 0.7) {
      return 0.3; // Moderate bonus
    } else {
      return 0.2; // Lower bonus for reaching final cells
    }
  }
  
  /**
   * Enhanced oscillation penalty that can detect multi-position patterns
   */
  static calculateOscillationPenalty(memory: EnhancedMemory): number {
    if (memory.detectOscillation()) {
      // Apply stronger penalty for detected oscillation patterns
      return -2.0;
    }
    return 0;
  }
}

/**
 * Main interface for all enhanced capabilities
 */
export class MazeAgentEnhancements {
  memory: EnhancedMemory;
  visionMemory: VisionMemory;
  
  constructor() {
    this.memory = new EnhancedMemory();
    this.visionMemory = new VisionMemory();
  }
  
  /**
   * Enhanced scan function with memory integration
   */
  enhanceScan(
    scanValue: number, 
    dx: number, 
    dy: number,
    agentX: number,
    agentY: number,
    encodedMaze: number[][]
  ): number {
    return EnhancedVision.enhanceScanWithMemory(
      scanValue,
      dx, 
      dy, 
      this.memory.visitedPositions,
      agentX,
      agentY,
      encodedMaze,
      this.memory
    );
  }
  
  /**
   * Enhanced direction sense with exit scent gradient
   */
  enhanceDirectionSense(
    dx: number, 
    dy: number, 
    distance: number,
    encodedMaze: number[][]
  ): number[] {
    const complexity = EnhancedDirectionSense.calculateMazeComplexity(encodedMaze);
    return EnhancedDirectionSense.getAdaptiveDirection(dx, dy, distance, complexity);
  }
  
  /**
   * Process a new position and update memory
   */
  recordPosition(
    position: [number, number], 
    direction: number,
    encodedMaze: number[][]
  ): void {
    this.memory.recordPosition(position, direction, encodedMaze);
  }
  
  /**
   * Process new vision input and update vision memory
   */
  recordVision(vision: number[]): void {
    this.visionMemory.addVision(vision);
  }
  
  /**
   * Apply momentum to movement decisions
   */
  applyMomentum(outputs: number[]): number[] {
    return EnhancedMovement.applyMomentum(
      outputs, 
      this.memory.lastDirection
    );
  }
  
  /**
   * Calculate penalties for oscillation detection
   */
  getOscillationPenalty(): number {
    return EnhancedReward.calculateOscillationPenalty(this.memory);
  }
  
  /**
   * Get adaptive reward scaling factor based on maze complexity
   */
  getRewardScale(encodedMaze: number[][]): number {
    return EnhancedReward.calculateDynamicRewardScale(encodedMaze);
  }
  
  /**
   * Find a backtrack target when agent is stuck
   */
  findBacktrackTarget(
    position: [number, number],
    path: [number, number][],
    encodedMaze: number[][]
  ): [number, number] | null {
    return EnhancedMovement.findBacktrackRoute(position, path, encodedMaze, this.memory);
  }
  
  /**
   * Get vision range adaptive to local complexity
   */
  getVisionRange(encodedMaze: number[][], position: [number, number]): number {
    return EnhancedVision.getAdaptiveVisionRange(encodedMaze, position);
  }
  
  /**
   * Detect wall pattern at current position
   */
  detectWallPattern(encodedMaze: number[][], position: [number, number]): string {
    return EnhancedVision.detectWallPattern(encodedMaze, position);
  }
  
  /**
   * Generate an anti-revisit signal to help agent avoid previously visited areas
   * The signal becomes stronger as cells are revisited more times
   * 
   * @param position - Current position of the agent
   * @returns A value between 0-1 indicating revisit pressure (higher = more revisits nearby)
   */
  getAntiRevisitSignal(position: [number, number]): number {
    const [x, y] = position;
    const currentKey = `${x},${y}`;
    const currentVisits = this.memory.visitCounts.get(currentKey) || 0;
    
    // Calculate average visit count of adjacent cells (stronger avoidance of revisits)
    const adjacentPositions = [
      [x+1, y], [x-1, y], [x, y+1], [x, y-1], // Direct neighbors
      [x+1, y+1], [x-1, y-1], [x+1, y-1], [x-1, y+1] // Diagonal neighbors 
    ];
    
    let totalVisits = currentVisits;
    let cellCount = 1; // Count current position
    
    for (const [adjX, adjY] of adjacentPositions) {
      const adjKey = `${adjX},${adjY}`;
      const visits = this.memory.visitCounts.get(adjKey) || 0;
      totalVisits += visits;
      if (visits > 0) cellCount++;
    }
    
    // Calculate normalized pressure (0-1)
    // Stronger pressure from repeated visits
    const avgVisits = cellCount > 0 ? totalVisits / cellCount : 0;
    return Math.min(1, avgVisits * 0.2); // Scale factor to normalize (0.2 = 5 visits to reach max)
  }
  
  /**
   * Apply anti-revisit bias to network outputs to strongly discourage revisiting areas
   * 
   * @param outputs - Network output values
   * @param position - Current agent position
   * @param encodedMaze - Maze representation
   * @returns Modified output values with anti-revisit bias applied
   */
  applyAntiRevisitBias(outputs: number[], position: [number, number], encodedMaze: number[][]): number[] {
    if (!outputs || outputs.length !== 4) return outputs; // Safety check
    
    const [x, y] = position;
    const biasedOutputs = [...outputs];
    
    // Direction vectors: North, East, South, West (must match the expected network outputs)
    const directions = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    
    // Calculate the highest output value for normalization reference
    const maxOutput = Math.max(...outputs.filter(v => !isNaN(v) && isFinite(v)));
    
    // Apply bias to each possible movement direction
    for (let i = 0; i < 4; i++) {
      const [dx, dy] = directions[i];
      const nx = x + dx;
      const ny = y + dy;
      
      // Skip if position is out of bounds or a wall
      if (nx < 0 || ny < 0 || 
          nx >= encodedMaze[0].length || 
          ny >= encodedMaze.length || 
          encodedMaze[ny][nx] === -1) {
        continue;
      }
      
      const posKey = `${nx},${ny}`;
      const visits = this.memory.visitCounts.get(posKey) || 0;
      
      // PRIORITY #1: Favor VISION - Vision is already handled in the mazeVision module
      // No adjustments needed here as we prioritize the vision system's outputs
      
      // PRIORITY #2: Already visited locations - Stronger penalty for revisits
      if (visits > 0) {
        // Exponential penalty based on visit count with strategic scaling
        // More revisits = much stronger penalty
        const revisitPenalty = Math.min(0.95, 0.25 * Math.pow(1.4, Math.min(6, visits)));
        biasedOutputs[i] *= (1 - revisitPenalty);
      } else {
        // Significant bonus for unexplored cells to emphasize exploration
        biasedOutputs[i] *= 1.25;
      }
      
      // Additional bias against dead ends
      if (this.memory.isDeadEnd(posKey)) {
        // Stronger penalty for dead ends to avoid wasting time
        biasedOutputs[i] *= 0.4;
      }
      
      // PRIORITY #3: minDistanceToExit - Only apply if exit location is known
      // Check if we have exit position information stored from previous visions
      if (this.memory.lastKnownExitPos) {
        const exitPos = this.memory.lastKnownExitPos;
        const currentDistToExit = manhattanDistance(position, exitPos);
        const nextDistToExit = manhattanDistance([nx, ny], exitPos);
        
        // Bonus for moving closer to exit, small penalty for moving away
        if (nextDistToExit < currentDistToExit) {
          biasedOutputs[i] *= 1.1; // Modest bonus for moving toward exit
        } else if (nextDistToExit > currentDistToExit) {
          biasedOutputs[i] *= 0.95; // Small penalty for moving away from exit
        }
      }
      
      // Bonus for moving toward junctions (potential exploration points)
      if (this.memory.junctions.has(posKey)) {
        const junction = this.memory.junctions.get(posKey)!;
        
        // Higher bonus for less-visited junctions with more exits 
        // This encourages exploration of promising areas
        const junctionBonus = 0.15 * junction.exits / Math.max(1, junction.visited);
        biasedOutputs[i] *= (1 + junctionBonus);
      }
      
      // PRIORITY #4: Other factors - Consider special movement patterns
      
      // Check for oscillation-breaking using the direction history
      const directionOpposite = (i + 2) % 4; // Opposite direction
      if (this.memory.lastDirection === directionOpposite) {
        // Small penalty for reversing direction (reduces oscillation)
        biasedOutputs[i] *= 0.9;
      }
      
      // Check for wall following pattern and discourage excessive wall hugging
      const wallPattern = this.detectWallPattern(encodedMaze, position);
      if (wallPattern === "corridor" && visits > 2) {
        // Gradually reduce value of revisiting corridors multiple times
        biasedOutputs[i] *= Math.pow(0.97, Math.min(4, visits - 2));
      }
    }
    
    return biasedOutputs;
  }
}