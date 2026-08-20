/**
 * MAP-Elites Quality-Diversity archive for the Neatenstein enemy population.
 *
 * Maintains a 2D grid (10×10) keyed by behavior descriptors (aggression,
 * positioning). Each cell stores the highest-performing candidate for that
 * behavioral niche. Candidates are admitted by fitness or novelty score,
 * enabling both quality-driven and diversity-driven archive growth.
 *
 * @module
 */

/** Grid dimension for each behavior axis. */
const GRID_SIZE = 10;

/** Maximum behavior descriptor value (normalized to [0, 1]). */
const MAX_BEHAVIOR = 1.0;
void MAX_BEHAVIOR; // referenced in JSDoc; retained for API documentation

/**
 * Interface for a candidate solution stored in the archive.
 */
export interface MapElitesCandidate {
  /** Weight vector for the candidate. */
  weights: Float32Array;
  /** Fitness score (higher is better). */
  fitness: number;
  /** Behavior descriptors in [0, 1]. */
  behaviorMetrics: {
    aggression: number;
    positioning: number;
    movementPattern?: number;
  };
  /** Optional novelty score for novelty-based admission. */
  noveltyScore?: number;
  /** Optional generation tag. */
  generation?: number;
}

/**
 * MAP-Elites archive state containing the 2D grid and occupancy metadata.
 */
export interface MapElitesArchive {
  /** 10×10 grid of candidates (null for empty cells). */
  grid: (MapElitesCandidate | null)[][];
}

/**
 * Creates a fresh MAP-Elites archive with a 10×10 grid of empty cells.
 *
 * @returns A new archive with all cells initialized to null.
 */
export function createMapElitesArchive(): MapElitesArchive {
  const grid: (MapElitesCandidate | null)[][] = [];
  for (let i = 0; i < GRID_SIZE; i++) {
    const row: (MapElitesCandidate | null)[] = [];
    for (let j = 0; j < GRID_SIZE; j++) {
      row.push(null);
    }
    grid.push(row);
  }
  return { grid };
}

/**
 * Maps a behavior descriptor value in [0, 1] to a grid index in [0, 9].
 *
 * @param value Behavior descriptor value.
 * @returns Grid index clamped to [0, GRID_SIZE - 1].
 */
function behaviorToIndex(value: number): number {
  const index = Math.floor(value * GRID_SIZE);
  if (index < 0) return 0;
  if (index >= GRID_SIZE) return GRID_SIZE - 1;
  return index;
}

/**
 * Computes the admission score for a candidate, using novelty score when
 * available, otherwise falling back to fitness.
 *
 * @param candidate The candidate to score.
 * @returns The admission score (higher is better).
 */
function admissionScore(candidate: MapElitesCandidate): number {
  if (typeof candidate.noveltyScore === 'number') {
    return Math.max(candidate.fitness, candidate.noveltyScore);
  }
  return candidate.fitness;
}

/**
 * Admits a candidate into the archive. If the target cell is empty, the
 * candidate is placed directly. If occupied, the candidate replaces the
 * existing entry only if it has a higher admission score.
 *
 * @param archive The MAP-Elites archive.
 * @param candidate The candidate to admit.
 */
export function addToMapElitesArchive(
  archive: MapElitesArchive,
  candidate: MapElitesCandidate,
): void {
  const row = behaviorToIndex(candidate.behaviorMetrics.aggression);
  const col = behaviorToIndex(candidate.behaviorMetrics.positioning);
  const existing = archive.grid[row][col];
  if (existing === null) {
    archive.grid[row][col] = candidate;
  } else if (admissionScore(candidate) > admissionScore(existing)) {
    archive.grid[row][col] = candidate;
  }
}

/**
 * Computes a novelty score for a candidate relative to the occupied cells in
 * the archive. Novelty is the average Euclidean distance in behavior space to
 * all occupied cells.
 *
 * @param archive The MAP-Elites archive.
 * @param candidate The candidate whose novelty to compute.
 * @returns Novelty score in [0, ∞) (typically near [0, 1]).
 */
export function computeNoveltyForArchive(
  archive: MapElitesArchive,
  candidate: MapElitesCandidate,
): number {
  const occupied: MapElitesCandidate[] = [];
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      if (archive.grid[i][j] !== null) {
        occupied.push(archive.grid[i][j]!);
      }
    }
  }
  if (occupied.length === 0) {
    return 1.0;
  }
  let totalDistance = 0;
  for (const entry of occupied) {
    const dA =
      candidate.behaviorMetrics.aggression - entry.behaviorMetrics.aggression;
    const dP =
      candidate.behaviorMetrics.positioning - entry.behaviorMetrics.positioning;
    totalDistance += Math.sqrt(dA * dA + dP * dP);
  }
  return totalDistance / occupied.length;
}

/**
 * Returns the number of occupied cells in the archive.
 *
 * @param archive The MAP-Elites archive.
 * @returns Count of non-null cells.
 */
export function getArchiveOccupiedCount(archive: MapElitesArchive): number {
  let count = 0;
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      if (archive.grid[i][j] !== null) {
        count++;
      }
    }
  }
  return count;
}

/**
 * Samples a specified number of candidates from occupied cells in the archive.
 * If fewer cells are occupied than requested, returns all occupied entries.
 *
 * @param archive The MAP-Elites archive.
 * @param count The number of samples to return.
 * @returns Array of sampled candidates.
 */
export function sampleFromArchive(
  archive: MapElitesArchive,
  count: number,
): MapElitesCandidate[] {
  const occupied: MapElitesCandidate[] = [];
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      if (archive.grid[i][j] !== null) {
        occupied.push(archive.grid[i][j]!);
      }
    }
  }
  if (occupied.length <= count) {
    return occupied;
  }
  // Simple deterministic sampling: take evenly spaced entries.
  const samples: MapElitesCandidate[] = [];
  const step = occupied.length / count;
  for (let i = 0; i < count; i++) {
    const idx = Math.floor(i * step);
    samples.push(occupied[idx]);
  }
  return samples;
}
