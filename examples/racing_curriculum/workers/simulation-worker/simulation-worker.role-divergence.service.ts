/**
 * Role-divergence observables for Tier 5 3v3 coevolution.
 *
 * Computes per-car metrics that quantify how each car's individual performance
 * relates to its team's outcome. These metrics are observability-only — they
 * do NOT change fitness or reproduction. The queen selection function in the
 * coevolution service handles reproductive consequences separately.
 *
 * Key concepts:
 * - **blockerDelta**: leave-one-out contribution to the team's best-finishing
 *   position. Computed as `teamBestWithCar - teamBestWithoutCar`. A non-zero
 *   delta means removing this car would change the team's best-finishing
 *   position. The queen (best finisher) always has a non-zero delta because
 *   removing her exposes the next-best finisher. Blockers (worst finishers)
 *   typically have a zero delta because their removal does not affect the
 *   team's best position.
 * - **inferredRole**: heuristic classification based on within-team finishing
 *   rank and team win/loss/tie status:
 *   - `queen` — best (lowest) individual finishing position on the team.
 *   - `blocker` — worst (highest) individual position on a winning or tied team.
 *   - `pacer` — mid-range individual position on the team.
 *   - `undifferentiated` — worst finisher on a losing team (no blocker role).
 *
 * These metrics complement the best-finishing-position queen-selection policy:
 * queen selection rewards the winning car's DNA, while role-divergence
 * observables provide visibility into role specialization without altering the
 * fitness landscape.
 */

/**
 * Per-car role-divergence metric for Tier 5 3v3 racing.
 *
 * @property carIndex - Index of the car in the race pack (0–5 for 6-car packs).
 * @property teamId - 0 for Team A, 1 for Team B.
 * @property individualPosition - The car's finishing position (1 = first place).
 * @property blockerDelta - Leave-one-out contribution: team best position with
 *   this car minus team best position without this car. Non-zero means this car
 *   is the team's best finisher (queen); zero means removing this car does not
 *   change the team's best position.
 * @property inferredRole - Heuristic role classification.
 */
export type RoleDivergenceMetric = {
  readonly carIndex: number;
  readonly teamId: 0 | 1;
  readonly individualPosition: number;
  readonly blockerDelta: number;
  readonly inferredRole: 'queen' | 'blocker' | 'pacer' | 'undifferentiated';
};

/**
 * Computes per-car role-divergence metrics for a finished race pack.
 *
 * This function is observability-only: it does NOT change fitness scores or
 * trigger reproduction. Use `selectQueenPerTeam` from the coevolution service
 * for queen-based reproduction wiring.
 *
 * @param carFinishPositions - Finish positions for all cars, indexed by carIndex.
 *   Lower numbers are better (1 = first place).
 * @param teamLayout - Team assignment per car (0 for Team A, 1 for Team B).
 * @param teamScores - Team scores indexed by teamId. Higher scores are better.
 *   Used to determine the winning team for role classification.
 * @returns One `RoleDivergenceMetric` per car, ordered by carIndex.
 *
 * @example
 * ```ts
 * // 6-car Tier 5 pack: Team A cars 0,1,2 finish at positions 1,4,6
 * // Team B cars 3,4,5 finish at positions 2,3,5
 * // Both teams score 10 (tied)
 * const metrics = computeRoleDivergenceMetrics(
 *   [1, 4, 6, 2, 3, 5],
 *   [0, 0, 0, 1, 1, 1],
 *   [10, 10],
 * );
 * // metrics[0].inferredRole === 'queen'   (best Team A finisher)
 * // metrics[2].inferredRole === 'blocker' (worst Team A finisher, tied)
 * // metrics[1].inferredRole === 'pacer'   (mid Team A finisher)
 * ```
 */
export function computeRoleDivergenceMetrics(
  carFinishPositions: readonly number[],
  teamLayout: readonly (0 | 1)[],
  teamScores: readonly number[],
): readonly RoleDivergenceMetric[] {
  // Step 1: Group car indices by team.
  const teamCarIndices: number[][] = [[], []];
  for (let carIndex = 0; carIndex < carFinishPositions.length; carIndex++) {
    const teamId = teamLayout[carIndex] ?? 0;
    teamCarIndices[teamId].push(carIndex);
  }

  // Step 2: Compute metrics for each car.
  const metrics: RoleDivergenceMetric[] = [];
  for (let carIndex = 0; carIndex < carFinishPositions.length; carIndex++) {
    const teamId = (teamLayout[carIndex] ?? 0) as 0 | 1;
    const individualPosition = carFinishPositions[carIndex];
    const blockerDelta = computeBlockerDelta(
      carFinishPositions,
      teamCarIndices[teamId],
      carIndex,
    );
    const inferredRole = inferRole(
      carIndex,
      teamCarIndices[teamId],
      carFinishPositions,
      teamId,
      teamScores,
    );

    metrics.push({
      carIndex,
      teamId,
      individualPosition,
      blockerDelta,
      inferredRole,
    });
  }

  return metrics;
}

/**
 * Computes the leave-one-out blockerDelta for a single car.
 *
 * blockerDelta = teamBestWithCar - teamBestWithoutCar.
 *
 * The queen (best finisher) has a non-zero delta because removing her exposes
 * the next-best finisher. Blockers (worst finishers) typically have a zero
 * delta because their removal does not change the team's best position.
 *
 * @param carFinishPositions - Finish positions for all cars.
 * @param teamCarIndices - Car indices on this car's team (including this car).
 * @param carIndex - The car being evaluated.
 * @returns The blockerDelta (0 when removing the car does not change the
 *   team's best position; non-zero when the car is the team's best finisher).
 */
function computeBlockerDelta(
  carFinishPositions: readonly number[],
  teamCarIndices: readonly number[],
  carIndex: number,
): number {
  // Step 1: Find the team's best (lowest) finishing position with this car.
  const teamPositionsWithCar = teamCarIndices.map(
    (index) => carFinishPositions[index] ?? Number.POSITIVE_INFINITY,
  );
  const bestWithCar = Math.min(...teamPositionsWithCar);

  // Step 2: Find the team's best finishing position without this car.
  const positionsWithoutCar = teamCarIndices
    .filter((index) => index !== carIndex)
    .map((index) => carFinishPositions[index] ?? Number.POSITIVE_INFINITY);
  const bestWithoutCar =
    positionsWithoutCar.length > 0
      ? Math.min(...positionsWithoutCar)
      : Number.POSITIVE_INFINITY;

  // Step 3: Delta = how much the best position changes when this car is removed.
  return bestWithCar - bestWithoutCar;
}

/**
 * Infers a car's role based on within-team finishing rank and team outcome.
 *
 * Role assignment logic:
 * - `queen` — best (lowest) individual finishing position on the team.
 * - `blocker` — worst (highest) individual position on a winning or tied team.
 * - `pacer` — mid-range individual position (not best, not worst).
 * - `undifferentiated` — worst finisher on a losing team.
 *
 * @param carIndex - The car being classified.
 * @param teamCarIndices - Car indices on this car's team.
 * @param carFinishPositions - Finish positions for all cars.
 * @param teamId - This car's team ID (0 or 1).
 * @param teamScores - Team scores indexed by teamId.
 * @returns The inferred role string.
 */
function inferRole(
  carIndex: number,
  teamCarIndices: readonly number[],
  carFinishPositions: readonly number[],
  teamId: 0 | 1,
  teamScores: readonly number[],
): 'queen' | 'blocker' | 'pacer' | 'undifferentiated' {
  // Step 1: Sort team car indices by finishing position (ascending = better).
  const sortedTeamCars = teamCarIndices
    .map((index) => ({
      carIndex: index,
      position: carFinishPositions[index] ?? Number.POSITIVE_INFINITY,
    }))
    .toSorted((a, b) => a.position - b.position);

  // Step 2: Find this car's rank within the team (1-based).
  const rank =
    sortedTeamCars.findIndex((entry) => entry.carIndex === carIndex) + 1;
  const teamSize = sortedTeamCars.length;

  // Step 3: Determine team outcome.
  const otherTeamId = (teamId === 0 ? 1 : 0) as 0 | 1;
  const teamScore = teamScores[teamId] ?? 0;
  const otherTeamScore = teamScores[otherTeamId] ?? 0;
  const teamWon = teamScore > otherTeamScore;
  const teamTied = teamScore === otherTeamScore;

  // Step 4: Assign role based on rank and team outcome.
  if (rank === 1) {
    return 'queen';
  }
  if (rank === teamSize) {
    // Worst finisher on team — blocker if team won or tied, undifferentiated if lost.
    if (teamWon || teamTied) {
      return 'blocker';
    }
    return 'undifferentiated';
  }
  // Mid-range finisher.
  return 'pacer';
}
