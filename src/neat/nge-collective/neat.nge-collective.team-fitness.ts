/**
 * @module neat.nge-collective.team-fitness
 *
 * Reusable team-level fitness evaluator seam for NGE Phase G collective benchmarks.
 *
 * This module provides the core orchestration that maps generic team groups into
 * reusable team-fitness results using an injected aggregation policy. NGE core owns
 * the fold structure; each benchmark consumer injects its own scoring rule so that
 * collective and multi-agent benchmarks share one evaluator contract without
 * hard-coding benchmark-local compensation into the core.
 *
 * ## Design contract
 *
 * ```mermaid
 * flowchart LR
 *   classDef core fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
 *   classDef consumer fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *   classDef result fill:#001522,stroke:#00c47a,color:#b8ffdf,stroke-width:1.5px;
 *
 *   ConsumerA["Consumer A\n(example benchmark)"]:::consumer -->|"inject best-finisher policy"| Core
 *   ConsumerB["Consumer B\n(example benchmark)"]:::consumer -->|"inject collective-score policy"| Core
 *   Core["createTeamFitnessEvaluator\nNGE core seam"]:::core -->|"folds each group once"| Output
 *   Output["TeamFitnessResult[ ]\nteamId · memberResults · teamFitness"]:::result
 * ```
 *
 * Consumer A and Consumer B are representative benchmark consumers: each converts
 * its own result shape into the generic `TeamResultGroup` shape and resolves team
 * fitness through this evaluator with a consumer-local aggregation policy. Any future
 * benchmark follows the same injection pattern without modifying the core fold.
 *
 * ## What this module does NOT include
 *
 * This module closes the team/group aggregation seam only. The following related
 * primitives remain incomplete in later NGE phases and must not be assumed available
 * through this contract:
 *
 * - **Generation barriers** — owner: worker-protocol phase (Phase 3+); not part of this seam.
 * - **Deterministic race-pack transport** — owner boundary still provisional (Phase 4).
 * - **Richer coevolution barrier semantics** — Phase 4+ concern; outside this boundary.
 *
 * Benchmark consumers that depend on those missing primitives must not route them
 * through this evaluator or paper over them with demo-local compensation.
 */
import type {
  TeamFitnessPolicy,
  TeamFitnessResult,
  TeamResultGroup,
} from './neat.nge-collective.types';

/**
 * Build a reusable team-level fitness evaluator for collective benchmarks.
 *
 * The returned evaluator preserves the planning seam: NGE core owns the
 * orchestration that maps generic team groups into reusable team-fitness
 * results, while each benchmark injects its own aggregation policy. That keeps
 * collective and multi-agent consumers on one shared evaluator contract
 * without leaking benchmark-local compensation into the core.
 *
 * @typeParam TTeamId - Stable identifier for the team being scored.
 * @typeParam TMemberResult - Benchmark- or policy-specific member result shape.
 * @param policy - Consumer-owned aggregation rule for one team group.
 * @returns Evaluator that folds each team group into one reusable team-fitness result.
 * @example
 * ```ts
 * const evaluateTeamFitness = createTeamFitnessEvaluator((group) =>
 *   group.memberResults.reduce(
 *     (totalScore, memberResult) => totalScore + memberResult.rawScore,
 *     0,
 *   ),
 * );
 *
 * const result = evaluateTeamFitness([
 *   {
 *     teamId: 'team-alpha',
 *     memberResults: [
 *       { memberId: 'alpha-0', rawScore: 4 },
 *       { memberId: 'alpha-1', rawScore: 6 },
 *     ],
 *   },
 * ]);
 *
 * result[0]?.teamFitness; // 10
 * ```
 */
export function createTeamFitnessEvaluator<
  TTeamId extends string,
  TMemberResult,
>(
  policy: TeamFitnessPolicy<TTeamId, TMemberResult>,
): (
  groups: readonly TeamResultGroup<TTeamId, TMemberResult>[],
) => readonly TeamFitnessResult<TTeamId, TMemberResult>[] {
  return function evaluateTeamFitness(
    groups: readonly TeamResultGroup<TTeamId, TMemberResult>[],
  ): readonly TeamFitnessResult<TTeamId, TMemberResult>[] {
    // Step 1: Preserve group ordering while applying one policy-driven fold per team.
    return groups.map((group) => buildTeamFitnessResult(group, policy));
  };
}

/**
 * Create one immutable team-fitness result from one generic team group.
 *
 * @typeParam TTeamId - Stable identifier for the team being scored.
 * @typeParam TMemberResult - Benchmark- or policy-specific member result shape.
 * @param group - Team-local member results.
 * @param policy - Consumer-owned aggregation rule for this group.
 * @returns Reusable team-fitness result preserving the original member slice.
 */
function buildTeamFitnessResult<TTeamId extends string, TMemberResult>(
  group: TeamResultGroup<TTeamId, TMemberResult>,
  policy: TeamFitnessPolicy<TTeamId, TMemberResult>,
): TeamFitnessResult<TTeamId, TMemberResult> {
  // Step 1: Run the injected policy exactly once for the current team group.
  const teamFitness = policy(group);

  return {
    teamId: group.teamId,
    memberResults: group.memberResults,
    teamFitness,
  };
}
