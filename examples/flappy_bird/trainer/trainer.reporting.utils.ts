/**
 * Formatting helpers for compact generation-log output.
 *
 * The trainer's console line is intentionally tokenized rather than narrated.
 * This file keeps that token order stable so humans can build scanning habits
 * across long runs and tools can parse the same line shape later if needed.
 */
import type { FlappyGenerationReport } from './trainer.types';
import type { FlappyMutationSchedule } from './trainer.evaluation-plan.utils';

/**
 * Builds one-line generation log tokens.
 *
 * The chosen order moves from identity (`gen`) to quality (`best`, `mean`,
 * `median`, `p90`, `std`) and then into operational context (`difficulty`,
 * mutation, seed counts).
 *
 * @param generationLabel - Generation label shown in logs.
 * @param bestFitness - Best resolved fitness value for this generation.
 * @param bestPipesPassed - Best resolved pipes passed value.
 * @param bestFramesSurvived - Best resolved frames survived value.
 * @param report - Optional aggregated generation report.
 * @param mutationSchedule - Active mutation schedule for this generation.
 * @returns Ordered log tokens for compact console output.
 */
export function buildGenerationLogParts(
  generationLabel: number,
  bestFitness: number,
  bestPipesPassed: number,
  bestFramesSurvived: number,
  report: FlappyGenerationReport | undefined,
  mutationSchedule: FlappyMutationSchedule,
): string[] {
  return [
    `gen=${generationLabel}`,
    `best=${bestFitness.toFixed(2)}`,
    `mean=${(report?.scoreMean ?? Number.NaN).toFixed(2)}`,
    `median=${(report?.scoreMedian ?? Number.NaN).toFixed(2)}`,
    `p90=${(report?.scoreP90 ?? Number.NaN).toFixed(2)}`,
    `std=${(report?.scoreStdDev ?? Number.NaN).toFixed(2)}`,
    `pipes=${bestPipesPassed}`,
    `frames=${bestFramesSurvived}`,
    `difficulty=${(report?.difficultyScale ?? 1).toFixed(2)}`,
    `mutRate=${(report?.mutationRate ?? mutationSchedule.mutationRate).toFixed(3)}`,
    `mutAmount=${(report?.mutationAmount ?? mutationSchedule.mutationAmount).toFixed(2)}`,
    `seeds=${report?.quickSeedCount ?? 0}/${report?.fullSeedCount ?? 0}/${report?.reevaluationSeedCount ?? 0}`,
  ];
}
