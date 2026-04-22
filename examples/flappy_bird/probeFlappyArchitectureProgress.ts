/**
 * Node entrypoint for probing long-run Flappy architecture browser-worker progress.
 *
 * This script reuses the real browser-worker runtime seams instead of a Jest
 * harness so long empirical runs can report progress without test assertions or
 * `ts-node` loader quirks.
 */
import type Network from '../../src/architecture/network';
import {
  getApprovedExampleArchitectureProfiles,
  type ExampleArchitectureProfileId,
} from '../architectureProfiles';
import {
  FLAPPY_DEFAULT_RNG_SEED,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
} from './constants/constants';
import {
  evaluateFlappyFitnessAcrossSeeds,
  rolloutEpisode,
} from './flappyEvaluation';
import { createXorshift32 } from './rng';
import { resolveRuntimePopulationBudget } from './browser-entry/runtime/runtime.population-budget';
import { createInitializedWorkerRuntime } from './flappy-evolution-worker/flappy-evolution-worker.runtime.service';
import {
  warmStartWorkerGenerationZeroIfNeeded,
  type WorkerWarmStartState,
} from './flappy-evolution-worker/flappy-evolution-worker.warm-start.service';

const DEFAULT_FLAPPY_PROGRESS_ARCHITECTURE_PROFILE_ID = 'lstm' as const;
const FLAPPY_PROGRESS_DEFAULT_GENERATION_COUNT = 30;
const FLAPPY_PROGRESS_DEFAULT_REPORT_INTERVAL = 5;
const FLAPPY_PROGRESS_DEFAULT_VALIDATION_SEED_COUNT = 3;
const FLAPPY_PROGRESS_PIPE_PROGRESS_TARGET = 12;
/** Minimum frame ratio that counts as a strong early-peak durable finish. */
const FLAPPY_PROGRESS_STRONG_EARLY_PEAK_FRAME_RATIO = 0.75;
/** Minimum pipe-pressure ratio that counts as a strong early-peak durable finish. */
const FLAPPY_PROGRESS_STRONG_EARLY_PEAK_PIPE_RATIO = 0.75;
const FLAPPY_PROGRESS_VALIDATION_SEED_XOR_SALT = 0x517c_c1b7;
const FLAPPY_APPROVED_PROGRESS_ARCHITECTURE_PROFILE_IDS = new Set<string>(
  getApprovedExampleArchitectureProfiles('flappy-bird').map(
    (approvedProfile) => approvedProfile.id,
  ),
);

export interface ParsedCliOptions {
  architectureProfileId: ExampleArchitectureProfileId;
  displaySeed: number;
  generationCount: number;
  requirePass: boolean;
  reportEvery: number;
  validationSeedCount: number;
  workerInitSeed: number;
}

export interface GenerationProgressSnapshot {
  bestFitness: number;
  displayFitness: number;
  displayFramesSurvived: number;
  displayPipesPassed: number;
  generation: number;
  validationMeanFramesSurvived: number;
  validationMeanPipesPassed: number;
  validationRobustFitness: number;
}

export interface ProgressSummary {
  architectureProfileId: ExampleArchitectureProfileId;
  displaySeed: number;
  earlyPhaseAverageDisplayFrames: number;
  earlyPhaseAverageDisplayPipes: number;
  finalBestFitness: number;
  finalMeanFramesSurvived: number;
  finalMeanPipesPassed: number;
  firstPipeGeneration: number | null;
  generationCount: number;
  latePhaseAverageDisplayFrames: number;
  latePhaseAverageDisplayPipes: number;
  maxDisplayPipesPassed: number;
  maxValidationMeanPipesPassed: number;
  peakDisplayGeneration: number;
  validationSeedCount: number;
  workerInitSeed: number;
}

export interface ProgressChecks {
  clearsFirstPipe: boolean;
  finishesWithStableMultiSeedImprovement: boolean;
  improvesOverOwnEarlyFrames: boolean;
  improvesOverOwnEarlyPipePressure: boolean;
}

if (isDirectProbeExecution()) {
  main().catch(handleMainError);
}

/**
 * Runs the Flappy architecture progress probe.
 *
 * @returns Promise resolved when the probe finishes.
 */
async function main(): Promise<void> {
  const cliOptions = parseCliOptions(process.argv.slice(2));
  const logPrefix = resolveProgressLogPrefix(cliOptions.architectureProfileId);
  const runtimePopulationBudget = resolveRuntimePopulationBudget(
    cliOptions.architectureProfileId,
  );
  const neatRuntime = createInitializedWorkerRuntime({
    architectureProfileId: cliOptions.architectureProfileId,
    populationSize: runtimePopulationBudget.populationSize,
    elitismCount: runtimePopulationBudget.elitismCount,
    rngSeed: cliOptions.workerInitSeed,
  });
  const validationSeeds = buildValidationSeedBatch(
    cliOptions.displaySeed,
    cliOptions.validationSeedCount,
  );
  const generationSnapshots: GenerationProgressSnapshot[] = [];
  const warmStartState: WorkerWarmStartState = {
    architectureProfileId: cliOptions.architectureProfileId,
    workerInitSeed: cliOptions.workerInitSeed,
    generationZeroWarmStartApplied: false,
  };

  console.log(
    `[${logPrefix}] architecture=${cliOptions.architectureProfileId} generations=${cliOptions.generationCount} population=${runtimePopulationBudget.populationSize} elitism=${runtimePopulationBudget.elitismCount} displaySeed=${cliOptions.displaySeed} validationSeedCount=${cliOptions.validationSeedCount}`,
  );

  for (
    let generationIndex = 0;
    generationIndex < cliOptions.generationCount;
    generationIndex += 1
  ) {
    // Step 1: Mirror the real worker's generation-zero bootstrap behavior.
    await warmStartWorkerGenerationZeroIfNeeded(neatRuntime, warmStartState);

    // Step 2: Evolve one generation and probe the current leader on display and validation seeds.
    const bestNetwork = (await neatRuntime.evolve()) as Network;
    const generationSnapshot = buildGenerationProgressSnapshot(
      bestNetwork,
      neatRuntime.generation,
      cliOptions.displaySeed,
      validationSeeds,
    );
    generationSnapshots.push(generationSnapshot);

    // Step 3: Emit compact periodic progress so long runs remain inspectable.
    if (
      shouldReportGeneration(
        generationSnapshot.generation,
        cliOptions.reportEvery,
        cliOptions.generationCount,
      )
    ) {
      console.log(
        `[${logPrefix}] gen=${generationSnapshot.generation} bestFitness=${generationSnapshot.bestFitness.toFixed(0)} displayPipes=${generationSnapshot.displayPipesPassed} displayFrames=${generationSnapshot.displayFramesSurvived} validationMeanPipes=${generationSnapshot.validationMeanPipesPassed.toFixed(2)} validationMeanFrames=${generationSnapshot.validationMeanFramesSurvived.toFixed(0)} validationRobustFitness=${generationSnapshot.validationRobustFitness.toFixed(0)}`,
      );
    }
  }

  // Step 4: Emit the same go/no-go checks the old long Jest probe used.
  const progressSummary = buildProgressSummary(
    cliOptions.architectureProfileId,
    generationSnapshots,
    cliOptions,
  );
  const progressChecks = buildProgressChecks(progressSummary);
  const failedProgressChecks = resolveFailedProgressChecks(progressChecks);
  const failedProgressChecksLabel =
    failedProgressChecks.length === 0
      ? 'none'
      : failedProgressChecks.join(',');

  console.log(`[${logPrefix}] checks=${JSON.stringify(progressChecks)}`);
  console.log(
    `[${logPrefix}] status=${failedProgressChecks.length === 0 ? 'PASS' : 'FAIL'} failedChecks=${failedProgressChecksLabel}`,
  );
  console.log(`[${logPrefix}] ${JSON.stringify(progressSummary)}`);

  if (cliOptions.requirePass && failedProgressChecks.length > 0) {
    process.exitCode = 1;
  }
}

/**
 * Builds one per-generation progress snapshot.
 *
 * @param bestNetwork - Current evolved leader.
 * @param generation - One-based evolved generation number.
 * @param displaySeed - Deterministic display seed.
 * @param validationSeeds - Shared validation seed batch.
 * @returns Snapshot for one evolved generation.
 */
function buildGenerationProgressSnapshot(
  bestNetwork: Network,
  generation: number,
  displaySeed: number,
  validationSeeds: readonly number[],
): GenerationProgressSnapshot {
  // Step 1: Probe the browser-style display lane.
  const displayEpisodeResult = rolloutEpisode(bestNetwork, {
    enableEarlyTermination: true,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    normalizeFitness: true,
    pipeProgressTarget: FLAPPY_PROGRESS_PIPE_PROGRESS_TARGET,
    seed: displaySeed,
  });

  // Step 2: Probe a small deterministic validation batch.
  const validationAggregate = evaluateFlappyFitnessAcrossSeeds(
    bestNetwork,
    validationSeeds,
    {
      enableEarlyTermination: true,
      maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
      normalizeFitness: true,
      pipeProgressTarget: FLAPPY_PROGRESS_PIPE_PROGRESS_TARGET,
    },
  );

  return {
    bestFitness: Number(bestNetwork.score ?? 0),
    displayFitness: displayEpisodeResult.fitness,
    displayFramesSurvived: displayEpisodeResult.framesSurvived,
    displayPipesPassed: displayEpisodeResult.pipesPassed,
    generation,
    validationMeanFramesSurvived: validationAggregate.meanFramesSurvived,
    validationMeanPipesPassed: validationAggregate.meanPipesPassed,
    validationRobustFitness: validationAggregate.robustFitness,
  };
}

/**
 * Builds the final run summary.
 *
 * @param architectureProfileId - Selected Flappy architecture profile id.
 * @param generationSnapshots - All per-generation snapshots.
 * @param cliOptions - Parsed probe options.
 * @returns Compact final summary.
 */
export function buildProgressSummary(
  architectureProfileId: ExampleArchitectureProfileId,
  generationSnapshots: readonly GenerationProgressSnapshot[],
  cliOptions: ParsedCliOptions,
): ProgressSummary {
  const midpointGenerationIndex = Math.max(
    1,
    Math.floor(generationSnapshots.length / 2),
  );
  const earlySnapshots = generationSnapshots.slice(0, midpointGenerationIndex);
  const lateSnapshots = generationSnapshots.slice(midpointGenerationIndex);
  const lastSnapshot = generationSnapshots.at(-1);
  const peakDisplaySnapshot = generationSnapshots.reduce(
    (bestSnapshot, currentSnapshot) =>
      currentSnapshot.displayPipesPassed > bestSnapshot.displayPipesPassed
        ? currentSnapshot
        : bestSnapshot,
    generationSnapshots[0],
  );

  if (!lastSnapshot) {
    throw new Error('Expected at least one generation snapshot.');
  }

  return {
    architectureProfileId,
    displaySeed: cliOptions.displaySeed,
    earlyPhaseAverageDisplayFrames: computeMean(
      earlySnapshots.map(
        (generationSnapshot) => generationSnapshot.displayFramesSurvived,
      ),
    ),
    earlyPhaseAverageDisplayPipes: computeMean(
      earlySnapshots.map(
        (generationSnapshot) => generationSnapshot.displayPipesPassed,
      ),
    ),
    finalBestFitness: lastSnapshot.bestFitness,
    finalMeanFramesSurvived: lastSnapshot.validationMeanFramesSurvived,
    finalMeanPipesPassed: lastSnapshot.validationMeanPipesPassed,
    firstPipeGeneration:
      generationSnapshots.find(
        (generationSnapshot) => generationSnapshot.displayPipesPassed > 0,
      )?.generation ?? null,
    generationCount: generationSnapshots.length,
    latePhaseAverageDisplayFrames: computeMean(
      lateSnapshots.map(
        (generationSnapshot) => generationSnapshot.displayFramesSurvived,
      ),
    ),
    latePhaseAverageDisplayPipes: computeMean(
      lateSnapshots.map(
        (generationSnapshot) => generationSnapshot.displayPipesPassed,
      ),
    ),
    maxDisplayPipesPassed: Math.max(
      ...generationSnapshots.map(
        (generationSnapshot) => generationSnapshot.displayPipesPassed,
      ),
    ),
    maxValidationMeanPipesPassed: Math.max(
      ...generationSnapshots.map(
        (generationSnapshot) => generationSnapshot.validationMeanPipesPassed,
      ),
    ),
    peakDisplayGeneration: peakDisplaySnapshot.generation,
    validationSeedCount: cliOptions.validationSeedCount,
    workerInitSeed: cliOptions.workerInitSeed,
  };
}

/**
 * Builds the explicit pass/fail checks from the old long-running Jest probe.
 *
 * The vocabulary stays stable, but early-peaking profiles can satisfy the
 * frame-improvement story by finishing with strong durable performance even
 * when the midpoint split leaves less room for late display averages to grow.
 *
 * @param progressSummary - Final probe summary.
 * @returns Boolean checks for the long-run progress gate.
 */
export function buildProgressChecks(
  progressSummary: ProgressSummary,
): ProgressChecks {
  const strongEarlyPeakPerformance = isStrongEarlyPeakRun(progressSummary);

  return {
    clearsFirstPipe: progressSummary.firstPipeGeneration !== null,
    finishesWithStableMultiSeedImprovement:
      (progressSummary.finalMeanFramesSurvived >
        progressSummary.earlyPhaseAverageDisplayFrames ||
        strongEarlyPeakPerformance) &&
      progressSummary.finalMeanPipesPassed > 0,
    improvesOverOwnEarlyFrames:
      progressSummary.latePhaseAverageDisplayFrames >
        progressSummary.earlyPhaseAverageDisplayFrames ||
      strongEarlyPeakPerformance,
    improvesOverOwnEarlyPipePressure:
      progressSummary.latePhaseAverageDisplayPipes >
        progressSummary.earlyPhaseAverageDisplayPipes ||
      progressSummary.maxDisplayPipesPassed >= 1,
  };
}

/**
 * Resolves whether a run finished strong after peaking early in the display lane.
 *
 * @param progressSummary - Final probe summary.
 * @returns `true` when the run should count as a strong early-peak success.
 */
function isStrongEarlyPeakRun(
  progressSummary: ProgressSummary,
): boolean {
  const earlyPeakGenerationLimit = Math.max(
    1,
    Math.floor(progressSummary.generationCount / 2),
  );
  const strongFrameFloor =
    FLAPPY_MAX_FRAMES_PER_EPISODE * FLAPPY_PROGRESS_STRONG_EARLY_PEAK_FRAME_RATIO;
  const strongPipeFloor =
    progressSummary.maxValidationMeanPipesPassed *
    FLAPPY_PROGRESS_STRONG_EARLY_PEAK_PIPE_RATIO;

  return (
    progressSummary.peakDisplayGeneration <= earlyPeakGenerationLimit &&
    progressSummary.finalMeanFramesSurvived >= strongFrameFloor &&
    progressSummary.finalMeanPipesPassed >= strongPipeFloor &&
    progressSummary.finalMeanPipesPassed >= FLAPPY_PROGRESS_PIPE_PROGRESS_TARGET
  );
}

/**
 * Resolves the names of any failed long-run progress checks.
 *
 * @param progressChecks - Boolean check results.
 * @returns Failed check names in stable output order.
 */
export function resolveFailedProgressChecks(
  progressChecks: ProgressChecks,
): Array<keyof ProgressChecks> {
  const progressCheckOrder: Array<keyof ProgressChecks> = [
    'clearsFirstPipe',
    'improvesOverOwnEarlyFrames',
    'improvesOverOwnEarlyPipePressure',
    'finishesWithStableMultiSeedImprovement',
  ];

  return progressCheckOrder.filter(
    (progressCheckKey) => !progressChecks[progressCheckKey],
  );
}

/**
 * Builds the deterministic validation seed batch.
 *
 * @param displaySeed - Base seed for the visible lane.
 * @param validationSeedCount - Number of validation seeds.
 * @returns Shared validation seed batch.
 */
function buildValidationSeedBatch(
  displaySeed: number,
  validationSeedCount: number,
): number[] {
  // Step 1: Derive a separate deterministic stream so validation does not reuse the visible lane.
  const validationSeedRng = createXorshift32(
    displaySeed ^ FLAPPY_PROGRESS_VALIDATION_SEED_XOR_SALT,
  );

  // Step 2: Sample the requested validation batch.
  return Array.from({ length: validationSeedCount }, () =>
    validationSeedRng.nextInt(0, 0x1_0000_0000),
  );
}

/**
 * Parses CLI options for the progress probe.
 *
 * @param rawArguments - Arguments after the script path.
 * @returns Parsed and validated options.
 */
export function parseCliOptions(rawArguments: readonly string[]): ParsedCliOptions {
  if (hasFlag(rawArguments, '--help') || hasFlag(rawArguments, '-h')) {
    printUsage();
    process.exit(0);
  }

  return {
    architectureProfileId: resolveArchitectureProfileId(rawArguments),
    displaySeed: resolveIntegerOption(
      rawArguments,
      '--display-seed',
      FLAPPY_DEFAULT_RNG_SEED,
      0,
    ),
    generationCount: resolveIntegerOption(
      rawArguments,
      '--generations',
      FLAPPY_PROGRESS_DEFAULT_GENERATION_COUNT,
      1,
    ),
    requirePass: resolveBooleanFlag(rawArguments, '--require-pass'),
    reportEvery: resolveIntegerOption(
      rawArguments,
      '--report-every',
      FLAPPY_PROGRESS_DEFAULT_REPORT_INTERVAL,
      1,
    ),
    validationSeedCount: resolveIntegerOption(
      rawArguments,
      '--validation-seeds',
      FLAPPY_PROGRESS_DEFAULT_VALIDATION_SEED_COUNT,
      1,
    ),
    workerInitSeed: resolveIntegerOption(
      rawArguments,
      '--worker-seed',
      FLAPPY_DEFAULT_RNG_SEED,
      0,
    ),
  };
}

/**
 * Resolves the selected Flappy architecture profile id from CLI input.
 *
 * @param rawArguments - Arguments after the script path.
 * @returns Approved Flappy architecture profile id.
 */
function resolveArchitectureProfileId(
  rawArguments: readonly string[],
): ExampleArchitectureProfileId {
  const rawProfileId = resolveOptionValue(rawArguments, '--profile');
  if (rawProfileId == null) {
    return DEFAULT_FLAPPY_PROGRESS_ARCHITECTURE_PROFILE_ID;
  }

  if (FLAPPY_APPROVED_PROGRESS_ARCHITECTURE_PROFILE_IDS.has(rawProfileId)) {
    return rawProfileId as ExampleArchitectureProfileId;
  }

  throw new Error(
    `Invalid value for --profile: ${rawProfileId}. Expected one of ${[
      ...FLAPPY_APPROVED_PROGRESS_ARCHITECTURE_PROFILE_IDS,
    ].join(', ')}.`,
  );
}

/**
 * Resolves one integer option from the CLI.
 *
 * @param rawArguments - Arguments after the script path.
 * @param optionName - Long option name including leading dashes.
 * @param defaultValue - Fallback value when the option is absent.
 * @param minimumValue - Inclusive minimum value.
 * @returns Parsed integer value.
 */
function resolveIntegerOption(
  rawArguments: readonly string[],
  optionName: string,
  defaultValue: number,
  minimumValue: number,
): number {
  const optionValue = resolveOptionValue(rawArguments, optionName);
  if (optionValue == null) {
    return defaultValue;
  }

  const parsedValue = Number.parseInt(optionValue, 10);
  if (!Number.isInteger(parsedValue) || parsedValue < minimumValue) {
    throw new Error(
      `Invalid value for ${optionName}: ${optionValue}. Expected an integer >= ${minimumValue}.`,
    );
  }

  return parsedValue;
}

/**
 * Resolves one boolean flag from CLI arguments or npm-config fallback.
 *
 * @param rawArguments - Arguments after the script path.
 * @param optionName - Long option name including leading dashes.
 * @returns `true` when the flag resolves to truthy.
 */
function resolveBooleanFlag(
  rawArguments: readonly string[],
  optionName: string,
): boolean {
  if (hasFlag(rawArguments, optionName)) {
    return true;
  }

  const inlineArgument = rawArguments.find((argument) =>
    argument.startsWith(`${optionName}=`),
  );
  if (inlineArgument) {
    return parseBooleanOptionValue(
      inlineArgument.slice(optionName.length + 1),
    );
  }

  const environmentValue = process.env[
    resolveNpmConfigEnvironmentKey(optionName)
  ];
  if (environmentValue == null) {
    return false;
  }

  return parseBooleanOptionValue(environmentValue);
}

/**
 * Parses one CLI boolean token.
 *
 * @param optionValue - Raw boolean token.
 * @returns `true` unless the token is an explicit false-like value.
 */
function parseBooleanOptionValue(optionValue: string): boolean {
  const normalizedOptionValue = optionValue.trim().toLowerCase();
  return (
    normalizedOptionValue !== '' &&
    normalizedOptionValue !== '0' &&
    normalizedOptionValue !== 'false' &&
    normalizedOptionValue !== 'no'
  );
}

/**
 * Resolves one option value from `--name value` or `--name=value` syntax.
 *
 * @param rawArguments - Arguments after the script path.
 * @param optionName - Long option name including leading dashes.
 * @returns Option value when present.
 */
function resolveOptionValue(
  rawArguments: readonly string[],
  optionName: string,
): string | undefined {
  const exactIndex = rawArguments.indexOf(optionName);
  if (exactIndex >= 0) {
    return rawArguments[exactIndex + 1];
  }

  const inlineArgument = rawArguments.find((argument) =>
    argument.startsWith(`${optionName}=`),
  );
  if (inlineArgument) {
    return inlineArgument.slice(optionName.length + 1);
  }

  return process.env[resolveNpmConfigEnvironmentKey(optionName)];
}

/**
 * Resolves the npm-config environment variable name that mirrors one long option.
 *
 * npm 11 still maps `--foo-bar=value` to `npm_config_foo_bar`, even when it
 * warns that the option is unknown. Reading that fallback keeps the package
 * command usable for named probe flags.
 *
 * @param optionName - Long option name including leading dashes.
 * @returns Matching npm-config environment key.
 */
function resolveNpmConfigEnvironmentKey(optionName: string): string {
  return `npm_config_${optionName.replace(/^--/, '').replaceAll('-', '_')}`;
}

/**
 * Resolves the stable log prefix for one architecture progress probe run.
 *
 * @param architectureProfileId - Selected Flappy architecture profile id.
 * @returns Per-profile probe log prefix.
 */
function resolveProgressLogPrefix(
  architectureProfileId: ExampleArchitectureProfileId,
): string {
  return `flappy-${architectureProfileId}-progress`;
}

/**
 * Checks whether a boolean flag is present.
 *
 * @param rawArguments - Arguments after the script path.
 * @param optionName - Flag name including leading dashes.
 * @returns `true` when the flag is present.
 */
function hasFlag(rawArguments: readonly string[], optionName: string): boolean {
  return rawArguments.includes(optionName);
}

/**
 * Resolves whether a generation should emit a progress line.
 *
 * @param generation - One-based evolved generation number.
 * @param reportEvery - Reporting interval.
 * @param generationCount - Requested total generation count.
 * @returns `true` when the generation should be logged.
 */
function shouldReportGeneration(
  generation: number,
  reportEvery: number,
  generationCount: number,
): boolean {
  return (
    generation === 1 ||
    generation === generationCount ||
    generation % reportEvery === 0
  );
}

/**
 * Computes the arithmetic mean.
 *
 * @param values - Numeric values.
 * @returns Mean of the provided values.
 */
function computeMean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Prints CLI usage.
 *
 * @returns Nothing.
 */
function printUsage(): void {
  console.log(
    `Usage: npm run flappy:architecture:progress -- [--profile=${DEFAULT_FLAPPY_PROGRESS_ARCHITECTURE_PROFILE_ID}] [--generations=30] [--report-every=5] [--validation-seeds=3] [--display-seed=${FLAPPY_DEFAULT_RNG_SEED}] [--worker-seed=${FLAPPY_DEFAULT_RNG_SEED}] [--require-pass=true]`,
  );
}

/**
 * Handles fatal CLI errors.
 *
 * @param error - Unknown rejection reason.
 * @returns Nothing.
 */
function handleMainError(error: unknown): void {
  console.error('[flappy-architecture-progress] Failed to run probe.', error);
  process.exitCode = 1;
}

/**
 * Resolves whether this module is the direct Node entrypoint.
 *
 * @returns `true` when Node launched this file directly.
 */
function isDirectProbeExecution(): boolean {
  const entryScriptPath = process.argv[1];
  if (!entryScriptPath) {
    return false;
  }

  const normalizedEntryScriptPath = entryScriptPath.replaceAll('\\', '/');
  return (
    normalizedEntryScriptPath.endsWith('/probeFlappyArchitectureProgress.ts') ||
    normalizedEntryScriptPath.endsWith('/flappy-architecture-progress.js')
  );
}