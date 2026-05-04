import {
  buildProgressChecks,
  buildProgressSummary,
  parseCliOptions,
  resolveFailedProgressChecks,
  type GenerationProgressSnapshot,
} from './probeFlappyArchitectureProgress';

const SAMPLE_GENERATION_SNAPSHOTS: readonly GenerationProgressSnapshot[] = [
  {
    bestFitness: 100,
    displayFitness: 90,
    displayFramesSurvived: 40,
    displayPipesPassed: 0,
    generation: 1,
    validationMeanFramesSurvived: 45,
    validationMeanPipesPassed: 0,
    validationRobustFitness: 80,
  },
  {
    bestFitness: 200,
    displayFitness: 180,
    displayFramesSurvived: 80,
    displayPipesPassed: 2,
    generation: 2,
    validationMeanFramesSurvived: 85,
    validationMeanPipesPassed: 1,
    validationRobustFitness: 160,
  },
  {
    bestFitness: 300,
    displayFitness: 270,
    displayFramesSurvived: 100,
    displayPipesPassed: 3,
    generation: 3,
    validationMeanFramesSurvived: 105,
    validationMeanPipesPassed: 2,
    validationRobustFitness: 240,
  },
] as const;

describe('parseCliOptions', () => {
  it('resolves one approved Flappy architecture profile from CLI arguments', () => {
    expect(
      parseCliOptions([
        '--profile=gru',
        '--generations=12',
        '--report-every=4',
        '--validation-seeds=2',
      ]),
    ).toMatchObject({
      architectureProfileId: 'gru',
      generationCount: 12,
      reportEvery: 4,
      validationSeedCount: 2,
    });
  });
});

describe('buildProgressSummary', () => {
  it('keeps the profile id and midpoint phase split in the final summary', () => {
    expect(
      buildProgressSummary('narx', SAMPLE_GENERATION_SNAPSHOTS, {
        architectureProfileId: 'narx',
        displaySeed: 77,
        generationCount: 3,
        reportEvery: 1,
        requirePass: false,
        validationSeedCount: 2,
        workerInitSeed: 99,
      }),
    ).toEqual({
      architectureProfileId: 'narx',
      displaySeed: 77,
      earlyPhaseAverageDisplayFrames: 40,
      earlyPhaseAverageDisplayPipes: 0,
      finalBestFitness: 300,
      finalMeanFramesSurvived: 105,
      finalMeanPipesPassed: 2,
      firstPipeGeneration: 2,
      generationCount: 3,
      latePhaseAverageDisplayFrames: 90,
      latePhaseAverageDisplayPipes: 2.5,
      maxDisplayPipesPassed: 3,
      maxValidationMeanPipesPassed: 2,
      peakDisplayGeneration: 3,
      validationSeedCount: 2,
      workerInitSeed: 99,
    });
  });
});

describe('buildProgressChecks', () => {
  it('preserves the existing long-run check vocabulary and failed-check order', () => {
    const progressSummary = buildProgressSummary(
      'mlp',
      SAMPLE_GENERATION_SNAPSHOTS,
      {
        architectureProfileId: 'mlp',
        displaySeed: 10,
        generationCount: 3,
        reportEvery: 1,
        requirePass: false,
        validationSeedCount: 2,
        workerInitSeed: 11,
      },
    );
    const progressChecks = buildProgressChecks(progressSummary);

    expect({
      failedChecks: resolveFailedProgressChecks(progressChecks),
      progressChecks,
    }).toEqual({
      failedChecks: [],
      progressChecks: {
        clearsFirstPipe: true,
        finishesWithStableMultiSeedImprovement: true,
        improvesOverOwnEarlyFrames: true,
        improvesOverOwnEarlyPipePressure: true,
      },
    });
  });

  it('treats sustained near-ceiling durable performance as passing the early-frame check', () => {
    expect(
      buildProgressChecks({
        architectureProfileId: 'random-sparse',
        displaySeed: 10,
        earlyPhaseAverageDisplayFrames: 5_000,
        earlyPhaseAverageDisplayPipes: 63,
        finalBestFitness: 3_444.789611489017,
        finalMeanFramesSurvived: 4_329.333333333333,
        finalMeanPipesPassed: 53,
        firstPipeGeneration: 1,
        generationCount: 12,
        latePhaseAverageDisplayFrames: 4_253.5,
        latePhaseAverageDisplayPipes: 53.166666666666664,
        maxDisplayPipesPassed: 63,
        maxValidationMeanPipesPassed: 63,
        peakDisplayGeneration: 1,
        validationSeedCount: 3,
        workerInitSeed: 11,
      }),
    ).toMatchObject({
      finishesWithStableMultiSeedImprovement: true,
      improvesOverOwnEarlyFrames: true,
    });
  });

  it('treats strong early-peak durable performance as passing the shared progress gate', () => {
    expect(
      buildProgressChecks({
        architectureProfileId: 'mlp',
        displaySeed: 10,
        earlyPhaseAverageDisplayFrames: 4_261.333333333333,
        earlyPhaseAverageDisplayPipes: 52,
        finalBestFitness: 18_629.772219235623,
        finalMeanFramesSurvived: 3_953.3333333333335,
        finalMeanPipesPassed: 47.333333333333336,
        firstPipeGeneration: 1,
        generationCount: 12,
        latePhaseAverageDisplayFrames: 3_726.6666666666665,
        latePhaseAverageDisplayPipes: 44,
        maxDisplayPipesPassed: 63,
        maxValidationMeanPipesPassed: 60,
        peakDisplayGeneration: 1,
        validationSeedCount: 3,
        workerInitSeed: 11,
      }),
    ).toMatchObject({
      finishesWithStableMultiSeedImprovement: true,
      improvesOverOwnEarlyFrames: true,
    });
  });

  it('keeps the early-frame check red when late performance falls off below the durable ceiling', () => {
    expect(
      buildProgressChecks({
        architectureProfileId: 'random-sparse',
        displaySeed: 10,
        earlyPhaseAverageDisplayFrames: 5_000,
        earlyPhaseAverageDisplayPipes: 63,
        finalBestFitness: 12_000,
        finalMeanFramesSurvived: 3_200,
        finalMeanPipesPassed: 20,
        firstPipeGeneration: 1,
        generationCount: 12,
        latePhaseAverageDisplayFrames: 3_000,
        latePhaseAverageDisplayPipes: 20,
        maxDisplayPipesPassed: 63,
        maxValidationMeanPipesPassed: 30,
        peakDisplayGeneration: 1,
        validationSeedCount: 3,
        workerInitSeed: 11,
      }),
    ).toMatchObject({
      finishesWithStableMultiSeedImprovement: false,
      improvesOverOwnEarlyFrames: false,
    });
  });
});
