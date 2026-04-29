import { FLAPPY_STATS_ROWS } from './constants.stats';

describe('FLAPPY_STATS_ROWS', () => {
  it('prioritizes live and generation-summary rows without duplicate max counters', () => {
    expect(FLAPPY_STATS_ROWS.map(({ key, label }) => ({ key, label }))).toEqual(
      [
        { key: 'currentHeader', label: 'Current run' },
        { key: 'currentFrames', label: 'Frames' },
        { key: 'currentPipes', label: 'Pipes' },
        { key: 'currentArchitecture', label: 'NN architecture' },
        { key: 'summaryHeader', label: 'Generation summary' },
        { key: 'summaryFitness', label: 'Fitness' },
        { key: 'summaryWinnerFrames', label: 'Winner frames' },
        { key: 'summaryWinnerPipes', label: 'Winner pipes' },
        { key: 'summaryAveragePipes', label: 'Avg pipes' },
        { key: 'summaryP90Frames', label: 'P90 frames' },
        { key: 'summaryArchitecture', label: 'NN architecture' },
        { key: 'telemetryHeader', label: 'Instrumentation' },
        { key: 'telemetryActivationsPerFrame', label: 'Act/frame' },
        { key: 'telemetrySimulationStepsPerRaf', label: 'Steps/RAF' },
        { key: 'telemetryHudUpdatesPerSecond', label: 'HUD upd/s' },
        { key: 'telemetryMinorGcPerMinute', label: 'Minor GC/min' },
        { key: 'status', label: 'Status' },
        { key: 'birds', label: 'Birds' },
      ],
    );
  });
});
