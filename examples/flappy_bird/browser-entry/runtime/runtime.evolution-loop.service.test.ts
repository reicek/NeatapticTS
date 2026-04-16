import {
  resolveGenerationSummaryHudValues,
  resolveEvolutionWaitLegendText,
  resolveGenerationPopulationSize,
} from './runtime.evolution-loop.service';

describe('resolveGenerationPopulationSize', () => {
  it('prefers the actual generation population length when the worker payload provides the full population', () => {
    expect(
      resolveGenerationPopulationSize([
        {} as never,
        {} as never,
        {} as never,
        {} as never,
        {} as never,
        {} as never,
        {} as never,
        {} as never,
      ], 30),
    ).toBe(8);
  });

  it('falls back to the startup budget when no generation population cache is available', () => {
    expect(resolveGenerationPopulationSize([], 30)).toBe(30);
  });
});

describe('resolveEvolutionWaitLegendText', () => {
  it('renders a natural-sounding overlay that names the generation being evolved', () => {
    expect(resolveEvolutionWaitLegendText(7)).toBe(
      'Evolving Gen 7...',
    );
  });
});

describe('resolveGenerationSummaryHudValues', () => {
  it('seeds summary placeholders until playback finishes', () => {
    expect(
      resolveGenerationSummaryHudValues({
        architectureLabel: 'NARX (14 nodes, 22 connections)',
        bestFitness: 4187,
      }),
    ).toEqual({
      summaryHeader: 'Generation summary',
      summaryFitness: '4187',
      summaryWinnerFrames: '-',
      summaryWinnerPipes: '-',
      summaryAveragePipes: '-',
      summaryP90Frames: '-',
      summaryArchitecture: 'NARX (14 nodes, 22 connections)',
    });
  });

  it('formats completed playback summary values for the HUD', () => {
    expect(
      resolveGenerationSummaryHudValues({
        architectureLabel: 'Sparse (10 nodes, 16 connections)',
        bestFitness: 6123,
        playbackSummary: {
          averagePipesPassed: 11.25,
          p90FramesSurvived: 2700,
          winnerFramesSurvived: 2700,
          winnerPipesPassed: 23,
        },
      }),
    ).toEqual({
      summaryHeader: 'Generation summary',
      summaryFitness: '6123',
      summaryWinnerFrames: '2700',
      summaryWinnerPipes: '23',
      summaryAveragePipes: '11.25',
      summaryP90Frames: '2700',
      summaryArchitecture: 'Sparse (10 nodes, 16 connections)',
    });
  });
});