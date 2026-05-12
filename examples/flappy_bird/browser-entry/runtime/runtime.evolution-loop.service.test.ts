import {
  resolveRuntimeArchitectureProgressUpdate,
  resolveGenerationSummaryHudValues,
  resolveEvolutionWaitLegendText,
  resolveGenerationPopulationSize,
  resolveRuntimeChampionCandidateNetworkJson,
  resolveWorkerInitPayload,
} from './runtime.evolution-loop.service';

function withNodeEnv<T>(nextNodeEnv: string, callback: () => T): T {
  const previousNodeEnv = process.env.NODE_ENV;
  process.env.NODE_ENV = nextNodeEnv;

  try {
    return callback();
  } finally {
    process.env.NODE_ENV = previousNodeEnv;
  }
}

describe('resolveGenerationPopulationSize', () => {
  it('prefers the actual generation population length when the worker payload provides the full population', () => {
    expect(
      resolveGenerationPopulationSize(
        [
          {} as never,
          {} as never,
          {} as never,
          {} as never,
          {} as never,
          {} as never,
          {} as never,
          {} as never,
        ],
        30,
      ),
    ).toBe(8);
  });

  it('falls back to the startup budget when no generation population cache is available', () => {
    expect(resolveGenerationPopulationSize([], 30)).toBe(30);
  });
});

describe('resolveEvolutionWaitLegendText', () => {
  it('renders a natural-sounding overlay that names the generation being evolved', () => {
    expect(resolveEvolutionWaitLegendText(7)).toBe('Evolving Gen 7...');
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

describe('resolveRuntimeArchitectureProgressUpdate', () => {
  it('stores the improved browser-local champion for the selected architecture profile when a new record is set', () => {
    const championNetworkJson = {
      connections: [{ weight: 1 }, { weight: 2 }, { weight: 3 }],
      nodes: [
        { bias: 0, type: 'input' },
        { bias: 42, type: 'hidden' },
        { bias: 1, type: 'output' },
      ],
      scope: 'champion',
    };
    const consoleInfoSpy = jest
      .spyOn(console, 'info')
      .mockImplementation(() => undefined);

    try {
      const progressUpdate = withNodeEnv('development', () =>
        resolveRuntimeArchitectureProgressUpdate({
          candidateBestScore: {
            pipesPassed: 6,
            framesSurvived: 280,
          },
          candidateChampionNetworkJson: championNetworkJson,
          championByProfileId: {
            mlp: { nodes: [{ bias: 1 }], scope: 'previous' },
          },
          historyByProfileId: {
            gru: { pipesPassed: 5, framesSurvived: 310 },
            mlp: { pipesPassed: 1, framesSurvived: 80 },
          },
          profileId: 'gru',
        }),
      );
      const saveLogLine = consoleInfoSpy.mock.calls.at(-1)?.[0];

      expect({
        didLogSave:
          typeof saveLogLine === 'string' &&
          saveLogLine.includes('saving champion profile=gru') &&
          saveLogLine.includes('pipes=6') &&
          saveLogLine.includes('frames=280') &&
          saveLogLine.includes('previousPipes=5') &&
          saveLogLine.includes('previousFrames=310') &&
          saveLogLine.includes('nodes=3') &&
          saveLogLine.includes('connections=3') &&
          saveLogLine.includes('inputs=1') &&
          saveLogLine.includes('hidden=1') &&
          saveLogLine.includes('outputs=1') &&
          saveLogLine.includes('fingerprint=0x'),
        progressUpdate,
      }).toEqual({
        didLogSave: true,
        progressUpdate: {
          championByProfileId: {
            gru: championNetworkJson,
            mlp: { nodes: [{ bias: 1 }], scope: 'previous' },
          },
          didImprove: true,
          historyByProfileId: {
            gru: { pipesPassed: 6, framesSurvived: 280 },
            mlp: { pipesPassed: 1, framesSurvived: 80 },
          },
        },
      });
    } finally {
      consoleInfoSpy.mockRestore();
    }
  });
});

describe('resolveRuntimeChampionCandidateNetworkJson', () => {
  it('prefers the playback winner network over the generation-best fallback', () => {
    const playbackWinnerNetworkJson = { scope: 'playback-winner' };
    const generationBestNetworkJson = { scope: 'generation-best' };

    expect(
      resolveRuntimeChampionCandidateNetworkJson({
        generationBestNetworkJson,
        playbackSummary: {
          averagePipesPassed: 1,
          p90FramesSurvived: 2,
          winnerFramesSurvived: 3,
          winnerPipesPassed: 4,
          winnerNetworkJson: playbackWinnerNetworkJson,
        },
      }),
    ).toBe(playbackWinnerNetworkJson);
  });
});

describe('resolveWorkerInitPayload', () => {
  it('includes the saved champion network for the selected architecture profile when one exists', () => {
    const championNetworkJson = {
      connections: [{ weight: 1 }, { weight: 2 }],
      nodes: [
        { bias: 0, type: 'input' },
        { bias: 42, type: 'hidden' },
        { bias: 1, type: 'output' },
      ],
      scope: 'champion',
    };
    const consoleInfoSpy = jest
      .spyOn(console, 'info')
      .mockImplementation(() => undefined);

    try {
      const workerInitPayload = withNodeEnv('development', () =>
        resolveWorkerInitPayload({
          architectureProfileId: 'gru',
          championByProfileId: {
            gru: championNetworkJson,
            narx: { nodes: [{ bias: 3 }], scope: 'other' },
          },
          elitismCount: 2,
          populationSize: 10,
          rngSeed: 12345,
        }),
      );
      const loadLogLine = consoleInfoSpy.mock.calls.at(-1)?.[0];

      expect({
        didLogLoad:
          typeof loadLogLine === 'string' &&
          loadLogLine.includes('loading saved champion profile=gru') &&
          loadLogLine.includes('nodes=3') &&
          loadLogLine.includes('connections=2') &&
          loadLogLine.includes('inputs=1') &&
          loadLogLine.includes('hidden=1') &&
          loadLogLine.includes('outputs=1') &&
          loadLogLine.includes('fingerprint=0x'),
        workerInitPayload,
      }).toEqual({
        didLogLoad: true,
        workerInitPayload: {
          architectureProfileId: 'gru',
          championNetworkJson,
          elitismCount: 2,
          populationSize: 10,
          rngSeed: 12345,
        },
      });
    } finally {
      consoleInfoSpy.mockRestore();
    }
  });
});
