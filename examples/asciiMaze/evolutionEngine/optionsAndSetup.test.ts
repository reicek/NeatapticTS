import type { IRunMazeEvolutionOptions } from './evolutionEngine.types';
import { normalizeRunOptions } from './optionsAndSetup';

function createRunOptions(
  evolutionAlgorithmConfig: IRunMazeEvolutionOptions['evolutionAlgorithmConfig'],
): IRunMazeEvolutionOptions {
  return {
    mazeConfig: { maze: ['SE'] },
    agentSimConfig: { maxSteps: 8 },
    evolutionAlgorithmConfig,
    reportingConfig: {
      dashboardManager: {} as never,
    },
  };
}

describe('normalizeRunOptions', () => {
  it('adds a builder-backed ASCII Maze seed network when a shared profile id is requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({ architectureProfileId: 'mlp' }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      hiddenLayerSizes:
        normalizedOptions.neatOptions.network?.describeArchitecture()
          .hiddenLayerSizes,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds:
        normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      hiddenLayerSizes: [6, 6],
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });

  it('defaults to the configured builder profile when no architecture profile id is provided on a fresh start', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({}),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      architectureProfileId: normalizedOptions.architectureProfileId,
      connectionCount:
        normalizedOptions.neatOptions.network?.connections.length,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds:
        normalizedOptions.neatOptions.network?.outputNodeIds.length,
      popSize: normalizedOptions.popSize,
    }).toEqual({
      architectureProfileId: 'random-sparse',
      connectionCount: 40,
      inputNodeIds: 6,
      outputNodeIds: 4,
      popSize: 100,
    });
  });

  it('defaults the ASCII Maze runtime to feed-forward-only growth unless recurrence is explicitly requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({}),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      allowRecurrent: normalizedOptions.allowRecurrent,
      neatAllowRecurrent: normalizedOptions.neatOptions.allowRecurrent,
    }).toEqual({
      allowRecurrent: false,
      neatAllowRecurrent: false,
    });
  });

  it('leaves the seed network unset when an explicit initial population is provided', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({ initialPopulation: [] }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect(normalizedOptions.neatOptions.network).toBeUndefined();
  });

  it('adds a builder-backed NARX seed network when the narx profile id is requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({ architectureProfileId: 'narx' }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      architectureProfileId: normalizedOptions.architectureProfileId,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds:
        normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      architectureProfileId: 'narx',
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });

  it('adds a builder-backed GRU seed network when the gru profile id is requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({ architectureProfileId: 'gru' }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      architectureProfileId: normalizedOptions.architectureProfileId,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds:
        normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      architectureProfileId: 'gru',
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });

  it('adds a builder-backed LSTM seed network when the lstm profile id is requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({ architectureProfileId: 'lstm' }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      architectureProfileId: normalizedOptions.architectureProfileId,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds:
        normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      architectureProfileId: 'lstm',
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });

  it('forwards explicit adaptive-mutation overrides into the NEAT runtime options', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({
        adaptiveMutation: {
          enabled: true,
          strategy: 'twoTier',
          adaptEvery: 5,
          sigma: 0.1,
          minRate: 0.001,
        },
      }),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect(normalizedOptions.neatOptions.adaptiveMutation).toEqual({
      enabled: true,
      strategy: 'twoTier',
      adaptEvery: 5,
      sigma: 0.1,
      minRate: 0.001,
    });
  });
});
