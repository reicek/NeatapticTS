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
      hiddenLayerSizes: [6],
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });

  it('defaults to the MLP builder profile when no architecture profile id is provided on a fresh start', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({}),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect({
      architectureProfileId: normalizedOptions.architectureProfileId,
      inputNodeIds: normalizedOptions.neatOptions.network?.inputNodeIds.length,
      outputNodeIds: normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      architectureProfileId: 'mlp',
      inputNodeIds: 6,
      outputNodeIds: 4,
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
      outputNodeIds: normalizedOptions.neatOptions.network?.outputNodeIds.length,
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
      outputNodeIds: normalizedOptions.neatOptions.network?.outputNodeIds.length,
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
      outputNodeIds: normalizedOptions.neatOptions.network?.outputNodeIds.length,
    }).toEqual({
      architectureProfileId: 'lstm',
      inputNodeIds: 6,
      outputNodeIds: 4,
    });
  });
});
