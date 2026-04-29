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

  it('keeps the ASCII Maze seed-network option unset when no shared profile id is requested', () => {
    const normalizedOptions = normalizeRunOptions(
      createRunOptions({}),
      jest.fn(),
      jest.fn(),
      jest.fn(),
      jest.fn(),
    );

    expect(normalizedOptions.neatOptions.network).toBeUndefined();
  });
});
