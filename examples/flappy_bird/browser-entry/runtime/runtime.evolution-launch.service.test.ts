/** @jest-environment jsdom */

jest.mock('./runtime.evolution-loop.service', () => ({
  runRuntimeEvolutionLoop: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../host/host', () => ({
  updateStatsTableValues: jest.fn(),
}));

jest.mock('./runtime.errors', () => ({
  resolveRuntimeHudErrorStatus: jest.fn(() => 'error'),
}));

import { launchRuntimeEvolution } from './runtime.evolution-launch.service';
import { runRuntimeEvolutionLoop } from './runtime.evolution-loop.service';

describe('launchRuntimeEvolution', () => {
  it('forwards viewContext.applyNetworkActivationOverlay to the evolution loop', () => {
    const applyNetworkActivationOverlay = jest.fn();
    const renderNetworkArchitecture = jest.fn();
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    const statsValueByKey = {} as Record<string, string>;
    const viewContext = {
      canvas,
      context,
      statsValueByKey,
      renderNetworkArchitecture,
      applyNetworkActivationOverlay,
      architectureSelectorController: {
        element: document.createElement('div'),
        setDisabled: jest.fn(),
        updateItems: jest.fn(),
      },
    };
    const runtimeStartContext = {
      config: {
        availableArchitectureProfiles: [],
        architectureChampionByProfileId: {},
        architectureHistoryByProfileId: {},
        populationSize: 10,
        elitismCount: 2,
        inputSize: 5,
        outputSize: 2,
        selectedArchitectureProfile: { id: 'mlp', label: 'MLP' },
      },
      evolutionWorker: { postMessage: jest.fn() } as unknown as Worker,
      runtimeTelemetryState: { updatesPerSecondWindowStartMs: 0 },
      hostElement: document.createElement('div'),
      viewContext,
    };
    const runtimeLifecycleState = {
      stopped: false,
      done: Promise.resolve(),
    };
    const stop = jest.fn();

    launchRuntimeEvolution(
      runtimeStartContext as unknown as Parameters<
        typeof launchRuntimeEvolution
      >[0],
      runtimeLifecycleState as unknown as Parameters<
        typeof launchRuntimeEvolution
      >[1],
      stop,
    );

    expect(runRuntimeEvolutionLoop).toHaveBeenCalledTimes(1);
    const [options] = (runRuntimeEvolutionLoop as jest.Mock).mock.calls[0];
    expect(options.applyNetworkActivationOverlay).toBe(
      applyNetworkActivationOverlay,
    );
  });
});
