/** @jest-environment jsdom */

jest.mock('../host/host', () => ({
  createCanvasHost: jest.fn(() => ({
    statsValueByKey: {},
  })),
  updateStatsTableValues: jest.fn(),
}));

jest.mock('../worker-channel/worker-channel', () => ({
  createEvolutionWorker: jest.fn(() => ({
    postMessage: jest.fn(),
  })),
}));

jest.mock('./runtime.errors', () => ({
  resolveRequiredRuntimeHostElement: jest.fn(() => document.createElement('div')),
}));

jest.mock('./runtime.telemetry.service', () => ({
  createRuntimeTelemetryState: jest.fn(() => ({
    updatesPerSecondWindowStartMs: 0,
  })),
}));

import {
  FLAPPY_BROWSER_ELITISM_COUNT,
  FLAPPY_BROWSER_POPULATION_SIZE,
} from '../../constants/constants';
import { createRuntimeStartContext } from './runtime.startup.service';

describe('createRuntimeStartContext', () => {
  it.each([
    {
      architectureProfileId: 'random-sparse' as const,
      expectedElitismCount: 6,
      expectedPopulationSize: 30,
    },
    {
      architectureProfileId: 'narx' as const,
      expectedElitismCount: 8,
      expectedPopulationSize: 40,
    },
    {
      architectureProfileId: 'gru' as const,
      expectedElitismCount: 4,
      expectedPopulationSize: 18,
    },
    {
      architectureProfileId: 'lstm' as const,
      expectedElitismCount: 1,
      expectedPopulationSize: 6,
    },
    {
      architectureProfileId: 'mlp' as const,
      expectedElitismCount: FLAPPY_BROWSER_ELITISM_COUNT,
      expectedPopulationSize: FLAPPY_BROWSER_POPULATION_SIZE,
    },
  ])(
    'resolves the expected browser budget for $architectureProfileId',
    ({
      architectureProfileId,
      expectedElitismCount,
      expectedPopulationSize,
    }) => {
      const runtimeStartContext = createRuntimeStartContext(
        document.createElement('div'),
        { architectureProfileId },
      );

      expect({
        elitismCount: runtimeStartContext.config.elitismCount,
        populationSize: runtimeStartContext.config.populationSize,
        selectedArchitectureProfileId:
          runtimeStartContext.config.selectedArchitectureProfile.id,
      }).toEqual({
        elitismCount: expectedElitismCount,
        populationSize: expectedPopulationSize,
        selectedArchitectureProfileId: architectureProfileId,
      });
    },
  );
});