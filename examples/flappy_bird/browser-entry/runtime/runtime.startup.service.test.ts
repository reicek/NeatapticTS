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
  resolveRequiredRuntimeHostElement: jest.fn(() =>
    document.createElement('div'),
  ),
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
  beforeEach(() => {
    Object.defineProperty(globalThis.navigator, 'hardwareConcurrency', {
      configurable: true,
      value: 4,
    });
  });

  it.each([
    {
      architectureProfileId: 'random-sparse' as const,
      expectedElitismCount: 4,
      expectedPopulationSize: 10,
    },
    {
      architectureProfileId: 'narx' as const,
      expectedElitismCount: 2,
      expectedPopulationSize: 10,
    },
    {
      architectureProfileId: 'gru' as const,
      expectedElitismCount: 2,
      expectedPopulationSize: 10,
    },
    {
      architectureProfileId: 'lstm' as const,
      expectedElitismCount: 2,
      expectedPopulationSize: 10,
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
  afterEach(() => {
    Reflect.deleteProperty(globalThis.navigator, 'hardwareConcurrency');
  });

  it('keeps the tuned recurrent browser budgets fixed on higher-core machines', () => {
    Object.defineProperty(globalThis.navigator, 'hardwareConcurrency', {
      configurable: true,
      value: 8,
    });

    const runtimeStartContext = createRuntimeStartContext(
      document.createElement('div'),
      { architectureProfileId: 'narx' },
    );

    expect({
      elitismCount: runtimeStartContext.config.elitismCount,
      populationSize: runtimeStartContext.config.populationSize,
    }).toEqual({
      elitismCount: 2,
      populationSize: 10,
    });
  });
});
