import Network from '../../../src/architecture/network/network';
import { FitnessEvaluator } from '../fitness';
import type { IFitnessEvaluationContext } from '../fitness.types';
import { createAsciiMazeWorkerPopulationFitnessEvaluator } from './evolutionEngine.worker-evaluation';

type WorkerEventListener = (event: Event) => void;

class StartupFailingWorker {
  readonly #listeners = new Map<string, Set<WorkerEventListener>>();

  addEventListener(type: string, listener: WorkerEventListener): void {
    const listenersForType = this.#listeners.get(type) ?? new Set();
    listenersForType.add(listener);
    this.#listeners.set(type, listenersForType);
  }

  removeEventListener(type: string, listener: WorkerEventListener): void {
    this.#listeners.get(type)?.delete(listener);
  }

  postMessage(): void {
    queueMicrotask(() => this.#dispatch('error'));
  }

  terminate(): void {
    this.#listeners.clear();
  }

  #dispatch(type: string): void {
    const event = new Event(type);
    this.#listeners.get(type)?.forEach((listener) => listener(event));
  }
}

describe('createAsciiMazeWorkerPopulationFitnessEvaluator', () => {
  const originalWorker = globalThis.Worker;

  afterEach(() => {
    jest.restoreAllMocks();
    restoreWorkerConstructor(originalWorker);
  });

  it('falls back to local scoring when the browser worker fails during startup', async () => {
    const fitnessByNetwork = new WeakMap<object, number>();
    const population = [
      createNetworkWithFitness(7),
      createNetworkWithFitness(11),
    ];
    const fitnessContext = createFitnessContext();
    const fitnessSpy = jest
      .spyOn(FitnessEvaluator, 'defaultFitnessEvaluator')
      .mockImplementation((network) => fitnessByNetwork.get(network) ?? 0);
    population.forEach((network) => {
      fitnessByNetwork.set(network, network.localFitness);
    });
    installWorkerConstructor(StartupFailingWorker);

    const evaluatePopulation = createAsciiMazeWorkerPopulationFitnessEvaluator(
      fitnessContext,
      {
        enabled: true,
        workerCount: 1,
        workerUrl: 'https://example.com/ascii-maze-evaluation.worker.bundle.js',
      },
      FitnessEvaluator.defaultFitnessEvaluator,
    );

    if (!evaluatePopulation) {
      throw new Error('Expected worker population evaluator to be created');
    }

    await evaluatePopulation(population);

    expect({
      localCallCount: fitnessSpy.mock.calls.length,
      scores: population.map((network) => network.score),
    }).toEqual({
      localCallCount: 2,
      scores: [7, 11],
    });
  });
});

function createNetworkWithFitness(
  localFitness: number,
): Network & { localFitness: number; score?: number } {
  const network = new Network(1, 1) as Network & {
    localFitness: number;
    score?: number;
  };
  network.localFitness = localFitness;
  return network;
}

function createFitnessContext(): IFitnessEvaluationContext {
  return {
    agentSimConfig: { maxSteps: 1 },
    distanceMap: [[0]],
    encodedMaze: [[0]],
    exitPosition: [0, 0],
    startPosition: [0, 0],
  };
}

function installWorkerConstructor(
  workerConstructor: typeof StartupFailingWorker,
): void {
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    value: workerConstructor,
  });
}

function restoreWorkerConstructor(
  originalWorker: typeof globalThis.Worker,
): void {
  if (originalWorker) {
    Object.defineProperty(globalThis, 'Worker', {
      configurable: true,
      value: originalWorker,
    });
    return;
  }

  delete (globalThis as { Worker?: typeof globalThis.Worker }).Worker;
}
