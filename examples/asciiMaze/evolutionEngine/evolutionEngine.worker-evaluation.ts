import {
  createNeatParallelPopulationEvaluator,
  exportTransferableInferencePayload,
  getTransferList,
  type ParallelInferenceWorkerLike,
  type TransferableInferencePayload,
} from '../../../src/architecture/network/worker-payload/network.worker-payload';
import { FitnessEvaluator } from '../fitness';
import type {
  FitnessEvaluatorFn,
  IFitnessEvaluationContext,
} from '../fitness.types';
import type { INetwork } from '../interfaces';
import type Network from '../../../src/architecture/network';
import type { EvolutionWorkerEvaluationConfig } from './evolutionEngine.types';

type AsciiMazeWorkerPopulationGenome = Network & {
  score?: number;
  clear?: () => void;
  activate(inputs: number[]): number[] | number;
};

/** Browser-worker bootstrap payload for one ASCII Maze genome evaluator. */
export interface AsciiMazeEvaluationWorkerPayload {
  readonly fitnessContext: IFitnessEvaluationContext;
  readonly inferencePayload: TransferableInferencePayload;
}

/** Worker bootstrap request sent from the browser host to one evaluator worker. */
export type AsciiMazeEvaluationWorkerRequest =
  | {
      readonly payload: AsciiMazeEvaluationWorkerPayload;
      readonly type: 'bootstrap';
    }
  | {
      readonly requestId: number;
      readonly type: 'evaluate';
    };

/** Worker response emitted by one ASCII Maze evaluator worker. */
export type AsciiMazeEvaluationWorkerResponse =
  | { readonly type: 'ready' }
  | {
      readonly fitness: number;
      readonly requestId: number;
      readonly type: 'evaluate-result';
    }
  | {
      readonly message: string;
      readonly requestId?: number;
      readonly type: 'request-error';
    };

interface AsciiMazeEvaluationWorker extends ParallelInferenceWorkerLike {
  evaluateFitness(): Promise<number>;
}

type AsciiMazeLocalGenomeEvaluator = (
  genome: AsciiMazeWorkerPopulationGenome,
  genomeIndex: number,
) => Promise<number> | number;

/**
 * Create the browser-worker population evaluator for ASCII Maze genome scoring.
 *
 * This helper keeps worker transport local to the example runtime. When the
 * default maze evaluator is active and browser workers are available, the NEAT
 * controller can score one whole population through a worker batch while still
 * preserving a local single-thread fallback for unsupported environments.
 *
 * @param fitnessContext - Read-only maze evaluation context shared by the run.
 * @param workerEvaluation - Browser worker controls for this run.
 * @param fitnessEvaluator - Active fitness delegate chosen by the caller.
 * @returns Population-wide fitness delegate, or `undefined` when worker mode is unavailable.
 */
export function createAsciiMazeWorkerPopulationFitnessEvaluator(
  fitnessContext: IFitnessEvaluationContext,
  workerEvaluation: EvolutionWorkerEvaluationConfig | undefined,
  fitnessEvaluator: FitnessEvaluatorFn,
):
  | ((population: AsciiMazeWorkerPopulationGenome[]) => Promise<void>)
  | undefined {
  if (
    fitnessEvaluator !== FitnessEvaluator.defaultFitnessEvaluator ||
    !canUseAsciiMazeWorkerEvaluation(workerEvaluation)
  ) {
    return undefined;
  }

  const evaluateGenomeLocally = createAsciiMazeLocalGenomeEvaluator(
    fitnessContext,
    fitnessEvaluator,
  );
  const evaluatePopulationInWorkers = createNeatParallelPopulationEvaluator<
    AsciiMazeWorkerPopulationGenome,
    AsciiMazeEvaluationWorkerPayload,
    AsciiMazeEvaluationWorker,
    number
  >({
    parallel: true,
    evaluateGenome: evaluateGenomeLocally,
    evaluateWithWorker: (worker) => worker.evaluateFitness(),
    openWorker: (payload) =>
      openAsciiMazeEvaluationWorker(payload, workerEvaluation.workerUrl),
    resolvePayload: (genome) => ({
      fitnessContext,
      inferencePayload: exportTransferableInferencePayload(genome),
    }),
    workerCount: workerEvaluation.workerCount,
  });

  return async (population): Promise<void> => {
    try {
      await evaluatePopulationInWorkers(population);
      return;
    } catch (error) {
      reportAsciiMazeWorkerEvaluationFallback(error);
    }

    await evaluateAsciiMazePopulationLocally(population, evaluateGenomeLocally);
  };
}

/**
 * Create the local ASCII Maze genome evaluator used by both fallback paths.
 *
 * @param fitnessContext - Read-only maze context shared by the active run.
 * @param fitnessEvaluator - Fitness delegate selected by the evolution engine.
 * @returns Local evaluator that clears recurrent state before scoring one genome.
 */
function createAsciiMazeLocalGenomeEvaluator(
  fitnessContext: IFitnessEvaluationContext,
  fitnessEvaluator: FitnessEvaluatorFn,
): AsciiMazeLocalGenomeEvaluator {
  return (genome) => {
    genome.clear?.();
    return fitnessEvaluator(genome as unknown as INetwork, fitnessContext);
  };
}

/**
 * Score an ASCII Maze population locally after worker evaluation becomes unusable.
 *
 * @param population - Ordered genome shelf to score in place.
 * @param evaluateGenome - Local scorer shared with the worker helper fallback.
 * @returns Promise resolved after every genome has a local score.
 */
async function evaluateAsciiMazePopulationLocally(
  population: AsciiMazeWorkerPopulationGenome[],
  evaluateGenome: AsciiMazeLocalGenomeEvaluator,
): Promise<void> {
  const localScores = await Promise.all(
    population.map((genome, genomeIndex) =>
      evaluateGenome(genome, genomeIndex),
    ),
  );

  localScores.forEach((localScore, genomeIndex) => {
    const genome = population[genomeIndex];
    if (genome) {
      genome.score = localScore;
    }
  });
}

/**
 * Report worker evaluation fallback without making console availability fatal.
 *
 * @param error - Worker startup or evaluation error that triggered local scoring.
 * @returns Nothing.
 */
function reportAsciiMazeWorkerEvaluationFallback(error: unknown): void {
  try {
    console.warn(
      '[asciiMaze] worker evaluation failed; falling back to local scoring.',
      error,
    );
  } catch {
    // Ignore console failures in restricted hosts.
  }
}

function canUseAsciiMazeWorkerEvaluation(
  workerEvaluation: EvolutionWorkerEvaluationConfig | undefined,
): workerEvaluation is EvolutionWorkerEvaluationConfig & {
  workerUrl: string;
} {
  return Boolean(
    workerEvaluation?.enabled &&
    workerEvaluation.workerUrl &&
    typeof globalThis.Worker === 'function',
  );
}

function openAsciiMazeEvaluationWorker(
  payload: AsciiMazeEvaluationWorkerPayload,
  workerUrl: string,
): Promise<AsciiMazeEvaluationWorker> {
  return new Promise<AsciiMazeEvaluationWorker>((resolve, reject) => {
    const worker = new Worker(workerUrl);
    const pendingRequests = new Map<
      number,
      {
        reject: (error: Error) => void;
        resolve: (fitness: number) => void;
      }
    >();
    let nextRequestId = 0;
    let ready = false;
    let released = false;

    const releaseWorker = (): void => {
      if (released) {
        return;
      }

      released = true;
      worker.removeEventListener('error', handleWorkerError);
      worker.removeEventListener('message', handleWorkerMessage);
      worker.removeEventListener('messageerror', handleWorkerMessageError);
      worker.terminate();
    };

    const rejectPendingRequests = (error: Error): void => {
      const pendingRequestEntries = Array.from(pendingRequests.values());
      pendingRequests.clear();
      pendingRequestEntries.forEach((pendingRequest) =>
        pendingRequest.reject(error),
      );
    };

    const failWorker = (error: Error): void => {
      rejectPendingRequests(error);
      releaseWorker();

      if (!ready) {
        reject(error);
      }
    };

    const handleWorkerError = (): void => {
      failWorker(new Error('ASCII Maze evaluation worker crashed.'));
    };

    const handleWorkerMessageError = (): void => {
      failWorker(
        new Error('ASCII Maze evaluation worker rejected one message payload.'),
      );
    };

    const handleWorkerMessage = (
      event: MessageEvent<AsciiMazeEvaluationWorkerResponse>,
    ): void => {
      const response = event.data;

      if (response?.type === 'ready') {
        ready = true;
        resolve({
          async evaluateFitness(): Promise<number> {
            if (released) {
              throw new Error(
                'ASCII Maze evaluation worker was already released.',
              );
            }

            const requestId = nextRequestId;
            nextRequestId += 1;

            return new Promise<number>((resolveFitness, rejectFitness) => {
              pendingRequests.set(requestId, {
                reject: rejectFitness,
                resolve: resolveFitness,
              });
              worker.postMessage({
                requestId,
                type: 'evaluate',
              } satisfies AsciiMazeEvaluationWorkerRequest);
            });
          },

          async release(): Promise<void> {
            rejectPendingRequests(
              new Error('ASCII Maze evaluation worker was released.'),
            );
            releaseWorker();
          },
        });
        return;
      }

      if (response?.type === 'evaluate-result') {
        const pendingRequest = pendingRequests.get(response.requestId);

        if (!pendingRequest) {
          return;
        }

        pendingRequests.delete(response.requestId);
        pendingRequest.resolve(response.fitness);
        return;
      }

      if (response?.type === 'request-error') {
        const requestError = new Error(response.message);

        if (typeof response.requestId === 'number') {
          const pendingRequest = pendingRequests.get(response.requestId);
          if (!pendingRequest) {
            return;
          }

          pendingRequests.delete(response.requestId);
          pendingRequest.reject(requestError);
          return;
        }

        failWorker(requestError);
      }
    };

    worker.addEventListener('error', handleWorkerError);
    worker.addEventListener('message', handleWorkerMessage);
    worker.addEventListener('messageerror', handleWorkerMessageError);
    worker.postMessage(
      {
        payload,
        type: 'bootstrap',
      } satisfies AsciiMazeEvaluationWorkerRequest,
      getTransferList(payload.inferencePayload),
    );
  });
}
