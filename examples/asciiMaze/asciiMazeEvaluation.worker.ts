import {
  createInferencePredictor,
  type InferencePredictor,
} from '../../src/neataptic';
import { FitnessEvaluator } from './fitness';
import type { INetwork } from './interfaces';
import type {
  AsciiMazeEvaluationWorkerPayload,
  AsciiMazeEvaluationWorkerRequest,
  AsciiMazeEvaluationWorkerResponse,
} from './evolutionEngine/evolutionEngine.worker-evaluation';

let workerPayload: AsciiMazeEvaluationWorkerPayload | null = null;
let workerPredictor: InferencePredictor | null = null;

/**
 * Register the dedicated ASCII Maze browser worker runtime.
 *
 * The worker keeps one warm predictor rebuilt from a transferable inference
 * payload and reuses the shared maze-fitness context for every evaluation
 * request posted by the browser entry population evaluator.
 *
 * @returns Nothing.
 */
function registerAsciiMazeEvaluationWorkerRuntime(): void {
  globalThis.addEventListener('message', handleWorkerMessage);
}

function handleWorkerMessage(
  event: MessageEvent<AsciiMazeEvaluationWorkerRequest>,
): void {
  const request = event.data;

  try {
    if (request.type === 'bootstrap') {
      workerPayload = request.payload;
      workerPredictor = createInferencePredictor(
        request.payload.inferencePayload,
      );
      postMessage({
        type: 'ready',
      } satisfies AsciiMazeEvaluationWorkerResponse);
      return;
    }

    if (request.type !== 'evaluate') {
      return;
    }

    if (!workerPayload || !workerPredictor) {
      throw new Error(
        'ASCII Maze evaluation worker received an evaluation request before bootstrap.',
      );
    }

    workerPredictor.reset();
    const workerNetwork = createWorkerEvaluationNetwork(workerPredictor);
    const fitness = FitnessEvaluator.defaultFitnessEvaluator(
      workerNetwork,
      workerPayload.fitnessContext,
    );

    postMessage({
      fitness,
      requestId: request.requestId,
      type: 'evaluate-result',
    } satisfies AsciiMazeEvaluationWorkerResponse);
  } catch (error) {
    postMessage({
      message: error instanceof Error ? error.message : String(error),
      requestId: request.type === 'evaluate' ? request.requestId : undefined,
      type: 'request-error',
    } satisfies AsciiMazeEvaluationWorkerResponse);
  }
}

function createWorkerEvaluationNetwork(
  predictor: InferencePredictor,
): INetwork {
  return {
    activate: (inputs: number[]) => predictor.predict(inputs),
    clear: () => predictor.reset(),
  };
}

registerAsciiMazeEvaluationWorkerRuntime();
