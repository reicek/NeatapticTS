import type Network from '../../../src/architecture/network';
import {
  evaluateInWorkers,
  exportTransferableInferencePayload,
  openSharedInferenceWorker,
  ParallelInferencePool,
  type SharedInferenceWorker,
} from '../../../src/neataptic';
import type {
  FlappyRolloutOptions,
  FlappySeedBatchEvaluation,
} from './evaluation.types';
import { evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker } from './evaluation.fitness.utils';

const DEFAULT_FLAPPY_EVALUATION_WORKER_COUNT = 4;

type WorkerPoolPayload = ReturnType<typeof exportTransferableInferencePayload>;

type WorkerPoolGenome = Network & {
  _id?: number;
  clear?: () => void;
  activate(inputs: number[]): number[] | number;
  score?: number;
};

/**
 * Optional delivery controls for the Flappy shared-worker evaluation pool.
 *
 * Browser examples usually resolve the worker bundle URL relative to the
 * evolution worker location. Tests can omit this and inject mocks instead.
 */
export interface FlappyEvaluationWorkerPoolOptions {
  workerUrl?: string;
}

/**
 * Bounded shared-worker pool for parallel Flappy evaluation across genomes.
 *
 * The pool parallelizes the expensive cross-genome part of evaluation while
 * keeping each individual genome on one persistent predictor for its seed
 * batch. That preserves recurrent reset semantics and avoids reopening a worker
 * for every single seeded rollout.
 */
export class FlappyEvaluationWorkerPool {
  readonly #workerCount: number;
  readonly #workerUrl?: string;
  readonly #workerPool: ParallelInferencePool<
    WorkerPoolPayload,
    SharedInferenceWorker
  >;
  #payloadByGenome = new Map<WorkerPoolGenome, WorkerPoolPayload>();

  /**
   * Creates one bounded shared-worker scheduler for Flappy evaluation.
   *
   * @param workerCount - Maximum number of simultaneously active shared workers.
   * @param options - Optional shared-worker delivery overrides.
   */
  constructor(
    workerCount = resolveDefaultFlappyEvaluationWorkerCount(),
    options: FlappyEvaluationWorkerPoolOptions = {},
  ) {
    this.#workerCount = Math.max(1, workerCount);
    this.#workerUrl = options.workerUrl;
    this.#workerPool = new ParallelInferencePool({
      openWorker: (payload) =>
        openSharedInferenceWorker(payload, {
          workerUrl: this.#workerUrl,
        }),
      workerCount: this.#workerCount,
    });
  }

  /**
   * Prepares exported payloads and empty slot state for one population shelf.
   *
   * @param genomes - Population that may be evaluated during the next generation.
   * @returns Nothing.
   */
  async initialize(genomes: readonly WorkerPoolGenome[]): Promise<void> {
    // Step 1: Release any previous shared-worker slots before switching populations.
    await this.dispose();

    // Step 2: Export the transferable predictor payload for each genome once.
    this.#payloadByGenome = new Map(
      genomes.map((genome) => [
        genome,
        exportTransferableInferencePayload(genome),
      ]),
    );

    // Step 3: Prime the shared generic scheduler with the exported payload shelf.
    await this.#workerPool.initialize(
      genomes.map((genome) => resolveRequiredGenomePayload(this.#payloadByGenome, genome)),
    );
  }

  /**
   * Evaluate one genome subset across a shared deterministic seed batch.
   *
   * @param genomes - Ordered genome shelf to score.
   * @param sharedSeeds - Shared deterministic seeds used for each genome.
   * @param rolloutOptions - Rollout controls reused across the whole batch.
   * @returns Aggregates keyed by genome in the caller's original order.
   */
  async evaluateGenomesAcrossSeeds(
    genomes: readonly WorkerPoolGenome[],
    sharedSeeds: readonly number[],
    rolloutOptions: FlappyRolloutOptions,
  ): Promise<Map<WorkerPoolGenome, FlappySeedBatchEvaluation>> {
    if (genomes.length === 0) {
      return new Map();
    }

    const orderedPayloads = await this.resolveOrderedPayloads(genomes);
    const batchResult = await evaluateInWorkers({
      inputs: genomes,
      resolvePayload: (_genome, genomeIndex) =>
        resolveRequiredOrderedGenomePayload(orderedPayloads, genomeIndex),
      evaluateWithWorker: async (sharedWorker, genome) =>
        evaluateFlappyFitnessAcrossSeedsWithSharedInferenceWorker(
          sharedWorker,
          sharedSeeds,
          {
            networkId:
              typeof genome._id === 'number'
                ? genome._id
                : undefined,
            rolloutOptions,
          },
        ),
      workerPool: this.#workerPool,
    });

    // Step 2: Rebuild the aggregate map in caller order for deterministic consumers.
    return new Map(
      genomes.map((genome, genomeIndex) => [
        genome,
        resolveRequiredGenomeAggregate(batchResult.results, genomeIndex),
      ]),
    );
  }

  /**
   * Resolves the ordered transferable payload shelf for one genome batch.
   *
   * @param genomes - Genome batch that may be evaluated next.
   * @returns Ordered transferable payload shelf aligned to the input genomes.
   */
  async resolveOrderedPayloads(
    genomes: readonly WorkerPoolGenome[],
  ): Promise<WorkerPoolPayload[]> {
    if (needsPopulationRefresh(this.#payloadByGenome, genomes)) {
      await this.initialize(genomes);
    }

    return genomes.map((genome) =>
      resolveRequiredGenomePayload(this.#payloadByGenome, genome),
    );
  }

  /**
   * Exposes the shared generic scheduler for public ordered batch helpers.
   */
  get parallelWorkerPool(): ParallelInferencePool<
    WorkerPoolPayload,
    SharedInferenceWorker
  > {
    return this.#workerPool;
  }

  /**
   * Releases every active shared worker and clears cached population state.
   *
   * @returns Nothing.
   */
  async dispose(): Promise<void> {
    this.#payloadByGenome.clear();
    await this.#workerPool.dispose();
  }
}

function resolveRequiredGenomeAggregate(
  orderedAggregates: readonly FlappySeedBatchEvaluation[],
  genomeIndex: number,
): FlappySeedBatchEvaluation {
  const aggregate = orderedAggregates[genomeIndex];

  if (!aggregate) {
    throw new Error('FlappyEvaluationWorkerPool did not resolve every queued genome.');
  }

  return aggregate;
}

function resolveRequiredGenomePayload(
  payloadByGenome: ReadonlyMap<WorkerPoolGenome, WorkerPoolPayload>,
  genome: WorkerPoolGenome,
): WorkerPoolPayload {
  const payload = payloadByGenome.get(genome);

  if (!payload) {
    throw new Error('FlappyEvaluationWorkerPool genome payload was not initialized.');
  }

  return payload;
}

function resolveRequiredOrderedGenomePayload(
  orderedPayloads: readonly WorkerPoolPayload[],
  genomeIndex: number,
): WorkerPoolPayload {
  const payload = orderedPayloads[genomeIndex];

  if (!payload) {
    throw new Error('FlappyEvaluationWorkerPool did not resolve every queued genome payload.');
  }

  return payload;
}

function needsPopulationRefresh(
  payloadByGenome: ReadonlyMap<WorkerPoolGenome, WorkerPoolPayload>,
  genomes: readonly WorkerPoolGenome[],
): boolean {
  if (payloadByGenome.size !== genomes.length) {
    return true;
  }

  return genomes.some((genome) => !payloadByGenome.has(genome));
}

function resolveDefaultFlappyEvaluationWorkerCount(): number {
  const detectedHardwareConcurrency = globalThis.navigator?.hardwareConcurrency;

  if (!Number.isFinite(detectedHardwareConcurrency)) {
    return DEFAULT_FLAPPY_EVALUATION_WORKER_COUNT;
  }

  return Math.max(
    1,
    Math.min(
      detectedHardwareConcurrency,
      DEFAULT_FLAPPY_EVALUATION_WORKER_COUNT,
    ),
  );
}