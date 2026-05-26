import Network from '../../architecture/network';
import {
  fromParameterVector,
  toParameterVector,
} from '../../architecture/network/serialize/network.serialize.utils';
import type {
  ParameterLayoutEntry,
  ParameterVector,
} from '../../architecture/network/serialize/network.serialize.utils.types';
import { fineTuneVector } from '../../architecture/network/training/network.training.isolate.utils';
import type { TrainingSample } from '../../architecture/network/training/network.training.utils.types';
import { evaluateCandidate } from './neat.hybrid';

type ParameterVectorSnapshot = {
  entries: string[];
  values: number[];
  version: number;
};

const BASE_FITNESS_SCORE = 11;
const TRAINED_FITNESS_SCORE = 29;

function createHybridBaseNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function createHybridDataset(): TrainingSample[] {
  return [
    { input: [0], output: [0] },
    { input: [1], output: [1] },
  ];
}

function createFineTuneOptions(seed: number) {
  return {
    learningRate: 0.25,
    seed,
    steps: 3,
  };
}

function summarizeParameterLayoutEntry(
  parameterEntry: ParameterLayoutEntry,
): string {
  if (parameterEntry.kind === 'bias') {
    return `bias:${parameterEntry.nodeId}`;
  }

  const innovationSummary =
    parameterEntry.innovation == null
      ? 'none'
      : String(parameterEntry.innovation);
  return `weight:${parameterEntry.from}->${parameterEntry.to}:innovation:${innovationSummary}`;
}

function summarizeParameterVector(
  parameterVector: ParameterVector,
): ParameterVectorSnapshot {
  return {
    entries: parameterVector.layout.entries.map(summarizeParameterLayoutEntry),
    values: Array.from(parameterVector.values),
    version: parameterVector.layout.version,
  };
}

function createNetworkSignature(network: Network): string {
  return JSON.stringify(summarizeParameterVector(toParameterVector(network)));
}

function createHybridScoringFixture(input: {
  baseNetwork: Network;
  dataset: TrainingSample[];
  trainingSeed: number;
}): {
  baselineSignature: string;
  fineTuneOptions: ReturnType<typeof createFineTuneOptions>;
  trainedSignature: string;
} {
  const baselineSignature = createNetworkSignature(input.baseNetwork);
  const fineTuneOptions = createFineTuneOptions(input.trainingSeed);
  const trainedVector = fineTuneVector(
    input.baseNetwork,
    toParameterVector(input.baseNetwork),
    input.dataset,
    fineTuneOptions,
  ).trainedVector;
  const trainedNetwork = input.baseNetwork.clone();
  fromParameterVector(trainedNetwork, trainedVector);
  const trainedSignature = createNetworkSignature(trainedNetwork);

  if (baselineSignature === trainedSignature) {
    throw new Error(
      'Expected the hybrid-policy fixture to produce a trained network that differs from the baseline candidate.',
    );
  }

  return {
    baselineSignature,
    fineTuneOptions,
    trainedSignature,
  };
}

function createScoreNetwork(input: {
  baselineSignature: string;
  trainedSignature: string;
}) {
  return async (candidate: Network): Promise<number> => {
    const candidateSignature = createNetworkSignature(candidate);

    if (candidateSignature === input.baselineSignature) {
      return BASE_FITNESS_SCORE;
    }

    if (candidateSignature === input.trainedSignature) {
      return TRAINED_FITNESS_SCORE;
    }

    return Number.NEGATIVE_INFINITY;
  };
}

describe('neat hybrid chapter', () => {
  describe('evaluateCandidate', () => {
    describe('given fine-tuning is disabled', () => {
      describe('when the never policy evaluates a candidate', () => {
        it('returns the base score and leaves the candidate untouched', async () => {
          // Arrange
          const dataset = createHybridDataset();
          const candidate = createHybridBaseNetwork(501);
          const scoringFixture = createHybridScoringFixture({
            baseNetwork: candidate,
            dataset,
            trainingSeed: 801,
          });

          // Act
          const evaluationResult = await evaluateCandidate(candidate, dataset, {
            policy: {
              fineTune: 'never',
              persistTrainedWeights: false,
            },
            fineTuneOptions: scoringFixture.fineTuneOptions,
            scoreNetwork: createScoreNetwork(scoringFixture),
          });
          const postRunCandidateSignature = createNetworkSignature(candidate);

          // Assert
          expect({
            candidate: postRunCandidateSignature,
            fitness: evaluationResult.fitness,
            trainedNetwork:
              evaluationResult.trainedNetwork == null
                ? null
                : createNetworkSignature(evaluationResult.trainedNetwork),
          }).toEqual({
            candidate: scoringFixture.baselineSignature,
            fitness: BASE_FITNESS_SCORE,
            trainedNetwork: null,
          });
        });
      });
    });

    describe('given conditional fine-tuning remains blocked', () => {
      describe('when the conditional policy is requested', () => {
        it('rejects with the deterministic-ranking blocker', async () => {
          // Arrange
          const dataset = createHybridDataset();
          const candidate = createHybridBaseNetwork(504);

          // Act
          const evaluationPromise = evaluateCandidate(candidate, dataset, {
            policy: {
              fineTune: 'conditional',
              persistTrainedWeights: false,
            },
            fineTuneOptions: createFineTuneOptions(804),
            scoreNetwork: async () => BASE_FITNESS_SCORE,
          });

          // Assert
          await expect(evaluationPromise).rejects.toThrow(
            'HybridEvaluationPolicy fineTune="conditional" is blocked until a deterministic ranking surface exists.',
          );
        });
      });
    });

    describe('given fine-tuning requires explicit options', () => {
      describe('when the always policy omits training settings', () => {
        it('rejects before training runs', async () => {
          // Arrange
          const dataset = createHybridDataset();
          const candidate = createHybridBaseNetwork(505);

          // Act
          const evaluationPromise = evaluateCandidate(candidate, dataset, {
            policy: {
              fineTune: 'always',
              persistTrainedWeights: false,
            },
            scoreNetwork: async () => BASE_FITNESS_SCORE,
          });

          // Assert
          await expect(evaluationPromise).rejects.toThrow(
            'evaluateCandidate requires fineTuneOptions when HybridEvaluationPolicy.fineTune is not "never".',
          );
        });
      });
    });

    describe('given fine-tuning always runs without Lamarckian persistence', () => {
      describe('when the trained variant is scored for fitness only', () => {
        it('returns the trained score and keeps the original candidate untouched', async () => {
          // Arrange
          const dataset = createHybridDataset();
          const candidate = createHybridBaseNetwork(502);
          const scoringFixture = createHybridScoringFixture({
            baseNetwork: candidate,
            dataset,
            trainingSeed: 802,
          });

          // Act
          const evaluationResult = await evaluateCandidate(candidate, dataset, {
            policy: {
              fineTune: 'always',
              persistTrainedWeights: false,
            },
            fineTuneOptions: scoringFixture.fineTuneOptions,
            scoreNetwork: createScoreNetwork(scoringFixture),
          });
          const postRunCandidateSignature = createNetworkSignature(candidate);

          // Assert
          expect({
            candidate: postRunCandidateSignature,
            fitness: evaluationResult.fitness,
            trainedNetwork:
              evaluationResult.trainedNetwork == null
                ? null
                : createNetworkSignature(evaluationResult.trainedNetwork),
          }).toEqual({
            candidate: scoringFixture.baselineSignature,
            fitness: TRAINED_FITNESS_SCORE,
            trainedNetwork: scoringFixture.trainedSignature,
          });
        });
      });
    });

    describe('given fine-tuning always runs with Lamarckian persistence enabled', () => {
      describe('when trained weights are explicitly allowed to persist', () => {
        it('applies the trained weights back to the original candidate', async () => {
          // Arrange
          const dataset = createHybridDataset();
          const candidate = createHybridBaseNetwork(503);
          const scoringFixture = createHybridScoringFixture({
            baseNetwork: candidate,
            dataset,
            trainingSeed: 803,
          });

          // Act
          const evaluationResult = await evaluateCandidate(candidate, dataset, {
            policy: {
              fineTune: 'always',
              persistTrainedWeights: true,
            },
            fineTuneOptions: scoringFixture.fineTuneOptions,
            scoreNetwork: createScoreNetwork(scoringFixture),
          });
          const postRunCandidateSignature = createNetworkSignature(candidate);

          // Assert
          expect({
            candidate: postRunCandidateSignature,
            fitness: evaluationResult.fitness,
          }).toEqual({
            candidate: scoringFixture.trainedSignature,
            fitness: TRAINED_FITNESS_SCORE,
          });
        });
      });
    });
  });
});
