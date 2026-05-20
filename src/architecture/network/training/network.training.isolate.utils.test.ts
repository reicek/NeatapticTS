import Network from '../network';
import { toParameterVector } from '../serialize/network.serialize.utils';
import type {
  ParameterLayoutEntry,
  ParameterVector,
} from '../serialize/network.serialize.utils.types';
import { fineTuneVector } from './network.training.isolate.utils';
import type { TrainingSample } from './network.training.utils.types';

type ParameterVectorSnapshot = {
  entries: string[];
  values: number[];
  version: number;
};

function createIsolationBaseNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function createOrderedFineTuneDataset(): TrainingSample[] {
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

function createFineTuneOptionsWithoutSeed() {
  return {
    learningRate: 0.25,
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

describe('network training chapter', () => {
  describe('fineTuneVector isolation helper', () => {
    describe('given a parameter vector and its base network are reused after fine-tuning', () => {
      describe('when fineTuneVector returns a trained vector', () => {
        it('keeps the input vector snapshot and base-network snapshot unchanged', () => {
          // Arrange
          const baseNetwork = createIsolationBaseNetwork(401);
          const parameterVector = toParameterVector(baseNetwork);
          const orderedFineTuneDataset = createOrderedFineTuneDataset();
          const baselineSnapshots = {
            baseNetwork: summarizeParameterVector(
              toParameterVector(baseNetwork),
            ),
            inputVector: summarizeParameterVector(parameterVector),
          };

          // Act
          fineTuneVector(
            baseNetwork,
            parameterVector,
            orderedFineTuneDataset,
            createFineTuneOptions(801),
          );
          const postRunSnapshots = {
            baseNetwork: summarizeParameterVector(
              toParameterVector(baseNetwork),
            ),
            inputVector: summarizeParameterVector(parameterVector),
          };

          // Assert
          expect(postRunSnapshots).toEqual(baselineSnapshots);
        });
      });
    });

    describe('given two fine-tune runs use the same seed and dataset order', () => {
      describe('when fineTuneVector runs twice on the same runtime', () => {
        it('returns element-wise identical trained-vector snapshots', () => {
          // Arrange
          const baseNetwork = createIsolationBaseNetwork(402);
          const parameterVector = toParameterVector(baseNetwork);
          const orderedFineTuneDataset = createOrderedFineTuneDataset();
          const fineTuneOptions = createFineTuneOptions(802);

          // Act
          const firstFineTuneResult = fineTuneVector(
            baseNetwork,
            parameterVector,
            orderedFineTuneDataset,
            fineTuneOptions,
          );
          const secondFineTuneResult = fineTuneVector(
            baseNetwork,
            parameterVector,
            orderedFineTuneDataset,
            fineTuneOptions,
          );
          const trainedVectorSnapshots = {
            first: summarizeParameterVector(firstFineTuneResult.trainedVector),
            second: summarizeParameterVector(
              secondFineTuneResult.trainedVector,
            ),
          };

          // Assert
          expect(trainedVectorSnapshots.first).toEqual(
            trainedVectorSnapshots.second,
          );
        });
      });
    });

    describe('given fineTuneVector runs without an explicit seed', () => {
      describe('when the helper fine-tunes on its working copy only', () => {
        it('keeps the input vector snapshot and base-network snapshot unchanged', () => {
          // Arrange
          const baseNetwork = createIsolationBaseNetwork(403);
          const parameterVector = toParameterVector(baseNetwork);
          const orderedFineTuneDataset = createOrderedFineTuneDataset();
          const baselineSnapshots = {
            baseNetwork: summarizeParameterVector(
              toParameterVector(baseNetwork),
            ),
            inputVector: summarizeParameterVector(parameterVector),
          };

          // Act
          fineTuneVector(
            baseNetwork,
            parameterVector,
            orderedFineTuneDataset,
            createFineTuneOptionsWithoutSeed(),
          );
          const postRunSnapshots = {
            baseNetwork: summarizeParameterVector(
              toParameterVector(baseNetwork),
            ),
            inputVector: summarizeParameterVector(parameterVector),
          };

          // Assert
          expect(postRunSnapshots).toEqual(baselineSnapshots);
        });
      });
    });
  });
});
