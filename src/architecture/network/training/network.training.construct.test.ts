import Node from '../../node';
import Network from '../network';

type TrainingDataset = Parameters<Network['train']>[0];

type ConstructedTrainingScenario = {
  network: Network;
  trainingDataset: TrainingDataset;
  probeInputValues: number[];
  inputNodeIds: number[];
  outputNodeIds: number[];
};

const INPUT_TO_HIDDEN_LEFT_WEIGHT = 0.05;
const INPUT_TO_HIDDEN_RIGHT_WEIGHT = -0.04;
const HIDDEN_TO_OUTPUT_WEIGHT = 0.03;
const HIDDEN_BIAS = 0;
const OUTPUT_BIAS = 0;
const TRAINING_RATE = 0.3;
const TRAINING_ITERATIONS = 40;
const TRAINING_TARGET_ERROR = 0.01;

function createConstructedTrainingScenario(): ConstructedTrainingScenario {
  const leftSensor = new Node('input');
  const rightSensor = new Node('input');
  const hiddenNode = new Node('hidden');
  const readoutNode = new Node('output');

  leftSensor.describe({ label: 'leftSensor' });
  rightSensor.describe({ label: 'rightSensor' });
  readoutNode.describe({ label: 'readout' });

  leftSensor.connect(hiddenNode);
  rightSensor.connect(hiddenNode);
  hiddenNode.connect(readoutNode);

  leftSensor.connections.out[0].weight = INPUT_TO_HIDDEN_LEFT_WEIGHT;
  rightSensor.connections.out[0].weight = INPUT_TO_HIDDEN_RIGHT_WEIGHT;
  hiddenNode.connections.out[0].weight = HIDDEN_TO_OUTPUT_WEIGHT;
  hiddenNode.bias = HIDDEN_BIAS;
  readoutNode.bias = OUTPUT_BIAS;

  const network = Network.construct(
    [hiddenNode, rightSensor, readoutNode, leftSensor],
    {
      inputNodes: ['rightSensor', 'leftSensor'],
      outputNodes: ['readout'],
    },
  ).network;

  return {
    network,
    trainingDataset: [{ input: [0.85, 0.15], output: [1] }],
    probeInputValues: [0.85, 0.15],
    inputNodeIds: [rightSensor.geneId, leftSensor.geneId],
    outputNodeIds: [readoutNode.geneId],
  };
}

function measureMeanAbsoluteOutputError(
  network: Network,
  trainingDataset: TrainingDataset,
): number {
  const totalAbsoluteError = trainingDataset.reduce(
    (runningTotal, trainingSample) => {
      network.clear();
      const [actualOutputValue = 0] = network.activate(trainingSample.input);

      return (
        runningTotal + Math.abs(trainingSample.output[0] - actualOutputValue)
      );
    },
    0,
  );

  return totalAbsoluteError / trainingDataset.length;
}

describe('network training chapter', () => {
  describe('construct-built feed-forward runtimes', () => {
    describe('given one constructed runtime has already activated a training sample', () => {
      describe('when propagate() is called with one valid target', () => {
        it('updates the live connection weights without adapter glue', () => {
          // Arrange
          const { network, probeInputValues } = createConstructedTrainingScenario();
          const initialConnectionWeights = network.connections.map(
            (connection) => connection.weight,
          );

          network.activate(probeInputValues, true);

          // Act
          network.propagate(TRAINING_RATE, 0, true, [1]);
          const updatedConnectionWeights = network.connections.map(
            (connection) => connection.weight,
          );

          // Assert
          expect(updatedConnectionWeights).not.toEqual(initialConnectionWeights);
        });
      });
    });

    describe('given one constructed runtime uses explicit input and output ids', () => {
      describe('when train() runs on a simple supervised sample', () => {
        it('improves the sample error while preserving public IO and topology intent', () => {
          // Arrange
          const {
            network,
            trainingDataset,
            inputNodeIds,
            outputNodeIds,
          } = createConstructedTrainingScenario();
          const initialMeanAbsoluteError = measureMeanAbsoluteOutputError(
            network,
            trainingDataset,
          );

          // Act
          network.train(trainingDataset, {
            iterations: TRAINING_ITERATIONS,
            error: TRAINING_TARGET_ERROR,
            rate: TRAINING_RATE,
          });
          const trainedMeanAbsoluteError = measureMeanAbsoluteOutputError(
            network,
            trainingDataset,
          );
          const actualTrainingBoundarySummary = {
            improvedError:
              trainedMeanAbsoluteError < initialMeanAbsoluteError,
            topologyIntent: network.getTopologyIntent(),
            inputNodeIds: network.inputNodeIds,
            outputNodeIds: network.outputNodeIds,
          };

          // Assert
          expect(actualTrainingBoundarySummary).toEqual({
            improvedError: true,
            topologyIntent: 'feed-forward',
            inputNodeIds,
            outputNodeIds,
          });
        });
      });
    });
  });
});