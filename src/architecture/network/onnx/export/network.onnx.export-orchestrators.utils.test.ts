import type { OnnxModel } from '../schema/network.onnx.schema.types';
import Node from '../../../node';
import {
  appendLstmPatternStubMetadata,
  collectLstmPatternStubs,
} from './network.onnx.export-orchestrators.utils';
import type { LstmPatternStub } from './network.onnx.export.types';

function createMinimalOnnxModel(
  metadataProps?: OnnxModel['metadata_props'],
): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
    },
    metadata_props: metadataProps,
  } as unknown as OnnxModel;
}

describe('network onnx export orchestrators utility chapter', () => {
  describe('collectLstmPatternStubs', () => {
    describe('given a valid 10-node hidden layer with self-connected memory neurons', () => {
      it('returns one heuristic LSTM stub', () => {
        // Arrange
        const inputLayer = [new Node('input')];
        const hiddenLayer = Array.from({ length: 10 }, () => new Node('hidden'));
        const outputLayer = [new Node('output')];
        hiddenLayer[4].connect(hiddenLayer[4]);
        hiddenLayer[5].connect(hiddenLayer[5]);

        // Act
        const lstmPatternStubs = collectLstmPatternStubs(
          [inputLayer, hiddenLayer, outputLayer],
          true,
        );

        // Assert
        expect(lstmPatternStubs).toEqual([{ layerIndex: 1, unitSize: 2 }]);
      });
    });

    describe('given the memory slice contains a malformed hidden node', () => {
      it('returns an empty stub list from the guarded fallback path', () => {
        // Arrange
        const inputLayer = [new Node('input')];
        const hiddenLayer = Array.from({ length: 10 }, () => new Node('hidden'));
        const outputLayer = [new Node('output')];
        hiddenLayer[4] = {} as unknown as Node;
        hiddenLayer[5].connect(hiddenLayer[5]);

        // Act
        const lstmPatternStubs = collectLstmPatternStubs(
          [inputLayer, hiddenLayer, outputLayer],
          true,
        );

        // Assert
        expect(lstmPatternStubs).toEqual([]);
      });
    });
  });

  describe('appendLstmPatternStubMetadata', () => {
    describe('given the model has no existing metadata_props', () => {
      it('initializes metadata_props from the null-coalescing fallback and appends the stub entry', () => {
        // Arrange – metadata_props is undefined → line 445 FALSE arm (model.metadata_props ?? [])
        const model = createMinimalOnnxModel(undefined);
        const stub: LstmPatternStub = {
          layerIndex: 0,
          unitSize: 2,
        };

        // Act
        appendLstmPatternStubMetadata(model, [stub]);

        // Assert
        expect(model.metadata_props).toEqual([
          {
            key: 'lstm_groups_stub',
            value: JSON.stringify([stub]),
          },
        ]);
      });
    });
  });
});
