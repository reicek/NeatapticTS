import type { OnnxModel } from '../schema/network.onnx.schema.types';
import { appendLstmPatternStubMetadata } from './network.onnx.export-orchestrators.utils';
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
