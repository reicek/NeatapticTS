import type { OnnxModel } from './network.onnx.schema.types';
import { serializeOnnxModelToBinary } from './network.onnx.schema.binary.utils';

function containsBytes(
  binaryPayload: Uint8Array,
  byteSequence: number[],
): boolean {
  return Buffer.from(binaryPayload).includes(Buffer.from(byteSequence));
}

describe('network onnx schema binary utility chapter', () => {
  describe('serializeOnnxModelToBinary', () => {
    it('encodes a minimal ModelProto shell into deterministic protobuf bytes', () => {
      // Arrange
      const onnxModel: OnnxModel = {
        ir_version: 9,
        opset_import: [{ domain: '', version: 18 }],
        graph: {
          inputs: [],
          outputs: [],
          initializer: [],
          node: [],
        },
      };

      // Act
      const binaryModel = serializeOnnxModelToBinary(onnxModel);

      // Assert
      expect(Array.from(binaryModel)).toEqual([8, 9, 58, 0, 66, 2, 16, 18]);
    });

    it('serializes the current typed-field schema surface into protobuf field bytes', () => {
      // Arrange
      const onnxModel: OnnxModel = {
        ir_version: 9,
        producer_name: 'phase8-schema',
        producer_version: '1.2.3',
        doc_string: 'schema-rich-model',
        opset_import: [{ domain: 'ai.onnx', version: 300 }],
        metadata_props: [{ key: 'stage', value: 'phase8' }],
        graph: {
          inputs: [
            {
              name: 'input',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: {
                    dim: [{ dim_param: 'N' }, { dim_value: 128 }],
                  },
                },
              },
            },
          ],
          outputs: [
            {
              name: 'output',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: {
                    dim: [{ dim_value: 1 }],
                  },
                },
              },
            },
          ],
          initializer: [
            {
              name: 'W0',
              data_type: 1,
              dims: [1, 128],
              float_data: [0.5, -1.25],
            },
            {
              name: 'Q0',
              data_type: 6,
              dims: [3],
              float_data: [],
              int32_data: [3, 300, -1],
            },
            {
              name: 'S0',
              data_type: 7,
              dims: [2],
              float_data: [],
              int64_data: [1, 128],
            },
            {
              name: 'BiasNoDims',
              data_type: 1,
              dims: [],
              float_data: [0.125],
            },
          ],
          node: [
            {
              name: 'Gemm_0',
              op_type: 'Gemm',
              input: ['input', 'W0', 'Q0'],
              output: ['hidden'],
              attributes: [
                { name: 'alpha_typed', type: 'FLOAT', f: 1.5 },
                { name: 'gain', f: 0.25 },
                { name: 'axis', i: -1 },
                { name: 'approximate', s: 'tanh' },
                { name: 'scales', floats: [0.5, -1.25] },
                { name: 'pads', ints: [1, 2, 3, 4] },
                { name: 'labels', strings: ['left', 'right'] },
                {
                  name: 'tensor_attr',
                  t: {
                    name: 'nested_tensor',
                    data_type: 1,
                    dims: [1],
                    float_data: [2],
                  },
                },
                {
                  name: 'graph_attr',
                  g: {
                    inputs: [
                      {
                        name: 'nested_graph_input',
                        type: {
                          tensor_type: {
                            elem_type: 1,
                            shape: { dim: [{ dim_value: 1 }] },
                          },
                        },
                      },
                    ],
                    outputs: [
                      {
                        name: 'nested_graph_output',
                        type: {
                          tensor_type: {
                            elem_type: 1,
                            shape: { dim: [{ dim_value: 1 }] },
                          },
                        },
                      },
                    ],
                    initializer: [
                      {
                        name: 'nested_graph_tensor',
                        data_type: 1,
                        dims: [1],
                        float_data: [1],
                      },
                    ],
                    node: [
                      {
                        name: 'NestedIdentity_0',
                        op_type: 'Identity',
                        input: ['nested_graph_input'],
                        output: ['nested_graph_output'],
                      },
                    ],
                  },
                },
                { name: 'empty' },
              ],
            },
            {
              name: '',
              op_type: 'Relu',
              input: ['hidden'],
              output: ['output'],
            },
          ],
        },
      };

      // Act
      const binaryModel = serializeOnnxModelToBinary(onnxModel);

      // Assert
      expect({
        hasProducerName: Buffer.from(binaryModel).includes(
          Buffer.from('phase8-schema'),
        ),
        hasProducerVersion: Buffer.from(binaryModel).includes(
          Buffer.from('1.2.3'),
        ),
        hasDocString: Buffer.from(binaryModel).includes(
          Buffer.from('schema-rich-model'),
        ),
        hasOpsetDomain: Buffer.from(binaryModel).includes(
          Buffer.from('ai.onnx'),
        ),
        hasMetadataPair:
          Buffer.from(binaryModel).includes(Buffer.from('stage')) &&
          Buffer.from(binaryModel).includes(Buffer.from('phase8')),
        hasGraphNames:
          Buffer.from(binaryModel).includes(Buffer.from('input')) &&
          Buffer.from(binaryModel).includes(Buffer.from('output')) &&
          Buffer.from(binaryModel).includes(Buffer.from('hidden')),
        hasTensorNames:
          Buffer.from(binaryModel).includes(Buffer.from('W0')) &&
          Buffer.from(binaryModel).includes(Buffer.from('Q0')) &&
          Buffer.from(binaryModel).includes(Buffer.from('S0')) &&
          Buffer.from(binaryModel).includes(Buffer.from('BiasNoDims')),
        hasNestedNames:
          Buffer.from(binaryModel).includes(Buffer.from('nested_tensor')) &&
          Buffer.from(binaryModel).includes(
            Buffer.from('nested_graph_tensor'),
          ) &&
          Buffer.from(binaryModel).includes(Buffer.from('nested_graph_output')),
        hasAttributeStrings:
          Buffer.from(binaryModel).includes(Buffer.from('tanh')) &&
          Buffer.from(binaryModel).includes(Buffer.from('left')) &&
          Buffer.from(binaryModel).includes(Buffer.from('right')),
        hasFloatPayload: containsBytes(
          binaryModel,
          [0, 0, 0, 63, 0, 0, 160, 191],
        ),
        hasPackedIntPayload: containsBytes(binaryModel, [3, 172, 2]),
        hasNegativeVarintPayload: containsBytes(
          binaryModel,
          [255, 255, 255, 255, 255, 255, 255, 255, 255, 1],
        ),
        hasAttributeTypeCode: containsBytes(binaryModel, [160, 1, 1]),
        hasBinaryLength: binaryModel.length > 0,
      }).toEqual({
        hasProducerName: true,
        hasProducerVersion: true,
        hasDocString: true,
        hasOpsetDomain: true,
        hasMetadataPair: true,
        hasGraphNames: true,
        hasTensorNames: true,
        hasNestedNames: true,
        hasAttributeStrings: true,
        hasFloatPayload: true,
        hasPackedIntPayload: true,
        hasNegativeVarintPayload: true,
        hasAttributeTypeCode: true,
        hasBinaryLength: true,
      });
    });

    it('serializes graph content even when the optional ir_version header is omitted', () => {
      // Arrange
      const onnxModel: OnnxModel = {
        graph: {
          inputs: [
            {
              name: 'input',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: { dim: [{ dim_value: 1 }] },
                },
              },
            },
          ],
          outputs: [
            {
              name: 'output',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: { dim: [{ dim_value: 1 }] },
                },
              },
            },
          ],
          initializer: [],
          node: [
            {
              name: '',
              op_type: 'Identity',
              input: ['input'],
              output: ['output'],
            },
          ],
        },
      };

      // Act
      const binaryModel = serializeOnnxModelToBinary(onnxModel);

      // Assert
      expect({
        hasBinaryLength: binaryModel.length > 0,
        hasInputName: Buffer.from(binaryModel).includes(Buffer.from('input')),
        hasOutputName: Buffer.from(binaryModel).includes(Buffer.from('output')),
        hasIdentityOp: Buffer.from(binaryModel).includes(
          Buffer.from('Identity'),
        ),
      }).toEqual({
        hasBinaryLength: true,
        hasInputName: true,
        hasOutputName: true,
        hasIdentityOp: true,
      });
    });

    it('falls back to URI-based UTF-8 encoding when TextEncoder is unavailable', () => {
      // Arrange
      const originalTextEncoder = globalThis.TextEncoder;
      const globalScope = globalThis as typeof globalThis & {
        TextEncoder?: typeof TextEncoder;
      };
      const onnxModel: OnnxModel = {
        graph: {
          inputs: [
            {
              name: 'Δinput',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: { dim: [{ dim_value: 1 }] },
                },
              },
            },
          ],
          outputs: [
            {
              name: '😀output',
              type: {
                tensor_type: {
                  elem_type: 1,
                  shape: { dim: [{ dim_value: 1 }] },
                },
              },
            },
          ],
          initializer: [],
          node: [
            {
              name: '',
              op_type: 'Identity',
              input: ['Δinput'],
              output: ['😀output'],
              attributes: [{ name: 'label', s: 'café' }],
            },
          ],
        },
      };

      // Act
      Reflect.deleteProperty(globalScope, 'TextEncoder');
      jest.resetModules();

      try {
        const { serializeOnnxModelToBinary: serializeWithoutTextEncoder } =
          require('./network.onnx.schema.binary.utils') as typeof import('./network.onnx.schema.binary.utils');
        const binaryModel = serializeWithoutTextEncoder(onnxModel);

        // Assert
        expect({
          hasDeltaBytes: containsBytes(binaryModel, [206, 148]),
          hasEmojiBytes: containsBytes(binaryModel, [240, 159, 152, 128]),
          hasCafeBytes: containsBytes(binaryModel, [99, 97, 102, 195, 169]),
        }).toEqual({
          hasDeltaBytes: true,
          hasEmojiBytes: true,
          hasCafeBytes: true,
        });
      } finally {
        globalScope.TextEncoder = originalTextEncoder;
        jest.resetModules();
      }
    });
  });
});
