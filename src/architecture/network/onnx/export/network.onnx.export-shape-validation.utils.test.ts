import type { OnnxModel } from '../schema/network.onnx.schema.types';
import { NetworkOnnxShapeValidationError } from '../network.onnx.errors';
import { validateOnnxModelShapes } from './network.onnx.export-shape-validation.utils';

function createValueInfo(
  name: string,
  dims: Array<number | string | null>,
): OnnxModel['graph']['inputs'][number] {
  return {
    name,
    type: {
      tensor_type: {
        elem_type: 1,
        shape: {
          dim: dims.map((dimensionValue) => {
            if (typeof dimensionValue === 'number') {
              return { dim_value: dimensionValue };
            }

            if (typeof dimensionValue === 'string') {
              return { dim_param: dimensionValue };
            }

            return {};
          }),
        },
      },
    },
  };
}

function createTensor(
  name: string,
  dims: number[],
  floatData: number[] | number,
): OnnxModel['graph']['initializer'][number] {
  return {
    name,
    data_type: 1,
    dims,
    float_data:
      typeof floatData === 'number'
        ? Array.from({ length: floatData }, () => 0)
        : floatData,
  };
}

function createInt64Tensor(
  name: string,
  dims: number[],
  int64Data: number[],
): OnnxModel['graph']['initializer'][number] {
  return {
    name,
    data_type: 7,
    dims,
    float_data: [],
    int64_data: int64Data,
  };
}

function createInt32Tensor(
  name: string,
  dims: number[],
  int32Data: number[],
): OnnxModel['graph']['initializer'][number] {
  return {
    name,
    data_type: 10,
    dims,
    float_data: [],
    int32_data: int32Data,
  };
}

function createModel(
  graphOverrides: Partial<OnnxModel['graph']>,
  metadataProps: OnnxModel['metadata_props'] = [],
): OnnxModel {
  return {
    graph: {
      inputs: [],
      outputs: [],
      initializer: [],
      node: [],
      ...graphOverrides,
    },
    metadata_props: metadataProps,
  };
}

describe('network onnx export shape-validation utils chapter', () => {
  describe('validateOnnxModelShapes()', () => {
    describe('given a compact dense Gemm graph', () => {
      it('accepts the inferred tensor dimensions', () => {
        // Arrange
        const onnxModel: OnnxModel = {
          graph: {
            inputs: [createValueInfo('input', [2])],
            outputs: [],
            initializer: [
              createTensor('W0', [3, 2], 6),
              createTensor('B0', [3], 3),
            ],
            node: [
              {
                op_type: 'Gemm',
                input: ['input', 'W0', 'B0'],
                output: ['Gemm_1'],
                name: 'gemm_l1',
                attributes: [{ name: 'transB', type: 'INT', i: 1 }],
              },
              {
                op_type: 'Tanh',
                input: ['Gemm_1'],
                output: ['Layer_1'],
                name: 'act_l1',
              },
            ],
          },
        };

        // Act + Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });

    describe('given a Gemm node whose bias width does not match the output width', () => {
      it('throws a shape validation error', () => {
        // Arrange
        const onnxModel: OnnxModel = {
          graph: {
            inputs: [createValueInfo('input', [2])],
            outputs: [],
            initializer: [
              createTensor('W0', [3, 2], 6),
              createTensor('B0', [2], 2),
            ],
            node: [
              {
                op_type: 'Gemm',
                input: ['input', 'W0', 'B0'],
                output: ['Gemm_1'],
                name: 'gemm_l1',
                attributes: [{ name: 'transB', type: 'INT', i: 1 }],
              },
            ],
          },
        };

        // Act
        const validationCallback = () => validateOnnxModelShapes(onnxModel);

        // Assert
        expect(validationCallback).toThrow(NetworkOnnxShapeValidationError);
      });
    });

    describe('given a Concat node whose non-concatenated dimensions disagree', () => {
      it('throws a shape validation error', () => {
        // Arrange
        const onnxModel: OnnxModel = {
          graph: {
            inputs: [
              createValueInfo('left', [2, 1]),
              createValueInfo('right', [3, 1]),
            ],
            outputs: [],
            initializer: [],
            node: [
              {
                op_type: 'Concat',
                input: ['left', 'right'],
                output: ['merged'],
                name: 'concat_l1',
                attributes: [{ name: 'axis', type: 'INT', i: 1 }],
              },
            ],
          },
        };

        // Act
        const validationCallback = () => validateOnnxModelShapes(onnxModel);

        // Assert
        expect(validationCallback).toThrow(NetworkOnnxShapeValidationError);
      });
    });

    describe('given initializer payload edge cases', () => {
      it('rejects mismatched int32 payload sizes', () => {
        // Arrange
        const onnxModel = createModel({
          initializer: [createInt32Tensor('W0', [2], [1])],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects mismatched int64 payload sizes', () => {
        // Arrange
        const onnxModel = createModel({
          initializer: [createInt64Tensor('shape', [2], [1])],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects mismatched float payload sizes', () => {
        // Arrange
        const onnxModel = createModel({
          initializer: [createTensor('W0', [2, 2], 3)],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });
    });

    describe('given graph-ledger setup edge cases', () => {
      it('defaults omitted input dimensions to one', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [null, 2])],
          node: [
            {
              op_type: 'Tanh',
              input: ['input'],
              output: ['Layer_1'],
              name: 'act_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects unresolved input tensors when no node can be resolved', () => {
        // Arrange
        const onnxModel = createModel({
          node: [
            {
              op_type: 'Relu',
              input: ['missing'],
              output: ['Layer_1'],
              name: 'act_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects nodes whose declared output count does not match inferred outputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          node: [
            {
              op_type: 'Relu',
              input: ['input'],
              output: ['Layer_1', 'Layer_2'],
              name: 'act_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects unsupported operators', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          node: [
            {
              op_type: 'CustomOp',
              input: ['input'],
              output: ['Layer_1'],
              name: 'custom_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects DynamicQuantizeLinear nodes that omit the scale or zero-point outputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          node: [
            {
              op_type: 'DynamicQuantizeLinear',
              input: ['input'],
              output: ['quantized', 'scale'],
              name: 'dynamic_quantize_input_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('resolves deferred nodes after a later node seeds the missing tensor shape', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          node: [
            {
              op_type: 'Relu',
              input: ['mid'],
              output: ['out'],
              name: 'act_l2',
            },
            {
              op_type: 'Tanh',
              input: ['input'],
              output: ['mid'],
              name: 'act_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('formats unnamed-node validation failures with the unnamed placeholder', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          node: [
            {
              op_type: 'CustomOp',
              input: ['input'],
              output: ['out'],
              name: '',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          '<unnamed-node>',
        );
      });
    });

    describe('given Concat edge cases beyond plain dimension mismatch', () => {
      it('rejects Concat nodes without any inputs', () => {
        // Arrange
        const onnxModel = createModel({
          node: [
            {
              op_type: 'Concat',
              input: [],
              output: ['merged'],
              name: 'concat_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects Concat inputs with different ranks', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [2, 1]),
            createValueInfo('right', [2, 1, 1]),
          ],
          node: [
            {
              op_type: 'Concat',
              input: ['left', 'right'],
              output: ['merged'],
              name: 'concat_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts mixed symbolic and numeric dimensions outside the concat axis', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', ['batch', 2]),
            createValueInfo('right', [5, 3]),
          ],
          node: [
            {
              op_type: 'Concat',
              input: ['left', 'right'],
              output: ['merged'],
              name: 'concat_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts symbolic axis dimensions by folding them into a combined token', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [2, 'seq']),
            createValueInfo('right', [2, 3]),
          ],
          node: [
            {
              op_type: 'Concat',
              input: ['left', 'right'],
              output: ['merged'],
              name: 'concat_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('uses axis zero by default when Concat does not declare one', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [1, 2]),
            createValueInfo('right', [1, 2]),
          ],
          node: [
            {
              op_type: 'Concat',
              input: ['left', 'right'],
              output: ['merged'],
              name: 'concat_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });

    describe('given Gemm edge cases beyond simple bias width mismatch', () => {
      it('rejects rank-3 left inputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 2, 2])],
          initializer: [
            createTensor('W0', [3, 2], 6),
            createTensor('B0', [3], 3),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects scalar Gemm inputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [])],
          initializer: [
            createTensor('W0', [3, 2], 6),
            createTensor('B0', [3], 3),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects non-matrix Gemm weights', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          initializer: [
            createTensor('W0', [3], 3),
            createTensor('B0', [3], 3),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts matrix-shaped Gemm biases when they broadcast to the output', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 2])],
          initializer: [
            createTensor('W0', [3, 2], 6),
            createTensor('B0', [2, 3], 6),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects Gemm biases with incompatible ranks', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 2])],
          initializer: [
            createTensor('W0', [3, 2], 6),
            createTensor('B0', [2, 3, 1], 6),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('falls back to a propagated pre-reshape input shape when the reshaped rank is incompatible', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [4])],
          initializer: [
            createInt64Tensor('shape', [2], [2, 2]),
            createTensor('W0', [3, 4], 12),
            createTensor('B0', [3], 3),
          ],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
            {
              op_type: 'Gemm',
              input: ['reshaped', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l2',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts transposed vector inputs for Gemm', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [4])],
          initializer: [
            createTensor('W0', [3, 1], 3),
            createTensor('B0', [3], 3),
          ],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0', 'B0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [
                { name: 'transA', type: 'INT', i: 1 },
                { name: 'transB', type: 'INT', i: 1 },
              ],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Gemm nodes without a bias tensor', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2])],
          initializer: [createTensor('W0', [3, 2], 6)],
          node: [
            {
              op_type: 'Gemm',
              input: ['input', 'W0'],
              output: ['Gemm_1'],
              name: 'gemm_l1',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('throws when a propagated fallback collapses to a scalar shape', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [])],
          initializer: [
            createInt64Tensor('shape', [1], [1]),
            createTensor('W0', [3, 2], 6),
          ],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
            {
              op_type: 'Gemm',
              input: ['reshaped', 'W0'],
              output: ['Gemm_1'],
              name: 'gemm_l2',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('throws when both the reshaped input and its propagated fallback remain incompatible', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [4])],
          initializer: [
            createInt64Tensor('shape', [2], [2, 2]),
            createTensor('W0', [3, 3], 9),
          ],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
            {
              op_type: 'Gemm',
              input: ['reshaped', 'W0'],
              output: ['Gemm_1'],
              name: 'gemm_l2',
              attributes: [{ name: 'transB', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });
    });

    describe('given reshape, flatten, transpose, and softmax edge cases', () => {
      it('requires an int64 reshape shape tensor', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          initializer: [createTensor('shape', [2], 2)],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects reshape dimensions smaller than negative one', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          initializer: [createInt64Tensor('shape', [2], [2, -2])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects reshape tensors with multiple inferred dimensions', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          initializer: [createInt64Tensor('shape', [2], [-1, -1])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('infers reshape zero-copy and single inferred dimensions', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 6])],
          initializer: [createInt64Tensor('shape', [2], [0, -1])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts symbolic reshape inference by folding dimensions into a token', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', ['batch', 4])],
          initializer: [createInt64Tensor('shape', [1], [-1])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects incompatible reshape products', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          initializer: [createInt64Tensor('shape', [2], [4, 2])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('uses the default reverse permutation for Transpose nodes', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3, 4])],
          node: [
            {
              op_type: 'Transpose',
              input: ['input'],
              output: ['transposed'],
              name: 'transpose_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects invalid Transpose permutations', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3, 4])],
          node: [
            {
              op_type: 'Transpose',
              input: ['input'],
              output: ['transposed'],
              name: 'transpose_l1',
              attributes: [{ name: 'perm', type: 'INTS', ints: [0, 1] }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts Softmax nodes with negative axes', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'Softmax',
              input: ['input'],
              output: ['normalized'],
              name: 'softmax_l1',
              attributes: [{ name: 'axis', type: 'INT', i: -1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects Softmax axes that fall outside the tensor rank', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [3])],
          node: [
            {
              op_type: 'Softmax',
              input: ['input'],
              output: ['normalized'],
              name: 'softmax_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 2 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts Flatten nodes whose leading fold is empty', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'Flatten',
              input: ['input'],
              output: ['flat'],
              name: 'flatten_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 0 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Flatten nodes with symbolic prefixes', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', ['batch', 2, 3])],
          node: [
            {
              op_type: 'Flatten',
              input: ['input'],
              output: ['flat'],
              name: 'flatten_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 2 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Flatten nodes whose input is already rank one', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [4])],
          node: [
            {
              op_type: 'Flatten',
              input: ['input'],
              output: ['flat'],
              name: 'flatten_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Flatten nodes with the default axis value', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'Flatten',
              input: ['input'],
              output: ['flat'],
              name: 'flatten_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts symbolic flatten prefixes whose numeric product is one', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', ['batch', 3])],
          node: [
            {
              op_type: 'Flatten',
              input: ['input'],
              output: ['flat'],
              name: 'flatten_l1',
              attributes: [{ name: 'axis', type: 'INT', i: 1 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });

    describe('given MatMul edge cases', () => {
      it('rejects scalar MatMul inputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', []),
            createValueInfo('right', [1, 1]),
          ],
          node: [
            {
              op_type: 'MatMul',
              input: ['left', 'right'],
              output: ['product'],
              name: 'matmul_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts MatMul when the left input is a vector', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [3]),
            createValueInfo('right', [3, 2]),
          ],
          node: [
            {
              op_type: 'MatMul',
              input: ['left', 'right'],
              output: ['product'],
              name: 'matmul_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts MatMul when the right input is a vector', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [2, 3]),
            createValueInfo('right', [3]),
          ],
          node: [
            {
              op_type: 'MatMul',
              input: ['left', 'right'],
              output: ['product'],
              name: 'matmul_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });

    describe('given QLinearMatMul edge cases', () => {
      it('rejects scalar QLinearMatMul inputs', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('left', [])],
          initializer: [
            createTensor('left_scale', [], [1]),
            createInt32Tensor('left_zero_point', [], [0]),
            createInt32Tensor('right', [1, 1], [1]),
            createTensor('right_scale', [], [1]),
            createInt32Tensor('right_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
          ],
          node: [
            {
              op_type: 'QLinearMatMul',
              input: [
                'left',
                'left_scale',
                'left_zero_point',
                'right',
                'right_scale',
                'right_zero_point',
                'output_scale',
                'output_zero_point',
              ],
              output: ['product'],
              name: 'qlinear_matmul_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts QLinearMatMul when the right input is a vector', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('left', [2, 3])],
          initializer: [
            createTensor('left_scale', [], [1]),
            createInt32Tensor('left_zero_point', [], [0]),
            createInt32Tensor('right', [3], [1, 1, 1]),
            createTensor('right_scale', [], [1]),
            createInt32Tensor('right_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
          ],
          node: [
            {
              op_type: 'QLinearMatMul',
              input: [
                'left',
                'left_scale',
                'left_zero_point',
                'right',
                'right_scale',
                'right_zero_point',
                'output_scale',
                'output_zero_point',
              ],
              output: ['product'],
              name: 'qlinear_matmul_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });

    describe('given QLinearConv edge cases', () => {
      it('rejects QLinearConv weights that are not rank four', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('input_scale', [], [1]),
            createInt32Tensor('input_zero_point', [], [0]),
            createInt32Tensor('W0', [2, 1, 9], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            createTensor('weight_scale', [], [1]),
            createInt32Tensor('weight_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
          ],
          node: [
            {
              op_type: 'QLinearConv',
              input: [
                'input',
                'input_scale',
                'input_zero_point',
                'W0',
                'weight_scale',
                'weight_zero_point',
                'output_scale',
                'output_zero_point',
              ],
              output: ['conv'],
              name: 'qlinear_conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts QLinearConv with a fused bias even when the node name has no layer index', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('input_scale', [], [1]),
            createInt32Tensor('input_zero_point', [], [0]),
            createInt32Tensor(
              'W0',
              [2, 1, 3, 3],
              Array.from({ length: 18 }, () => 1),
            ),
            createTensor('weight_scale', [], [1]),
            createInt32Tensor('weight_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
            createInt32Tensor('B0', [2], [0, 0]),
          ],
          node: [
            {
              op_type: 'QLinearConv',
              input: [
                'input',
                'input_scale',
                'input_zero_point',
                'W0',
                'weight_scale',
                'weight_zero_point',
                'output_scale',
                'output_zero_point',
                'B0',
              ],
              output: ['conv'],
              name: 'qlinear_conv_custom',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts QLinearConv without a fused bias tensor', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('input_scale', [], [1]),
            createInt32Tensor('input_zero_point', [], [0]),
            createInt32Tensor(
              'W0',
              [2, 1, 3, 3],
              Array.from({ length: 18 }, () => 1),
            ),
            createTensor('weight_scale', [], [1]),
            createInt32Tensor('weight_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
          ],
          node: [
            {
              op_type: 'QLinearConv',
              input: [
                'input',
                'input_scale',
                'input_zero_point',
                'W0',
                'weight_scale',
                'weight_zero_point',
                'output_scale',
                'output_zero_point',
              ],
              output: ['conv'],
              name: 'qlinear_conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects QLinearConv inputs that are not rank four', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 4, 4])],
          initializer: [
            createTensor('input_scale', [], [1]),
            createInt32Tensor('input_zero_point', [], [0]),
            createInt32Tensor(
              'W0',
              [2, 1, 3, 3],
              Array.from({ length: 18 }, () => 1),
            ),
            createTensor('weight_scale', [], [1]),
            createInt32Tensor('weight_zero_point', [], [0]),
            createTensor('output_scale', [], [1]),
            createInt32Tensor('output_zero_point', [], [0]),
          ],
          node: [
            {
              op_type: 'QLinearConv',
              input: [
                'input',
                'input_scale',
                'input_zero_point',
                'W0',
                'weight_scale',
                'weight_zero_point',
                'output_scale',
                'output_zero_point',
              ],
              output: ['conv'],
              name: 'qlinear_conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });
    });

    describe('given broadcast and Conv edge cases', () => {
      it('rejects binary operators that are missing a required input tensor', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('left', [2, 3])],
          node: [
            {
              op_type: 'Add',
              input: ['left'],
              output: ['sum'],
              name: 'add_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects incompatible broadcast dimensions', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [2, 3]),
            createValueInfo('right', [4, 3]),
          ],
          node: [
            {
              op_type: 'Add',
              input: ['left', 'right'],
              output: ['sum'],
              name: 'add_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts symbolic broadcast dimensions when the symbols agree', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', ['batch', 3]),
            createValueInfo('right', ['batch', 1]),
          ],
          node: [
            {
              op_type: 'Add',
              input: ['left', 'right'],
              output: ['sum'],
              name: 'add_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Conv nodes without bias tensors', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [createTensor('W0', [2, 1, 3, 3], 18)],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0'],
              output: ['conv'],
              name: 'conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts broadcasts across different ranks by padding the shorter shape', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [
            createValueInfo('left', [3]),
            createValueInfo('right', [2, 3]),
          ],
          node: [
            {
              op_type: 'Add',
              input: ['left', 'right'],
              output: ['sum'],
              name: 'add_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('infers Conv output shapes from kernel dimensions when attributes are omitted', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('W0', [2, 1, 3, 3], 18),
            createTensor('B0', [2], 2),
          ],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0', 'B0'],
              output: ['conv'],
              name: 'conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('prefers declared Conv mapping output geometry when the flattened width matches', () => {
        // Arrange
        const onnxModel = createModel(
          {
            inputs: [createValueInfo('input', [1, 4])],
            initializer: [
              createTensor('W0', [2, 1, 1, 1], 2),
              createTensor('B0', [2], 2),
            ],
            node: [
              {
                op_type: 'Conv',
                input: ['input', 'W0', 'B0'],
                output: ['conv'],
                name: 'conv_l1',
              },
            ],
          },
          [
            {
              key: 'conv2d_specs',
              value: JSON.stringify([
                {
                  layerIndex: 1,
                  inHeight: 2,
                  inWidth: 2,
                  inChannels: 1,
                  kernelHeight: 1,
                  kernelWidth: 1,
                  strideHeight: 1,
                  strideWidth: 1,
                  outHeight: 2,
                  outWidth: 2,
                  outChannels: 2,
                },
              ]),
            },
          ],
        );

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts Conv nodes whose names do not encode a layer index', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('W0', [1, 1, 1, 1], 1),
            createTensor('B0', [1], 1),
          ],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0', 'B0'],
              output: ['conv'],
              name: 'conv',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('reconstructs mapped Conv input geometry from a rank-one tensor when widths already match', () => {
        // Arrange
        const onnxModel = createModel(
          {
            inputs: [createValueInfo('input', [4])],
            initializer: [
              createTensor('W0', [1, 1, 1, 1], 1),
              createTensor('B0', [1], 1),
            ],
            node: [
              {
                op_type: 'Conv',
                input: ['input', 'W0', 'B0'],
                output: ['conv'],
                name: 'conv_l1',
              },
            ],
          },
          [
            {
              key: 'conv2d_specs',
              value: JSON.stringify([
                {
                  layerIndex: 1,
                  inHeight: 2,
                  inWidth: 2,
                  inChannels: 1,
                  kernelHeight: 1,
                  kernelWidth: 1,
                  strideHeight: 1,
                  strideWidth: 1,
                  outHeight: 2,
                  outWidth: 2,
                  outChannels: 1,
                },
              ]),
            },
          ],
        );

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects Conv weights that are not rank four', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('W0', [2, 3, 3], 18),
            createTensor('B0', [2], 2),
          ],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0', 'B0'],
              output: ['conv'],
              name: 'conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects non-spatial Conv inputs when no mapping can recover a rank-four shape', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [4, 4, 1])],
          initializer: [
            createTensor('W0', [2, 1, 3, 3], 18),
            createTensor('B0', [2], 2),
          ],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0', 'B0'],
              output: ['conv'],
              name: 'conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('rejects Conv biases that are not vectors', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          initializer: [
            createTensor('W0', [2, 1, 3, 3], 18),
            createTensor('B0', [2, 1], 2),
          ],
          node: [
            {
              op_type: 'Conv',
              input: ['input', 'W0', 'B0'],
              output: ['conv'],
              name: 'conv_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('reconstructs mapped Conv input geometry from a flattened vector when widths differ', () => {
        // Arrange
        const onnxModel = createModel(
          {
            inputs: [createValueInfo('input', [2])],
            initializer: [
              createTensor('W0', [1, 1, 1, 1], 1),
              createTensor('B0', [1], 1),
            ],
            node: [
              {
                op_type: 'Conv',
                input: ['input', 'W0', 'B0'],
                output: ['conv'],
                name: 'conv_l1',
              },
            ],
          },
          [
            {
              key: 'conv2d_specs',
              value: JSON.stringify([
                {
                  layerIndex: 1,
                  inHeight: 2,
                  inWidth: 2,
                  inChannels: 1,
                  kernelHeight: 1,
                  kernelWidth: 1,
                  strideHeight: 1,
                  strideWidth: 1,
                  outHeight: 2,
                  outWidth: 2,
                  outChannels: 1,
                },
              ]),
            },
          ],
        );

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('rejects symbolic spatial dimensions when windowed output needs numeric geometry', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 'height', 4])],
          node: [
            {
              op_type: 'MaxPool',
              input: ['input'],
              output: ['pooled'],
              name: 'pool_l1',
              attributes: [
                { name: 'kernel_shape', type: 'INTS', ints: [2, 2] },
                { name: 'strides', type: 'INTS', ints: [1, 1] },
              ],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts pool nodes with invalid strides by preserving the numeric input extent', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          node: [
            {
              op_type: 'MaxPool',
              input: ['input'],
              output: ['pooled'],
              name: 'pool_l1',
              attributes: [
                { name: 'kernel_shape', type: 'INTS', ints: [2, 2] },
                { name: 'strides', type: 'INTS', ints: [0, 0] },
              ],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('reconstructs mapped Conv input geometry from a rank-two tensor when widths differ', () => {
        // Arrange
        const onnxModel = createModel(
          {
            inputs: [createValueInfo('input', [2, 2])],
            initializer: [
              createTensor('W0', [1, 1, 1, 1], 1),
              createTensor('B0', [1], 1),
            ],
            node: [
              {
                op_type: 'Conv',
                input: ['input', 'W0', 'B0'],
                output: ['conv'],
                name: 'conv_l1',
              },
            ],
          },
          [
            {
              key: 'conv2d_specs',
              value: JSON.stringify([
                {
                  layerIndex: 1,
                  inHeight: 2,
                  inWidth: 2,
                  inChannels: 1,
                  kernelHeight: 1,
                  kernelWidth: 1,
                  strideHeight: 1,
                  strideWidth: 1,
                  outHeight: 2,
                  outWidth: 2,
                  outChannels: 1,
                },
              ]),
            },
          ],
        );

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts pooled tensors whose inputs are already non-spatial', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'MaxPool',
              input: ['input'],
              output: ['pooled'],
              name: 'pool_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts pooled tensors when the output window would collapse below one cell', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 1, 1])],
          node: [
            {
              op_type: 'MaxPool',
              input: ['input'],
              output: ['pooled'],
              name: 'pool_l1',
              attributes: [
                { name: 'kernel_shape', type: 'INTS', ints: [3, 3] },
                { name: 'strides', type: 'INTS', ints: [1, 1] },
              ],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts pooled tensors when kernel and stride attributes fall back to defaults', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [1, 1, 4, 4])],
          node: [
            {
              op_type: 'MaxPool',
              input: ['input'],
              output: ['pooled'],
              name: 'pool_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('copies zero-valued reshape dimensions from missing input positions as zeros', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          initializer: [createInt64Tensor('shape', [3], [0, 0, 0])],
          node: [
            {
              op_type: 'Reshape',
              input: ['input', 'shape'],
              output: ['reshaped'],
              name: 'reshape_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });
    });

    describe('given fused recurrent edge cases', () => {
      it('requires hidden_size for fused recurrent operators', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'GRU',
              input: ['input'],
              output: ['gru'],
              name: 'gru_l1',
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).toThrow(
          NetworkOnnxShapeValidationError,
        );
      });

      it('accepts rank-one recurrent sources by returning a hidden vector', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [3])],
          node: [
            {
              op_type: 'GRU',
              input: ['input'],
              output: ['gru'],
              name: 'gru_l1',
              attributes: [{ name: 'hidden_size', type: 'INT', i: 4 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });

      it('accepts rank-two recurrent sources by preserving the batch dimension', () => {
        // Arrange
        const onnxModel = createModel({
          inputs: [createValueInfo('input', [2, 3])],
          node: [
            {
              op_type: 'GRU',
              input: ['input'],
              output: ['gru'],
              name: 'gru',
              attributes: [{ name: 'hidden_size', type: 'INT', i: 4 }],
            },
          ],
        });

        // Assert
        expect(() => validateOnnxModelShapes(onnxModel)).not.toThrow();
      });
    });
  });
});