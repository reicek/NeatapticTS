import {
  createFloat16StoragePayload,
  decodeFloat16Int32Data,
  encodeFloat16Int32Data,
  ONNX_FLOAT16_DATA_TYPE,
  ONNX_FLOAT_DATA_TYPE,
  readOnnxTensorFloatData,
} from './network.onnx.schema.tensor-data.utils';

describe('network onnx schema tensor-data utility chapter', () => {
  describe('encodeFloat16Int32Data and decodeFloat16Int32Data', () => {
    it('round-trips common finite values within float16 tolerance', () => {
      // Arrange
      const sourceValues = [0, 1, -2.5, 0.33325];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect(
        decodedValues.every(
          (decodedValue, valueIndex) =>
            Math.abs(decodedValue - sourceValues[valueIndex]) < 1e-3,
        ),
      ).toBe(true);
    });

    it('preserves infinity and nan classifications', () => {
      // Arrange
      const sourceValues = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NaN];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect({
        positiveInfinity: decodedValues[0],
        negativeInfinity: decodedValues[1],
        nanIsNaN: Number.isNaN(decodedValues[2]),
      }).toEqual({
        positiveInfinity: Number.POSITIVE_INFINITY,
        negativeInfinity: Number.NEGATIVE_INFINITY,
        nanIsNaN: true,
      });
    });

    it('rounds overflowing finite values up to float16 infinity', () => {
      // Arrange
      const sourceValues = [65_520];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect(decodedValues[0]).toBe(Number.POSITIVE_INFINITY);
    });

    it('encodes extremely large finite values directly as float16 infinity', () => {
      // Arrange
      const sourceValues = [1e20];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect(decodedValues[0]).toBe(Number.POSITIVE_INFINITY);
    });

    it('underflows values smaller than the float16 subnormal floor to zero', () => {
      // Arrange
      const sourceValues = [1e-20];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect(decodedValues[0]).toBe(0);
    });

    it('encodes tiny finite values into float16 subnormals before decoding', () => {
      // Arrange
      const sourceValues = [1e-6];

      // Act
      const decodedValues = decodeFloat16Int32Data(
        encodeFloat16Int32Data(sourceValues),
      );

      // Assert
      expect(decodedValues[0] > 0 && decodedValues[0] < 6.2e-5).toBe(true);
    });

    it('decodes packed float16 subnormal payloads', () => {
      // Arrange
      const packedSubnormal = [1];

      // Act
      const decodedValues = decodeFloat16Int32Data(packedSubnormal);

      // Assert
      expect(decodedValues[0] > 0 && decodedValues[0] < 1e-4).toBe(true);
    });
  });

  describe('readOnnxTensorFloatData', () => {
    it('returns float_data unchanged for float32 tensors', () => {
      // Arrange
      const tensor = {
        data_type: ONNX_FLOAT_DATA_TYPE,
        float_data: [0.1, 0.2, 0.3],
      };

      // Act
      const decodedValues = readOnnxTensorFloatData(tensor);

      // Assert
      expect(decodedValues).toEqual([0.1, 0.2, 0.3]);
    });

    it('treats missing packed float16 payloads as an empty value array', () => {
      // Arrange
      const packedTensor = {
        data_type: ONNX_FLOAT16_DATA_TYPE,
        float_data: [],
      };

      // Act
      const decodedValues = readOnnxTensorFloatData(packedTensor);

      // Assert
      expect(decodedValues).toEqual([]);
    });

    it('decodes packed float16 payloads', () => {
      // Arrange
      const packedTensor = {
        data_type: ONNX_FLOAT16_DATA_TYPE,
        float_data: [],
        int32_data: encodeFloat16Int32Data([1.5, -0.5]),
      };

      // Act
      const decodedValues = readOnnxTensorFloatData(packedTensor);

      // Assert
      expect(
        decodedValues.every(
          (decodedValue, valueIndex) =>
            Math.abs(decodedValue - [1.5, -0.5][valueIndex]) < 1e-3,
        ),
      ).toBe(true);
    });
  });

  describe('createFloat16StoragePayload', () => {
    it('returns a float16 payload with empty float_data and packed int32 words', () => {
      // Arrange
      const sourceValues = [0.5, -1.25, 2];

      // Act
      const payload = createFloat16StoragePayload(sourceValues);

      // Assert
      expect(payload).toEqual({
        data_type: ONNX_FLOAT16_DATA_TYPE,
        float_data: [],
        int32_data: encodeFloat16Int32Data(sourceValues),
      });
    });
  });
});