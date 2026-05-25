import type { OnnxTensor } from './network.onnx.schema.types';

/** ONNX TensorProto enum value for float32 tensors. */
export const ONNX_FLOAT_DATA_TYPE = 1;

/** ONNX TensorProto enum value for float16 tensors. */
export const ONNX_FLOAT16_DATA_TYPE = 10;

const FLOAT32_EXPONENT_BIAS = 127;
const FLOAT16_EXPONENT_BIAS = 15;
const FLOAT32_SIGN_MASK = 0x8000_0000;
const FLOAT32_EXPONENT_MASK = 0x7f80_0000;
const FLOAT32_MANTISSA_MASK = 0x007f_ffff;
const FLOAT16_SIGN_MASK = 0x8000;
const FLOAT16_EXPONENT_MASK = 0x1f;
const FLOAT16_MANTISSA_MASK = 0x03ff;
const FLOAT16_INFINITY = 0x7c00;
const FLOAT16_NAN = 0x7e00;
const FLOAT16_MIN_NORMAL_EXPONENT = -14;
const FLOAT16_MIN_SUBNORMAL_EXPONENT = -24;
const FLOAT16_MANTISSA_SHIFT = 13;
const FLOAT16_MANTISSA_ROUNDING_BIAS = 0x1000;

const float32ValueBuffer = new Float32Array(1);
const float32BitsBuffer = new Uint32Array(float32ValueBuffer.buffer);
const float32DecodeBitsBuffer = new Uint32Array(1);
const float32DecodeValueBuffer = new Float32Array(
  float32DecodeBitsBuffer.buffer,
);

/**
 * Create a float16-backed tensor payload from float32 values.
 * This helper emits the exact storage fields expected by ONNX initializer writers so callers can downgrade precision while keeping exporter and importer tensor contracts structurally consistent.
 *
 * @param floatValues Float32-domain values to pack.
 * @returns ONNX tensor storage fields for a float16 initializer.
 */
export function createFloat16StoragePayload(
  floatValues: number[],
): Pick<OnnxTensor, 'data_type' | 'float_data' | 'int32_data'> {
  return {
    data_type: ONNX_FLOAT16_DATA_TYPE,
    float_data: [],
    int32_data: encodeFloat16Int32Data(floatValues),
  };
}

/**
 * Encode float32-domain values into packed float16 words stored as int32 entries.
 * Packing through this utility keeps round-trip behavior aligned with the paired decoder used by import and schema audit paths, including special-value handling.
 *
 * @param floatValues Float values to encode.
 * @returns Packed float16 words.
 */
export function encodeFloat16Int32Data(floatValues: number[]): number[] {
  return floatValues.map((floatValue) => encodeFloat16Bits(floatValue));
}

/**
 * Decode packed float16 words stored as int32 entries into float32-domain values.
 * Decoder output is normalized to JavaScript number values so upstream importer logic can reuse one scalar path regardless of original tensor storage precision.
 *
 * @param packedValues Packed float16 words.
 * @returns Decoded float values.
 */
export function decodeFloat16Int32Data(packedValues: number[]): number[] {
  return packedValues.map((packedValue) => decodeFloat16Bits(packedValue));
}

/**
 * Read a tensor's floating-point values regardless of whether it is stored as float32 or float16.
 * This abstraction gives importer and analysis utilities one read entrypoint that transparently handles native float shelves and packed float16 compatibility shelves.
 *
 * @param tensor Source ONNX tensor.
 * @returns Decoded floating-point values.
 */
export function readOnnxTensorFloatData(
  tensor: Pick<OnnxTensor, 'data_type' | 'float_data' | 'int32_data'>,
): number[] {
  if (tensor.data_type === ONNX_FLOAT16_DATA_TYPE) {
    return decodeFloat16Int32Data(tensor.int32_data ?? []);
  }

  return [...tensor.float_data];
}

/**
 * Encode one float32-domain value into one float16 word.
 *
 * @param floatValue Float value to encode.
 * @returns Packed float16 bits.
 */
function encodeFloat16Bits(floatValue: number): number {
  if (Number.isNaN(floatValue)) {
    return FLOAT16_NAN;
  }

  if (floatValue === Number.POSITIVE_INFINITY) {
    return FLOAT16_INFINITY;
  }

  if (floatValue === Number.NEGATIVE_INFINITY) {
    return FLOAT16_SIGN_MASK | FLOAT16_INFINITY;
  }

  float32ValueBuffer[0] = floatValue;
  const floatBits = float32BitsBuffer[0];
  const signBits = (floatBits >>> 16) & FLOAT16_SIGN_MASK;
  const rawExponent = (floatBits & FLOAT32_EXPONENT_MASK) >>> 23;
  const rawMantissa = floatBits & FLOAT32_MANTISSA_MASK;

  if (rawExponent === 0) {
    return signBits;
  }

  const unbiasedExponent = rawExponent - FLOAT32_EXPONENT_BIAS;
  if (unbiasedExponent > FLOAT16_EXPONENT_MASK) {
    return signBits | FLOAT16_INFINITY;
  }

  if (unbiasedExponent < FLOAT16_MIN_SUBNORMAL_EXPONENT) {
    return signBits;
  }

  if (unbiasedExponent < FLOAT16_MIN_NORMAL_EXPONENT) {
    const shiftedMantissa =
      (rawMantissa | 0x0080_0000) >>
      (FLOAT16_MIN_NORMAL_EXPONENT - unbiasedExponent);
    const roundedSubnormalMantissa =
      (shiftedMantissa + FLOAT16_MANTISSA_ROUNDING_BIAS) >>
      FLOAT16_MANTISSA_SHIFT;
    return signBits | roundedSubnormalMantissa;
  }

  let roundedMantissa = rawMantissa + FLOAT16_MANTISSA_ROUNDING_BIAS;
  let halfExponent = unbiasedExponent + FLOAT16_EXPONENT_BIAS;

  if (roundedMantissa & 0x0080_0000) {
    roundedMantissa = 0;
    halfExponent += 1;
  }

  if (halfExponent >= FLOAT16_EXPONENT_MASK) {
    return signBits | FLOAT16_INFINITY;
  }

  return (
    signBits |
    (halfExponent << 10) |
    ((roundedMantissa >> FLOAT16_MANTISSA_SHIFT) & FLOAT16_MANTISSA_MASK)
  );
}

/**
 * Decode one packed float16 word into a float32-domain value.
 *
 * @param packedValue Packed float16 bits.
 * @returns Decoded float value.
 */
function decodeFloat16Bits(packedValue: number): number {
  const signBits = (packedValue & FLOAT16_SIGN_MASK) << 16;
  const exponentBits = (packedValue >>> 10) & FLOAT16_EXPONENT_MASK;
  let mantissaBits = packedValue & FLOAT16_MANTISSA_MASK;

  if (exponentBits === 0) {
    if (mantissaBits === 0) {
      float32DecodeBitsBuffer[0] = signBits;
      return float32DecodeValueBuffer[0];
    }

    let normalizedExponent = FLOAT16_MIN_NORMAL_EXPONENT;
    while ((mantissaBits & 0x0400) === 0) {
      mantissaBits <<= 1;
      normalizedExponent -= 1;
    }

    mantissaBits &= FLOAT16_MANTISSA_MASK;
    float32DecodeBitsBuffer[0] =
      signBits |
      ((normalizedExponent + FLOAT32_EXPONENT_BIAS) << 23) |
      (mantissaBits << FLOAT16_MANTISSA_SHIFT);
    return float32DecodeValueBuffer[0];
  }

  if (exponentBits === FLOAT16_EXPONENT_MASK) {
    float32DecodeBitsBuffer[0] =
      signBits |
      FLOAT32_EXPONENT_MASK |
      (mantissaBits << FLOAT16_MANTISSA_SHIFT);
    return float32DecodeValueBuffer[0];
  }

  float32DecodeBitsBuffer[0] =
    signBits |
    ((exponentBits - FLOAT16_EXPONENT_BIAS + FLOAT32_EXPONENT_BIAS) << 23) |
    (mantissaBits << FLOAT16_MANTISSA_SHIFT);
  return float32DecodeValueBuffer[0];
}
