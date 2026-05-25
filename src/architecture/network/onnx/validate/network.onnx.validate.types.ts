/** External validator identifier for the current Phase 8 validation lane. */
export type OnnxBinaryValidatorName = 'onnxruntime-node';

/** Failure categories surfaced by the current Phase 8 binary validation lane. */
export type OnnxBinaryValidationErrorCategory =
  | 'invalid-binary'
  | 'invalid-model'
  | 'runtime-load-failed';

/**
 * Explicit binary compatibility policy resolved from one decoded `ModelProto`.
 *
 * The repo currently preserves a lower-opset contract for the declared same-family
 * subset rather than claiming the full ONNX 1.22.0 `ai.onnx` opset 27 baseline.
 */
export type OnnxBinaryCompatibilityPolicy = {
  irVersion: number;
  standardDomain: 'ai.onnx';
  encodedStandardDomain: '' | 'ai.onnx';
  declaredOpset: number;
  referenceOpset: number;
  usesLowerOpsetContract: boolean;
};

/** Result payload returned by the current Phase 8 binary validation lane. */
export type OnnxBinaryValidationResult = {
  isValid: boolean;
  validator: OnnxBinaryValidatorName;
  compatibilityPolicy?: OnnxBinaryCompatibilityPolicy;
  errorCategory?: OnnxBinaryValidationErrorCategory;
  errorMessage?: string;
};
