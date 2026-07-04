# architecture/network/onnx/validate

External validator identifier for the current Phase 8 validation lane.

## architecture/network/onnx/validate/network.onnx.validate.types.ts

### OnnxBinaryCompatibilityPolicy

Explicit binary compatibility policy resolved from one decoded `ModelProto`.

The repo currently preserves a lower-opset contract for the declared same-family
subset rather than claiming the full ONNX 1.22.0 `ai.onnx` opset 27 baseline.

### OnnxBinaryValidationErrorCategory

Failure categories surfaced by the current Phase 8 binary validation lane.

### OnnxBinaryValidationResult

Result payload returned by the current Phase 8 binary validation lane.

### OnnxBinaryValidatorName

External validator identifier for the current Phase 8 validation lane.

## architecture/network/onnx/validate/network.onnx.validate.ts

### validateOnnxBinaryModel

```ts
validateOnnxBinaryModel(
  binaryModel: Uint8Array<ArrayBufferLike>,
): Promise<OnnxBinaryValidationResult>
```

Validate one binary `ModelProto` payload through the current Phase 8 compliance lane.

Validation executes in three ordered stages:
1. Decode and verify the protobuf payload structurally.
2. Confirm the declared headers match the repo's explicit lower-opset policy.
3. Ask ONNX Runtime to accept the model as an external consumer.

This is stronger than the repo's internal JSON-first checks, but it is still not a
full Phase 9 runtime-parity claim because it does not compare inference outputs.

Parameters:
- `binaryModel` - Binary `ModelProto` payload to validate.

Returns: Structured validation result for the current external acceptance lane.
