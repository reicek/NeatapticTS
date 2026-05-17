import type {
  OnnxAttribute,
  OnnxDimension,
  OnnxGraph,
  OnnxMetadataProperty,
  OnnxModel,
  OnnxNode,
  OnnxTensor,
  OnnxTensorType,
  OnnxValueInfo,
} from './network.onnx.schema.types';

const PROTOBUF_VARINT_WIRE_TYPE = 0;
const PROTOBUF_FIXED32_WIRE_TYPE = 5;
const PROTOBUF_LENGTH_DELIMITED_WIRE_TYPE = 2;
const FLOAT32_BYTE_WIDTH = 4;
const textEncoder =
  typeof TextEncoder === 'undefined' ? undefined : new TextEncoder();

const ATTRIBUTE_TYPE_CODES = {
  FLOAT: 1,
  INT: 2,
  STRING: 3,
  TENSOR: 4,
  GRAPH: 5,
  FLOATS: 6,
  INTS: 7,
  STRINGS: 8,
} as const;

/**
 * Serialize an ONNX-like model into protobuf ModelProto bytes.
 *
 * This helper turns the shared exporter model view into the deterministic
 * binary `ModelProto` surface that Phase 8 and Phase 9 treat as the primary
 * runtime-validated artifact for the approved subset.
 * It intentionally uses ONNX's typed tensor storage fields such as
 * `float_data`, `int32_data`, and `int64_data` rather than widening into
 * `raw_data` or `external_data` yet.
 *
 * @param onnxModel ONNX-like model payload.
 * @returns Binary protobuf ModelProto bytes.
 */
export function serializeOnnxModelToBinary(onnxModel: OnnxModel): Uint8Array {
  return concatByteChunks(collectModelFieldChunks(onnxModel));
}

function collectModelFieldChunks(onnxModel: OnnxModel): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  if (onnxModel.ir_version !== undefined) {
    fieldChunks.push(encodeInt64Field(1, onnxModel.ir_version));
  }

  if (onnxModel.producer_name) {
    fieldChunks.push(encodeStringField(2, onnxModel.producer_name));
  }

  if (onnxModel.producer_version) {
    fieldChunks.push(encodeStringField(3, onnxModel.producer_version));
  }

  if (onnxModel.doc_string) {
    fieldChunks.push(encodeStringField(6, onnxModel.doc_string));
  }

  fieldChunks.push(
    encodeMessageField(7, collectGraphFieldChunks(onnxModel.graph)),
  );

  onnxModel.opset_import?.forEach((operatorSetImport) => {
    fieldChunks.push(
      encodeMessageField(
        8,
        collectOperatorSetFieldChunks(operatorSetImport.domain, operatorSetImport.version),
      ),
    );
  });

  onnxModel.metadata_props?.forEach((metadataProperty) => {
    fieldChunks.push(
      encodeMessageField(14, collectMetadataPropertyFieldChunks(metadataProperty)),
    );
  });

  return fieldChunks;
}

function collectGraphFieldChunks(onnxGraph: OnnxGraph): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  onnxGraph.node.forEach((graphNode) => {
    fieldChunks.push(encodeMessageField(1, collectNodeFieldChunks(graphNode)));
  });

  onnxGraph.initializer.forEach((initializerTensor) => {
    fieldChunks.push(
      encodeMessageField(5, collectTensorFieldChunks(initializerTensor)),
    );
  });

  onnxGraph.inputs.forEach((graphInput) => {
    fieldChunks.push(encodeMessageField(11, collectValueInfoFieldChunks(graphInput)));
  });

  onnxGraph.outputs.forEach((graphOutput) => {
    fieldChunks.push(encodeMessageField(12, collectValueInfoFieldChunks(graphOutput)));
  });

  return fieldChunks;
}

function collectValueInfoFieldChunks(onnxValueInfo: OnnxValueInfo): Uint8Array[] {
  return [
    encodeStringField(1, onnxValueInfo.name),
    encodeMessageField(2, collectTypeFieldChunks(onnxValueInfo.type.tensor_type)),
  ];
}

function collectTypeFieldChunks(onnxTensorType: OnnxTensorType): Uint8Array[] {
  return [
    encodeMessageField(1, collectTensorTypeFieldChunks(onnxTensorType)),
  ];
}

function collectTensorTypeFieldChunks(
  onnxTensorType: OnnxTensorType,
): Uint8Array[] {
  return [
    encodeInt32Field(1, onnxTensorType.elem_type),
    encodeMessageField(2, collectTensorShapeFieldChunks(onnxTensorType.shape.dim)),
  ];
}

function collectTensorShapeFieldChunks(
  onnxDimensions: OnnxDimension[],
): Uint8Array[] {
  return onnxDimensions.map((onnxDimension) =>
    encodeMessageField(1, collectDimensionFieldChunks(onnxDimension)),
  );
}

function collectDimensionFieldChunks(onnxDimension: OnnxDimension): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  if (onnxDimension.dim_value !== undefined) {
    fieldChunks.push(encodeInt64Field(1, onnxDimension.dim_value));
  }

  if (onnxDimension.dim_param !== undefined) {
    fieldChunks.push(encodeStringField(2, onnxDimension.dim_param));
  }

  return fieldChunks;
}

function collectTensorFieldChunks(onnxTensor: OnnxTensor): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  if (onnxTensor.dims.length > 0) {
    fieldChunks.push(encodePackedInt64Field(1, onnxTensor.dims));
  }

  fieldChunks.push(encodeInt32Field(2, onnxTensor.data_type));

  if (onnxTensor.float_data.length > 0) {
    fieldChunks.push(encodePackedFloatField(4, onnxTensor.float_data));
  }

  if (onnxTensor.int32_data && onnxTensor.int32_data.length > 0) {
    fieldChunks.push(encodePackedInt32Field(5, onnxTensor.int32_data));
  }

  if (onnxTensor.int64_data && onnxTensor.int64_data.length > 0) {
    fieldChunks.push(encodePackedInt64Field(7, onnxTensor.int64_data));
  }

  fieldChunks.push(encodeStringField(8, onnxTensor.name));

  return fieldChunks;
}

function collectNodeFieldChunks(onnxNode: OnnxNode): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  onnxNode.input.forEach((inputName) => {
    fieldChunks.push(encodeStringField(1, inputName));
  });

  onnxNode.output.forEach((outputName) => {
    fieldChunks.push(encodeStringField(2, outputName));
  });

  if (onnxNode.name) {
    fieldChunks.push(encodeStringField(3, onnxNode.name));
  }

  fieldChunks.push(encodeStringField(4, onnxNode.op_type));

  onnxNode.attributes?.forEach((attributeEntry) => {
    fieldChunks.push(
      encodeMessageField(5, collectAttributeFieldChunks(attributeEntry)),
    );
  });

  return fieldChunks;
}

function collectAttributeFieldChunks(onnxAttribute: OnnxAttribute): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [encodeStringField(1, onnxAttribute.name)];
  const resolvedTypeCode = resolveAttributeTypeCode(onnxAttribute);

  if (onnxAttribute.f !== undefined) {
    fieldChunks.push(encodeFixed32Field(2, onnxAttribute.f));
  }

  if (onnxAttribute.i !== undefined) {
    fieldChunks.push(encodeInt64Field(3, onnxAttribute.i));
  }

  if (onnxAttribute.s !== undefined) {
    fieldChunks.push(encodeBytesField(4, encodeUtf8(onnxAttribute.s)));
  }

  if (onnxAttribute.t) {
    fieldChunks.push(encodeMessageField(5, collectTensorFieldChunks(onnxAttribute.t)));
  }

  if (onnxAttribute.g) {
    fieldChunks.push(encodeMessageField(6, collectGraphFieldChunks(onnxAttribute.g)));
  }

  if (onnxAttribute.floats && onnxAttribute.floats.length > 0) {
    fieldChunks.push(encodePackedFloatField(7, onnxAttribute.floats));
  }

  if (onnxAttribute.ints && onnxAttribute.ints.length > 0) {
    fieldChunks.push(encodePackedInt64Field(8, onnxAttribute.ints));
  }

  if (onnxAttribute.strings && onnxAttribute.strings.length > 0) {
    onnxAttribute.strings.forEach((stringValue) => {
      fieldChunks.push(encodeBytesField(9, encodeUtf8(stringValue)));
    });
  }

  if (resolvedTypeCode !== undefined) {
    fieldChunks.push(encodeInt32Field(20, resolvedTypeCode));
  }

  return fieldChunks;
}

function collectOperatorSetFieldChunks(
  domain: string,
  version: number,
): Uint8Array[] {
  const fieldChunks: Uint8Array[] = [];

  if (domain.length > 0) {
    fieldChunks.push(encodeStringField(1, domain));
  }

  fieldChunks.push(encodeInt64Field(2, version));
  return fieldChunks;
}

function collectMetadataPropertyFieldChunks(
  metadataProperty: OnnxMetadataProperty,
): Uint8Array[] {
  return [
    encodeStringField(1, metadataProperty.key),
    encodeStringField(2, metadataProperty.value),
  ];
}

function resolveAttributeTypeCode(
  onnxAttribute: OnnxAttribute,
): number | undefined {
  if (onnxAttribute.type) {
    return ATTRIBUTE_TYPE_CODES[
      onnxAttribute.type as keyof typeof ATTRIBUTE_TYPE_CODES
    ];
  }

  if (onnxAttribute.f !== undefined) {
    return ATTRIBUTE_TYPE_CODES.FLOAT;
  }

  if (onnxAttribute.i !== undefined) {
    return ATTRIBUTE_TYPE_CODES.INT;
  }

  if (onnxAttribute.s !== undefined) {
    return ATTRIBUTE_TYPE_CODES.STRING;
  }

  if (onnxAttribute.t) {
    return ATTRIBUTE_TYPE_CODES.TENSOR;
  }

  if (onnxAttribute.g) {
    return ATTRIBUTE_TYPE_CODES.GRAPH;
  }

  if (onnxAttribute.floats && onnxAttribute.floats.length > 0) {
    return ATTRIBUTE_TYPE_CODES.FLOATS;
  }

  if (onnxAttribute.ints && onnxAttribute.ints.length > 0) {
    return ATTRIBUTE_TYPE_CODES.INTS;
  }

  if (onnxAttribute.strings && onnxAttribute.strings.length > 0) {
    return ATTRIBUTE_TYPE_CODES.STRINGS;
  }

  return undefined;
}

function encodeInt32Field(fieldNumber: number, value: number): Uint8Array {
  return encodeVarintField(fieldNumber, value);
}

function encodeInt64Field(fieldNumber: number, value: number): Uint8Array {
  return encodeVarintField(fieldNumber, value);
}

function encodeVarintField(fieldNumber: number, value: number): Uint8Array {
  return concatByteChunks([
    encodeFieldTag(fieldNumber, PROTOBUF_VARINT_WIRE_TYPE),
    encodeSignedVarint(value),
  ]);
}

function encodeFixed32Field(fieldNumber: number, value: number): Uint8Array {
  return concatByteChunks([
    encodeFieldTag(fieldNumber, PROTOBUF_FIXED32_WIRE_TYPE),
    encodeFloat32LittleEndian(value),
  ]);
}

function encodeStringField(fieldNumber: number, value: string): Uint8Array {
  return encodeBytesField(fieldNumber, encodeUtf8(value));
}

function encodeBytesField(fieldNumber: number, value: Uint8Array): Uint8Array {
  return concatByteChunks([
    encodeFieldTag(fieldNumber, PROTOBUF_LENGTH_DELIMITED_WIRE_TYPE),
    encodeUnsignedVarint(value.length),
    value,
  ]);
}

function encodeMessageField(
  fieldNumber: number,
  messageFieldChunks: Uint8Array[],
): Uint8Array {
  return encodeBytesField(fieldNumber, concatByteChunks(messageFieldChunks));
}

function encodePackedFloatField(
  fieldNumber: number,
  values: number[],
): Uint8Array {
  const packedBytes = new Uint8Array(values.length * FLOAT32_BYTE_WIDTH);
  const packedView = new DataView(packedBytes.buffer);

  values.forEach((value, valueIndex) => {
    packedView.setFloat32(valueIndex * FLOAT32_BYTE_WIDTH, value, true);
  });

  return encodeBytesField(fieldNumber, packedBytes);
}

function encodePackedInt32Field(
  fieldNumber: number,
  values: number[],
): Uint8Array {
  return encodeBytesField(
    fieldNumber,
    concatByteChunks(values.map((value) => encodeSignedVarint(value))),
  );
}

function encodePackedInt64Field(
  fieldNumber: number,
  values: number[],
): Uint8Array {
  return encodeBytesField(
    fieldNumber,
    concatByteChunks(values.map((value) => encodeSignedVarint(value))),
  );
}

function encodeFieldTag(fieldNumber: number, wireType: number): Uint8Array {
  return encodeUnsignedVarint((fieldNumber << 3) | wireType);
}

function encodeSignedVarint(value: number): Uint8Array {
  return encodeUnsignedVarint(BigInt.asUintN(64, BigInt(value)));
}

function encodeUnsignedVarint(value: number | bigint): Uint8Array {
  let remainingValue = typeof value === 'bigint' ? value : BigInt(value);
  const encodedBytes: number[] = [];

  while (remainingValue > 0x7fn) {
    encodedBytes.push(Number((remainingValue & 0x7fn) | 0x80n));
    remainingValue >>= 7n;
  }

  encodedBytes.push(Number(remainingValue));
  return Uint8Array.from(encodedBytes);
}

function encodeFloat32LittleEndian(value: number): Uint8Array {
  const encodedBytes = new Uint8Array(FLOAT32_BYTE_WIDTH);
  new DataView(encodedBytes.buffer).setFloat32(0, value, true);
  return encodedBytes;
}

function encodeUtf8(value: string): Uint8Array {
  if (textEncoder) {
    return textEncoder.encode(value);
  }

  const encodedValue = encodeURIComponent(value).replace(
    /%([0-9A-F]{2})/g,
    (_match, encodedByte: string) =>
      String.fromCharCode(Number.parseInt(encodedByte, 16)),
  );

  return Uint8Array.from(encodedValue, (character) => character.charCodeAt(0));
}

function concatByteChunks(byteChunks: Uint8Array[]): Uint8Array {
  const totalLength = byteChunks.reduce(
    (currentLength, byteChunk) => currentLength + byteChunk.length,
    0,
  );
  const mergedBytes = new Uint8Array(totalLength);
  let writeOffset = 0;

  byteChunks.forEach((byteChunk) => {
    mergedBytes.set(byteChunk, writeOffset);
    writeOffset += byteChunk.length;
  });

  return mergedBytes;
}