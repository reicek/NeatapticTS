import onnxProto from 'onnx-proto';
import { InferenceSession } from 'onnxruntime-node';
import {
  CURRENT_ONNX_REFERENCE_OPSET,
  ONNX_IR_VERSION,
  ONNX_STANDARD_DOMAIN,
  ONNX_STANDARD_DOMAIN_ALIAS,
} from '../export/network.onnx.export-setup.utils';
import type {
  OnnxBinaryCompatibilityPolicy,
  OnnxBinaryValidationResult,
} from './network.onnx.validate.types';

const ONNX_RUNTIME_NODE_VALIDATOR = 'onnxruntime-node' as const;

type OnnxLongLike = number | { toString(): string };

type DecodedBinaryValidationResult =
  | {
      isValid: true;
      decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>;
    }
  | { isValid: false; errorMessage: string };

type CompatibilityPolicyResolutionResult =
  | { isValid: true; compatibilityPolicy: OnnxBinaryCompatibilityPolicy }
  | { isValid: false; errorMessage: string };

/**
 * Validate one binary `ModelProto` payload through the current Phase 8 compliance lane.
 *
 * Validation executes in three ordered stages:
 * 1. Decode and verify the protobuf payload structurally.
 * 2. Confirm the declared headers match the repo's explicit lower-opset policy.
 * 3. Ask ONNX Runtime to accept the model as an external consumer.
 *
 * This is stronger than the repo's internal JSON-first checks, but it is still not a
 * full Phase 9 runtime-parity claim because it does not compare inference outputs.
 *
 * @param binaryModel Binary `ModelProto` payload to validate.
 * @returns Structured validation result for the current external acceptance lane.
 */
export async function validateOnnxBinaryModel(
  binaryModel: Uint8Array,
): Promise<OnnxBinaryValidationResult> {
  const decodedBinaryValidation = decodeAndVerifyBinaryModel(binaryModel);
  if (!decodedBinaryValidation.isValid) {
    return {
      isValid: false,
      validator: ONNX_RUNTIME_NODE_VALIDATOR,
      errorCategory: 'invalid-binary',
      errorMessage: decodedBinaryValidation.errorMessage,
    };
  }

  const compatibilityPolicyResolution = resolveCompatibilityPolicy(
    decodedBinaryValidation.decodedModel,
  );
  if (!compatibilityPolicyResolution.isValid) {
    return {
      isValid: false,
      validator: ONNX_RUNTIME_NODE_VALIDATOR,
      errorCategory: 'invalid-model',
      errorMessage: compatibilityPolicyResolution.errorMessage,
    };
  }

  try {
    await validateExternalRuntimeLoad(binaryModel);

    return {
      isValid: true,
      validator: ONNX_RUNTIME_NODE_VALIDATOR,
      compatibilityPolicy: compatibilityPolicyResolution.compatibilityPolicy,
    };
  } catch (error) {
    return {
      isValid: false,
      validator: ONNX_RUNTIME_NODE_VALIDATOR,
      compatibilityPolicy: compatibilityPolicyResolution.compatibilityPolicy,
      errorCategory: 'runtime-load-failed',
      errorMessage: resolveErrorMessage(error),
    };
  }
}

function decodeAndVerifyBinaryModel(
  binaryModel: Uint8Array,
): DecodedBinaryValidationResult {
  try {
    const decodedModel = onnxProto.onnx.ModelProto.decode(binaryModel);
    const plainDecodedModel = onnxProto.onnx.ModelProto.toObject(decodedModel);
    const verificationError =
      onnxProto.onnx.ModelProto.verify(plainDecodedModel);

    if (verificationError) {
      return {
        isValid: false,
        errorMessage: verificationError,
      };
    }

    return {
      isValid: true,
      decodedModel,
    };
  } catch (error) {
    return {
      isValid: false,
      errorMessage: resolveErrorMessage(error),
    };
  }
}

function resolveCompatibilityPolicy(
  decodedModel: InstanceType<typeof onnxProto.onnx.ModelProto>,
): CompatibilityPolicyResolutionResult {
  const resolvedIrVersion = readLongLikeNumber(decodedModel.irVersion);
  if (resolvedIrVersion < 1) {
    return {
      isValid: false,
      errorMessage: 'Binary ModelProto must declare a positive ir_version.',
    };
  }

  const standardDomainImports = decodedModel.opsetImport.filter(
    (operatorSetImport) =>
      operatorSetImport.domain === ONNX_STANDARD_DOMAIN_ALIAS ||
      operatorSetImport.domain === ONNX_STANDARD_DOMAIN,
  );
  if (standardDomainImports.length !== 1) {
    return {
      isValid: false,
      errorMessage:
        'Binary ModelProto must declare exactly one standard-domain opset import for the current Phase 8 subset.',
    };
  }

  const declaredOpset = readLongLikeNumber(standardDomainImports[0]!.version);
  if (declaredOpset < 1) {
    return {
      isValid: false,
      errorMessage:
        'Binary ModelProto must declare a positive standard-domain opset.',
    };
  }

  return {
    isValid: true,
    compatibilityPolicy: {
      irVersion: resolvedIrVersion,
      standardDomain: ONNX_STANDARD_DOMAIN,
      encodedStandardDomain:
        standardDomainImports[0]!.domain === ONNX_STANDARD_DOMAIN
          ? ONNX_STANDARD_DOMAIN
          : ONNX_STANDARD_DOMAIN_ALIAS,
      declaredOpset,
      referenceOpset: CURRENT_ONNX_REFERENCE_OPSET,
      usesLowerOpsetContract: declaredOpset < CURRENT_ONNX_REFERENCE_OPSET,
    },
  };
}

async function validateExternalRuntimeLoad(
  binaryModel: Uint8Array,
): Promise<void> {
  const inferenceSession = await InferenceSession.create(binaryModel);
  inferenceSession.release();
}

function readLongLikeNumber(value: OnnxLongLike): number {
  return Number.parseInt(value.toString(), 10);
}

function resolveErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
