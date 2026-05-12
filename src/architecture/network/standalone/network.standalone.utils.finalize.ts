import type { StandaloneGenerationContext as GenerationContext } from '../network.types';
import {
  ACTIVATION_PRECISION_F16,
  ACTIVATION_PRECISION_F32,
  FLOAT32_ARRAY_TYPE,
  FLOAT64_ARRAY_TYPE,
  INVALID_INPUT_SIZE_ERROR_MIDDLE,
  INVALID_INPUT_SIZE_ERROR_PREFIX,
  UINT16_ARRAY_TYPE,
} from './network.standalone.utils.types';
import {
  buildStandaloneInputSizeMismatchErrorFactorySource,
  NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME,
} from './network.standalone.errors';

/**
 * Assemble the final standalone IIFE source string.
 *
 * @param generationContext Mutable generation context.
 * @returns Final generated source string.
 */
export function assembleStandaloneSource(
  generationContext: GenerationContext,
): string {
  const activationArrayLiteral = buildActivationArrayLiteral(generationContext);
  const activationBufferType = resolveActivationBufferType(generationContext);
  const precisionHelperSource = buildPrecisionHelperSource(generationContext);
  const workingBufferBootstrapSource =
    buildWorkingBufferBootstrapSource(generationContext);
  const expectedInputSize = generationContext.standaloneProps.input;
  const initialActivationLiteral = buildInitialBufferLiteral(
    generationContext,
    'A',
    generationContext.initialActivations,
  );
  const initialStateLiteral = buildInitialBufferLiteral(
    generationContext,
    'S',
    generationContext.initialStates,
  );

  let generatedSource = '';
  generatedSource += `(function(){\n`;
  generatedSource += `${generationContext.activationFunctionSources.join('\n')}\n`;
  generatedSource += `${buildStandaloneInputSizeMismatchErrorFactorySource()}\n`;
  generatedSource += precisionHelperSource;
  generatedSource += `var F = [${activationArrayLiteral}];\n`;
  generatedSource += `var A = new ${activationBufferType}([${initialActivationLiteral}]);\n`;
  generatedSource += `var S = new ${activationBufferType}([${initialStateLiteral}]);\n`;
  generatedSource += `function activate(input){\n`;
  generatedSource += buildInputGuardLine(expectedInputSize);
  generatedSource += workingBufferBootstrapSource;
  generatedSource += generationContext.bodyLines.join('\n');
  generatedSource += `}\n`;
  generatedSource += `return activate;\n})();`;
  return generatedSource;
}

/**
 * Build deterministic activation function array literal by function index ordering.
 *
 * @param generationContext Mutable generation context.
 * @returns Comma-separated activation function names.
 */
function buildActivationArrayLiteral(
  generationContext: GenerationContext,
): string {
  const activationEntries = Object.entries(
    generationContext.activationFunctionIndexMap,
  );
  activationEntries.sort(function sortByActivationIndex(leftEntry, rightEntry) {
    return leftEntry[1] - rightEntry[1];
  });

  const activationNames: string[] = [];
  for (const activationEntry of activationEntries) {
    activationNames.push(activationEntry[0]);
  }

  return activationNames.join(',');
}

/**
 * Resolve typed-array constructor name based on configured activation precision.
 *
 * @param generationContext Mutable generation context.
 * @returns Constructor name used in generated source.
 */
function resolveActivationBufferType(
  generationContext: GenerationContext,
): string {
  const precision = generationContext.resolvedActivationPrecision;

  if (precision === ACTIVATION_PRECISION_F16) {
    return UINT16_ARRAY_TYPE;
  }

  if (precision === ACTIVATION_PRECISION_F32) {
    return FLOAT32_ARRAY_TYPE;
  }
  return FLOAT64_ARRAY_TYPE;
}

/**
 * Build generated precision helper functions when standalone storage uses float16.
 *
 * @param generationContext Mutable generation context.
 * @returns Helper function source or an empty string for native precision paths.
 */
function buildPrecisionHelperSource(
  generationContext: GenerationContext,
): string {
  if (
    generationContext.resolvedActivationPrecision !== ACTIVATION_PRECISION_F16
  ) {
    return '';
  }

  let helperSource = '';
  helperSource += `function encodeFloat16(value){\n`;
  helperSource += `if(Number.isNaN(value)){return 32256;}\n`;
  helperSource += `if(value===Infinity){return 31744;}\n`;
  helperSource += `if(value===-Infinity){return 64512;}\n`;
  helperSource += `if(value===0){return Object.is(value,-0)?32768:0;}\n`;
  helperSource += `var sign=value<0||Object.is(value,-0)?32768:0;\n`;
  helperSource += `var absoluteValue=Math.abs(value);\n`;
  helperSource += `if(absoluteValue>=65504){return sign|31743;}\n`;
  helperSource += `if(absoluteValue<0.00006103515625){return sign|Math.round(absoluteValue/5.960464477539063e-8);}\n`;
  helperSource += `var exponent=Math.floor(Math.log2(absoluteValue));\n`;
  helperSource += `var mantissa=absoluteValue/Math.pow(2,exponent)-1;\n`;
  helperSource += `var encodedExponent=exponent+15;\n`;
  helperSource += `var encodedMantissa=Math.round(mantissa*1024);\n`;
  helperSource += `if(encodedMantissa===1024){encodedExponent++;encodedMantissa=0;}\n`;
  helperSource += `if(encodedExponent>=31){return sign|31744;}\n`;
  helperSource += `return sign|(encodedExponent<<10)|(encodedMantissa&1023);\n`;
  helperSource += `}\n`;
  helperSource += `function decodeFloat16(bits){\n`;
  helperSource += `var sign=(bits&32768)!==0?-1:1;\n`;
  helperSource += `var exponent=(bits>>10)&31;\n`;
  helperSource += `var mantissa=bits&1023;\n`;
  helperSource += `if(exponent===0){return mantissa===0?sign*0:sign*mantissa*5.960464477539063e-8;}\n`;
  helperSource += `if(exponent===31){return mantissa===0?sign*Infinity:NaN;}\n`;
  helperSource += `return sign*Math.pow(2,exponent-15)*(1+mantissa/1024);\n`;
  helperSource += `}\n`;
  helperSource += `function createWorkingBuffer(storageBuffer){\n`;
  helperSource += `var workingBuffer=new Float32Array(storageBuffer.length);\n`;
  helperSource += `for(var storageIndex=0;storageIndex<storageBuffer.length;storageIndex++){workingBuffer[storageIndex]=decodeFloat16(storageBuffer[storageIndex]);}\n`;
  helperSource += `return workingBuffer;\n`;
  helperSource += `}\n`;
  helperSource += `function finalizeStoredOutput(outputValues,workingActivations,workingStates,activationStorage,stateStorage){\n`;
  helperSource += `for(var storageIndex=0;storageIndex<activationStorage.length;storageIndex++){activationStorage[storageIndex]=encodeFloat16(workingActivations[storageIndex]);stateStorage[storageIndex]=encodeFloat16(workingStates[storageIndex]);}\n`;
  helperSource += `return outputValues;\n`;
  helperSource += `}\n`;

  return helperSource;
}

/**
 * Build working-buffer bootstrap lines for the float16 standalone path.
 *
 * @param generationContext Mutable generation context.
 * @returns Working-buffer setup lines or an empty string for native precision paths.
 */
function buildWorkingBufferBootstrapSource(
  generationContext: GenerationContext,
): string {
  if (
    generationContext.resolvedActivationPrecision !== ACTIVATION_PRECISION_F16
  ) {
    return '';
  }

  return 'var WA = createWorkingBuffer(A);\nvar WS = createWorkingBuffer(S);\n';
}

/**
 * Build the array literal used to seed one generated storage buffer.
 *
 * @param generationContext Mutable generation context.
 * @param values Initial numeric values for the buffer.
 * @returns Literal values matching the generated storage precision.
 */
function buildInitialBufferLiteral(
  generationContext: GenerationContext,
  bufferName: 'A' | 'S',
  values: number[],
): string {
  if (
    generationContext.resolvedActivationPrecision !== ACTIVATION_PRECISION_F16
  ) {
    return values.join(',');
  }

  return values.map((value) => encodeFloat16Value(value).toString()).join(',');
}

/**
 * Encode one JavaScript number into an unsigned float16 storage word.
 *
 * @param value Numeric value being snapshotted into generated float16 storage.
 * @returns Unsigned 16-bit integer containing IEEE 754 binary16 bits.
 */
function encodeFloat16Value(value: number): number {
  if (Number.isNaN(value)) {
    return 32_256;
  }

  if (value === Number.POSITIVE_INFINITY) {
    return 31_744;
  }

  if (value === Number.NEGATIVE_INFINITY) {
    return 64_512;
  }

  if (value === 0) {
    return Object.is(value, -0) ? 32_768 : 0;
  }

  const sign = value < 0 || Object.is(value, -0) ? 32_768 : 0;
  const absoluteValue = Math.abs(value);

  if (absoluteValue >= 65_504) {
    return sign | 31_743;
  }

  if (absoluteValue < 0.00006103515625) {
    return sign | Math.round(absoluteValue / 5.960464477539063e-8);
  }

  let encodedExponent = Math.floor(Math.log2(absoluteValue)) + 15;
  let encodedMantissa = Math.round(
    (absoluteValue / Math.pow(2, encodedExponent - 15) - 1) * 1024,
  );

  if (encodedMantissa === 1024) {
    encodedExponent++;
    encodedMantissa = 0;
  }

  return sign | (encodedExponent << 10) | (encodedMantissa & 1023);
}

/**
 * Build generated input length guard line.
 *
 * @param expectedInputSize Required input vector size.
 * @returns Guard statement line including trailing newline.
 */
function buildInputGuardLine(expectedInputSize: number): string {
  return `if (!input || input.length !== ${expectedInputSize}) { throw ${NETWORK_STANDALONE_INPUT_SIZE_MISMATCH_ERROR_NAME}('${INVALID_INPUT_SIZE_ERROR_PREFIX}${expectedInputSize}${INVALID_INPUT_SIZE_ERROR_MIDDLE}' + (input ? input.length : 'undefined')); }\n`;
}
