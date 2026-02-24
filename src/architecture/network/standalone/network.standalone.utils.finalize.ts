import type { StandaloneGenerationContext as GenerationContext } from '../network.types';
import {
  ACTIVATION_PRECISION_F32,
  FLOAT32_ARRAY_TYPE,
  FLOAT64_ARRAY_TYPE,
  INVALID_INPUT_SIZE_ERROR_MIDDLE,
  INVALID_INPUT_SIZE_ERROR_PREFIX,
} from './network.standalone.utils.types';

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
  const activationArrayType = resolveActivationArrayType(generationContext);
  const expectedInputSize = generationContext.standaloneProps.input;

  let generatedSource = '';
  generatedSource += `(function(){\n`;
  generatedSource += `${generationContext.activationFunctionSources.join('\n')}\n`;
  generatedSource += `var F = [${activationArrayLiteral}];\n`;
  generatedSource += `var A = new ${activationArrayType}([${generationContext.initialActivations.join(
    ',',
  )}]);\n`;
  generatedSource += `var S = new ${activationArrayType}([${generationContext.initialStates.join(
    ',',
  )}]);\n`;
  generatedSource += `function activate(input){\n`;
  generatedSource += buildInputGuardLine(expectedInputSize);
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
function resolveActivationArrayType(
  generationContext: GenerationContext,
): string {
  const precision = generationContext.standaloneProps._activationPrecision;
  if (precision === ACTIVATION_PRECISION_F32) {
    return FLOAT32_ARRAY_TYPE;
  }
  return FLOAT64_ARRAY_TYPE;
}

/**
 * Build generated input length guard line.
 *
 * @param expectedInputSize Required input vector size.
 * @returns Guard statement line including trailing newline.
 */
function buildInputGuardLine(expectedInputSize: number): string {
  return `if (!input || input.length !== ${expectedInputSize}) { throw new Error('${INVALID_INPUT_SIZE_ERROR_PREFIX}${expectedInputSize}${INVALID_INPUT_SIZE_ERROR_MIDDLE}' + (input ? input.length : 'undefined')); }\n`;
}
