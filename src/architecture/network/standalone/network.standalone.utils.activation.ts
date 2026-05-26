import type Node from '../../node';
import type { StandaloneGenerationContext as GenerationContext } from '../network.types';
import { stripCoverage } from './network.standalone.utils.coverage';
import {
  ARROW_TOKEN,
  BUILTIN_ACTIVATION_SNIPPETS,
  FALLBACK_IDENTITY_BODY,
  FUNCTION_PREFIX,
} from './network.standalone.utils.types';
import type { StandaloneSquashFunction } from './network.standalone.utils.types';

/**
 * Resolve a stable activation function identifier for standalone code emission.
 *
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index for anonymous-name fallback.
 * @returns Activation function identifier.
 */
export function resolveSquashName(
  currentNode: Node,
  nodeTraversalIndex: number,
): string {
  const explicitName = (currentNode.squash as { name?: string }).name;
  return explicitName ?? `anonymous_squash_${nodeTraversalIndex}`;
}

/**
 * Register an activation implementation once and return its stable index in the generated activation table.
 *
 * @param generationContext Mutable generation context.
 * @param squashName Activation function name.
 * @param squashFunction Activation function implementation.
 * @param nodeTraversalIndex Current node index for fallback naming.
 * @returns Activation function index within generated `F` array.
 */
export function ensureActivationFunctionIndex(
  generationContext: GenerationContext,
  squashName: string,
  squashFunction: StandaloneSquashFunction,
  nodeTraversalIndex: number,
): number {
  const existingIndex =
    generationContext.activationFunctionIndexMap[squashName];
  if (typeof existingIndex === 'number') {
    return existingIndex;
  }

  const functionSource = resolveActivationFunctionSource(
    squashName,
    squashFunction,
    nodeTraversalIndex,
  );
  registerActivationFunction(generationContext, squashName, functionSource);

  return generationContext.activationFunctionIndexMap[squashName];
}

/**
 * Resolve standalone-ready activation source by preferring built-in snippets and normalizing custom bodies.
 *
 * @param squashName Activation function name.
 * @param squashFunction Activation function implementation.
 * @param nodeTraversalIndex Current node index for fallback flow.
 * @returns Named function source string.
 */
function resolveActivationFunctionSource(
  squashName: string,
  squashFunction: StandaloneSquashFunction,
  nodeTraversalIndex: number,
): string {
  const builtinSource = resolveBuiltinActivationSource(squashName);
  if (typeof builtinSource === 'string') {
    return normalizeBuiltinSource(builtinSource, squashName);
  }

  return normalizeCustomSource(
    squashFunction.toString(),
    squashName,
    nodeTraversalIndex,
  );
}

/**
 * Resolve built-in activation snippets for both canonical and exported helper names.
 *
 * @param squashName Activation function name.
 * @returns Built-in activation source when known.
 */
function resolveBuiltinActivationSource(
  squashName: string,
): string | undefined {
  const directBuiltinSource = BUILTIN_ACTIVATION_SNIPPETS[squashName];
  if (typeof directBuiltinSource === 'string') {
    return directBuiltinSource;
  }

  const builtinAliasSuffix = 'Activation';
  if (!squashName.endsWith(builtinAliasSuffix)) {
    return undefined;
  }

  const canonicalBuiltinName = squashName.slice(0, -builtinAliasSuffix.length);
  return BUILTIN_ACTIVATION_SNIPPETS[canonicalBuiltinName];
}

/**
 * Normalize built-in activation source to a named function and strip coverage artifacts.
 *
 * @param sourceCode Built-in function source.
 * @param squashName Required function name.
 * @returns Cleaned named function source.
 */
function normalizeBuiltinSource(
  sourceCode: string,
  squashName: string,
): string {
  const normalizedSource = ensureNamedFunctionSource(sourceCode, squashName);
  return stripCoverage(normalizedSource);
}

/**
 * Normalize custom activation source with function/arrow/fallback handling.
 *
 * @param sourceCode Custom function source.
 * @param squashName Required function name.
 * @param nodeTraversalIndex Current node index for traversal-context parity.
 * @returns Cleaned named function source.
 */
function normalizeCustomSource(
  sourceCode: string,
  squashName: string,
  nodeTraversalIndex: number,
): string {
  const cleanedSource = stripCoverage(sourceCode);

  if (cleanedSource.startsWith(FUNCTION_PREFIX)) {
    return ensureNamedFunctionSource(cleanedSource, squashName);
  }

  if (cleanedSource.includes(ARROW_TOKEN)) {
    return convertArrowToNamedFunction(cleanedSource, squashName);
  }

  void nodeTraversalIndex;
  return `function ${squashName}${FALLBACK_IDENTITY_BODY}`;
}

/**
 * Ensure generated function source starts with the required named signature.
 *
 * @param sourceCode Function source.
 * @param squashName Required function name.
 * @returns Named function source.
 */
function ensureNamedFunctionSource(
  sourceCode: string,
  squashName: string,
): string {
  const expectedPrefix = `${FUNCTION_PREFIX} ${squashName}`;
  if (sourceCode.startsWith(expectedPrefix)) {
    return sourceCode;
  }

  const parameterStartIndex = sourceCode.indexOf('(');
  if (parameterStartIndex < 0) {
    return `function ${squashName}${FALLBACK_IDENTITY_BODY}`;
  }

  return `function ${squashName}${sourceCode.substring(parameterStartIndex)}`;
}

/**
 * Convert an arrow-function source string into a named function declaration source.
 *
 * @param sourceCode Arrow-function source.
 * @param squashName Required function name.
 * @returns Named function source.
 */
function convertArrowToNamedFunction(
  sourceCode: string,
  squashName: string,
): string {
  const arrowTokenIndex = sourceCode.indexOf(ARROW_TOKEN);
  const parameterSegment = sourceCode.substring(0, arrowTokenIndex).trim();
  const bodySegment = sourceCode
    .substring(arrowTokenIndex + ARROW_TOKEN.length)
    .trim();
  const normalizedParameters = normalizeArrowParameters(parameterSegment);
  const normalizedBody = normalizeArrowBody(bodySegment);
  return `function ${squashName}(${normalizedParameters})${normalizedBody}`;
}

/**
 * Normalize arrow parameter segment into comma-separated parameter list content.
 *
 * @param parameterSegment Raw arrow parameter segment.
 * @returns Parameter list body (without surrounding parentheses).
 */
function normalizeArrowParameters(parameterSegment: string): string {
  const startsWithParentheses = parameterSegment.startsWith('(');
  const endsWithParentheses = parameterSegment.endsWith(')');

  if (startsWithParentheses && endsWithParentheses) {
    return parameterSegment.substring(1, parameterSegment.length - 1).trim();
  }

  return parameterSegment;
}

/**
 * Normalize an arrow body to a function-body block.
 *
 * Preserves block bodies and wraps expression bodies with an explicit return
 * so generated standalone activation functions remain syntactically stable.
 *
 * @param bodySegment Raw arrow body segment.
 * @returns Function body block string.
 */
function normalizeArrowBody(bodySegment: string): string {
  if (bodySegment.startsWith('{')) {
    return bodySegment;
  }

  return `{ return ${bodySegment}; }`;
}

/**
 * Persist a normalized activation source and bind its name to the next generated activation index.
 *
 * @param generationContext Mutable generation context.
 * @param squashName Activation function name.
 * @param functionSource Named function source to store.
 */
function registerActivationFunction(
  generationContext: GenerationContext,
  squashName: string,
  functionSource: string,
): void {
  generationContext.emittedActivationSource[squashName] = functionSource;
  generationContext.activationFunctionSources.push(functionSource);
  generationContext.activationFunctionIndexMap[squashName] =
    generationContext.nextActivationFunctionIndex;
  generationContext.nextActivationFunctionIndex += 1;
}
