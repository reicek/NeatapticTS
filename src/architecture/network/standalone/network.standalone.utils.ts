import type Network from '../../network';
import type Node from '../../node';
import type {
  NetworkStandaloneProps,
  NodeWithIndex,
  StandaloneGenerationContext as GenerationContext,
} from '../network.types';

const OUTPUT_NODE_TYPE = 'output';
const SINGLE_TERM_FALLBACK = '0';
const MASK_MULTIPLIER_IDENTITY = 1;
const INPUT_LOOP_LINE =
  'for(var inputIndex = 0; inputIndex < input.length; inputIndex++) A[inputIndex] = input[inputIndex];';
const ACTIVATION_PRECISION_F32 = 'f32';
const FLOAT32_ARRAY_TYPE = 'Float32Array';
const FLOAT64_ARRAY_TYPE = 'Float64Array';
const FUNCTION_PREFIX = 'function';
const ARROW_TOKEN = '=>';
const FALLBACK_IDENTITY_BODY = '(x){ return x; }';
const COVERAGE_REPLACEMENT = '';
const NO_OUTPUT_NODES_ERROR =
  'Cannot create standalone function: network has no output nodes.';
const INVALID_INPUT_SIZE_ERROR_PREFIX = 'Invalid input size. Expected ';
const INVALID_INPUT_SIZE_ERROR_MIDDLE = ', got ';
const ISTANBUL_IGNORE_BLOCK_REGEX = /\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g;
const COVERAGE_COUNTER_REGEX = /cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g;
const COVERAGE_CALL_REGEX = /cov_[\w$]+\(\)/g;
const SOURCE_MAP_REGEX = /^\s*\/\/ # sourceMappingURL=.*\s*$/gm;
const STRAY_COMMA_OPEN_REGEX = /\(\s*,\s*/g;
const STRAY_COMMA_CLOSE_REGEX = /\s*,\s*\)/g;
const SOLITARY_SEMICOLON_REGEX = /^\s*;\s*$/gm;
const REPEATED_SEMICOLON_REGEX = /;{2,}/g;
const EMPTY_TOKEN_REGEX = /^\s*[,;]?\s*$/gm;

const BUILTIN_ACTIVATION_SNIPPETS: Record<string, string> = {
  logistic: 'function logistic(x){ return 1 / (1 + Math.exp(-x)); }',
  tanh: 'function tanh(x){ return Math.tanh(x); }',
  relu: 'function relu(x){ return x > 0 ? x : 0; }',
  identity: 'function identity(x){ return x; }',
  step: 'function step(x){ return x > 0 ? 1 : 0; }',
  softsign: 'function softsign(x){ return x / (1 + Math.abs(x)); }',
  sinusoid: 'function sinusoid(x){ return Math.sin(x); }',
  gaussian: 'function gaussian(x){ return Math.exp(-Math.pow(x, 2)); }',
  bentIdentity:
    'function bentIdentity(x){ return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x; }',
  bipolar: 'function bipolar(x){ return x > 0 ? 1 : -1; }',
  bipolarSigmoid:
    'function bipolarSigmoid(x){ return 2 / (1 + Math.exp(-x)) - 1; }',
  hardTanh: 'function hardTanh(x){ return Math.max(-1, Math.min(1, x)); }',
  absolute: 'function absolute(x){ return Math.abs(x); }',
  inverse: 'function inverse(x){ return 1 - x; }',
  selu: 'function selu(x){ var a=1.6732632423543772,s=1.0507009873554805; var fx=x>0?x:a*Math.exp(x)-a; return fx*s; }',
  softplus:
    'function softplus(x){ if(x>30)return x; if(x<-30)return Math.exp(x); return Math.max(0,x)+Math.log(1+Math.exp(-Math.abs(x))); }',
  swish: 'function swish(x){ var s=1/(1+Math.exp(-x)); return x*s; }',
  gelu: 'function gelu(x){ var cdf=0.5*(1.0+Math.tanh(Math.sqrt(2.0/Math.PI)*(x+0.044715*Math.pow(x,3)))); return x*cdf; }',
  mish: 'function mish(x){ var sp_x; if(x>30){sp_x=x;}else if(x<-30){sp_x=Math.exp(x);}else{sp_x=Math.log(1+Math.exp(x));} var tanh_sp_x=Math.tanh(sp_x); return x*tanh_sp_x; }',
};

/**
 * Standalone forward pass code generator.
 *
 * Purpose:
 *  Transforms a dynamic Network instance (object graph with Nodes / Connections / gating metadata)
 *  into a self-contained JavaScript function string that, when evaluated, returns an `activate(input)`
 *  function capable of performing forward propagation without the original library runtime.
 *
 * Why generate code?
 *  - Deployment: Embed a compact, dependency‑free inference function in environments where bundling
 *    the full evolutionary framework is unnecessary (e.g. model cards, edge scripts, CI sanity checks).
 *  - Performance: Remove dynamic indirection (property lookups, virtual dispatch) by specializing
 *    the computation graph into straight‑line code and simple loops; JS engines can optimize this.
 *  - Pedagogy: Emitted source is readable—users can inspect how weighted sums + activations compose.
 *
 * Features Supported:
 *  - Standard feed‑forward connections with optional gating (multiplicative modulation).
 *  - Single self-connection per node (handled as recurrent term S[i] * weight before activation).
 *  - Arbitrary activation functions: built‑in ones are emitted via canonical snippets; custom user
 *    functions are stringified and sanitized via stripCoverage(). Arrow or anonymous functions are
 *    normalized into named `function <name>(...)` forms for clarity and stable ordering.
 *
 * Not Supported / Simplifications:
 *  - No dynamic dropout, noise injection, or stochastic depth—those would require runtime randomness.
 *  - Assumes all node indices are stable and sequential (enforced prior to generation).
 *  - Gradient / backprop logic intentionally omitted (forward inference only).
 */

/**
 * Remove instrumentation / coverage artifacts and trivial formatting detritus from function strings.
 * Keeps emitted activation functions as clean as possible for readability and engine optimization.
 */
function stripCoverage(code: string): string {
  let cleanedCode = code;
  cleanedCode = cleanedCode.replace(
    ISTANBUL_IGNORE_BLOCK_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(
    COVERAGE_COUNTER_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(COVERAGE_CALL_REGEX, COVERAGE_REPLACEMENT);
  cleanedCode = cleanedCode.replace(SOURCE_MAP_REGEX, COVERAGE_REPLACEMENT);
  cleanedCode = cleanedCode.replace(STRAY_COMMA_OPEN_REGEX, '( ');
  cleanedCode = cleanedCode.replace(STRAY_COMMA_CLOSE_REGEX, ' )');
  cleanedCode = cleanedCode.trim();
  cleanedCode = cleanedCode.replace(
    SOLITARY_SEMICOLON_REGEX,
    COVERAGE_REPLACEMENT,
  );
  cleanedCode = cleanedCode.replace(REPEATED_SEMICOLON_REGEX, ';');
  cleanedCode = cleanedCode.replace(EMPTY_TOKEN_REGEX, COVERAGE_REPLACEMENT);
  return cleanedCode;
}

/**
 * Generate a standalone JavaScript source string that returns an `activate(input:number[])` function.
 *
 * Implementation Steps:
 *  1. Validate presence of output nodes (must produce something observable).
 *  2. Assign stable sequential indices to nodes (used as array offsets in generated code).
 *  3. Collect initial activation/state values into typed array initializers for warm starting.
 *  4. For each non-input node, build a line computing S[i] (pre-activation sum with bias) and A[i]
 *     (post-activation output). Gating multiplies activation by gate activations; self-connection adds
 *     recurrent term S[i] * weight before activation.
 *  5. De-duplicate activation functions: each unique squash name is emitted once; references become
 *     indices into array F of function references for compactness.
 *  6. Emit an IIFE producing the activate function with internal arrays A (activations) and S (states).
 *
 * @param net Network instance to snapshot.
 * @returns Source string (ES5-compatible) – safe to eval in sandbox to obtain activate function.
 * @throws If network lacks output nodes.
 */
export function generateStandalone(net: Network): string {
  // Step 1: Convert to the internal network view and validate required structure.
  const standaloneProps = asStandaloneProps(net);
  ensureOutputNodesExist(standaloneProps);

  // Step 2: Initialize generation context and seed runtime arrays.
  const generationContext = createGenerationContext(standaloneProps);
  seedNodeIndexesAndState(generationContext);

  // Step 3: Build activate(input) body using tiny orchestration helpers.
  appendInputSeedLine(generationContext);
  appendAllNodeComputationLines(generationContext);
  const outputIndexes = collectOutputIndexes(generationContext);
  appendOutputReturnLine(generationContext, outputIndexes);

  // Step 4: Fold all generated parts into final standalone source.
  return assembleStandaloneSource(generationContext);
}

/**
 * Cast a network instance to the internal standalone generation view.
 *
 * @param net Network instance to cast.
 * @returns Internal network properties used by the standalone generator.
 */
function asStandaloneProps(net: Network): NetworkStandaloneProps {
  return net as unknown as NetworkStandaloneProps;
}

/**
 * Validate that the network has at least one output node.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Void.
 * @throws If no output node exists.
 */
function ensureOutputNodesExist(standaloneProps: NetworkStandaloneProps): void {
  const hasOutputNode = standaloneProps.nodes.some(function hasOutput(
    node: Node,
  ) {
    return node.type === OUTPUT_NODE_TYPE;
  });

  if (!hasOutputNode) {
    throw new Error(NO_OUTPUT_NODES_ERROR);
  }
}

/**
 * Create a fresh generation context used across orchestration steps.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Initialized generation context.
 */
function createGenerationContext(
  standaloneProps: NetworkStandaloneProps,
): GenerationContext {
  return {
    standaloneProps,
    emittedActivationSource: {},
    activationFunctionSources: [],
    activationFunctionIndexMap: {},
    nextActivationFunctionIndex: 0,
    initialActivations: [],
    initialStates: [],
    bodyLines: [],
  };
}

/**
 * Seed index, activation, and state arrays from network nodes.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
function seedNodeIndexesAndState(generationContext: GenerationContext): void {
  const allNodes = generationContext.standaloneProps.nodes;

  for (
    let nodeTraversalIndex = 0;
    nodeTraversalIndex < allNodes.length;
    nodeTraversalIndex++
  ) {
    const currentNode = allNodes[nodeTraversalIndex] as NodeWithIndex;
    currentNode.index = nodeTraversalIndex;
    generationContext.initialActivations.push(currentNode.activation);
    generationContext.initialStates.push(currentNode.state);
  }
}

/**
 * Append the generated input-copy loop to the standalone body.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
function appendInputSeedLine(generationContext: GenerationContext): void {
  generationContext.bodyLines.push(INPUT_LOOP_LINE);
}

/**
 * Append compute lines for all non-input nodes.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
function appendAllNodeComputationLines(
  generationContext: GenerationContext,
): void {
  const firstComputableNodeIndex = generationContext.standaloneProps.input;
  const nodeCount = generationContext.standaloneProps.nodes.length;

  for (
    let nodeTraversalIndex = firstComputableNodeIndex;
    nodeTraversalIndex < nodeCount;
    nodeTraversalIndex++
  ) {
    appendSingleNodeComputationLines(generationContext, nodeTraversalIndex);
  }
}

/**
 * Append state and activation lines for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index currently being emitted.
 * @returns Void.
 */
function appendSingleNodeComputationLines(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
): void {
  const currentNode =
    generationContext.standaloneProps.nodes[nodeTraversalIndex];
  const squashName = resolveSquashName(currentNode, nodeTraversalIndex);
  const activationFunctionIndex = ensureActivationFunctionIndex(
    generationContext,
    squashName,
    currentNode.squash,
    nodeTraversalIndex,
  );
  const sumExpression = buildNodeSumExpression(currentNode, nodeTraversalIndex);
  appendStateLine(
    generationContext,
    nodeTraversalIndex,
    sumExpression,
    currentNode.bias,
  );
  appendActivationLine(
    generationContext,
    nodeTraversalIndex,
    activationFunctionIndex,
    currentNode.mask,
  );
}

/**
 * Resolve a stable activation function name for emission.
 *
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index for anonymous-name fallback.
 * @returns Activation function identifier.
 */
function resolveSquashName(
  currentNode: Node,
  nodeTraversalIndex: number,
): string {
  const explicitName = (currentNode.squash as { name?: string }).name;
  return explicitName ?? `anonymous_squash_${nodeTraversalIndex}`;
}

/**
 * Ensure an activation function is registered and return its table index.
 *
 * @param generationContext Mutable generation context.
 * @param squashName Activation function name.
 * @param squashFunction Activation function implementation.
 * @param nodeTraversalIndex Current node index for fallback naming.
 * @returns Activation function index within generated `F` array.
 */
function ensureActivationFunctionIndex(
  generationContext: GenerationContext,
  squashName: string,
  squashFunction: (inputValue: number, derivate?: boolean) => number,
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
 * Resolve emitted source for built-in or custom activation functions.
 *
 * @param squashName Activation function name.
 * @param squashFunction Activation function implementation.
 * @param nodeTraversalIndex Current node index for fallback flow.
 * @returns Named function source string.
 */
function resolveActivationFunctionSource(
  squashName: string,
  squashFunction: (inputValue: number, derivate?: boolean) => number,
  nodeTraversalIndex: number,
): string {
  const builtinSource = BUILTIN_ACTIVATION_SNIPPETS[squashName];
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
  if (arrowTokenIndex < 0) {
    return `function ${squashName}${FALLBACK_IDENTITY_BODY}`;
  }

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
 * Normalize arrow body into a function-body block.
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
 * Register a function source and allocate its numeric index.
 *
 * @param generationContext Mutable generation context.
 * @param squashName Activation function name.
 * @param functionSource Named function source to store.
 * @returns Void.
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

/**
 * Build the pre-activation sum expression for one node.
 *
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index.
 * @returns String expression used for generated `S[index]` assignment.
 */
function buildNodeSumExpression(
  currentNode: Node,
  nodeTraversalIndex: number,
): string {
  const incomingTerms = collectIncomingTerms(currentNode);
  const selfConnectionTerms = collectSelfConnectionTerms(
    currentNode,
    nodeTraversalIndex,
  );
  const allTerms = mergeTermCollections(incomingTerms, selfConnectionTerms);
  return foldTermsIntoExpression(allTerms);
}

/**
 * Collect feed-forward inbound connection terms for a node.
 *
 * @param currentNode Current node.
 * @returns Weighted term expressions.
 */
function collectIncomingTerms(currentNode: Node): string[] {
  const terms: string[] = [];

  for (const incomingConnection of currentNode.connections.in) {
    const fromNodeIndex = getOptionalNodeIndex(incomingConnection.from);
    if (typeof fromNodeIndex !== 'number') {
      continue;
    }

    let connectionTerm = `A[${fromNodeIndex}] * ${incomingConnection.weight}`;
    connectionTerm = appendGateMultiplier(
      connectionTerm,
      incomingConnection.gater,
    );
    terms.push(connectionTerm);
  }

  return terms;
}

/**
 * Collect recurrent self-connection term for a node when present.
 *
 * @param currentNode Current node.
 * @param nodeTraversalIndex Node index used for self-state reference.
 * @returns Zero or one recurrent term expressions.
 */
function collectSelfConnectionTerms(
  currentNode: Node,
  nodeTraversalIndex: number,
): string[] {
  if (currentNode.connections.self.length === 0) {
    return [];
  }

  const selfConnection = currentNode.connections.self[0];
  let connectionTerm = `S[${nodeTraversalIndex}] * ${selfConnection.weight}`;
  connectionTerm = appendGateMultiplier(connectionTerm, selfConnection.gater);
  return [connectionTerm];
}

/**
 * Append a gate activation multiplier to a connection term when a gate exists.
 *
 * @param connectionTerm Base connection term.
 * @param gateNode Optional gate node.
 * @returns Term with optional gate multiplier.
 */
function appendGateMultiplier(
  connectionTerm: string,
  gateNode: Node | null,
): string {
  const gateNodeIndex = getOptionalNodeIndex(gateNode);
  if (typeof gateNodeIndex !== 'number') {
    return connectionTerm;
  }

  return `${connectionTerm} * A[${gateNodeIndex}]`;
}

/**
 * Resolve optional generated node index from a node reference.
 *
 * @param nodeReference Optional node reference.
 * @returns Node index when available.
 */
function getOptionalNodeIndex(nodeReference: Node | null): number | undefined {
  if (!nodeReference) {
    return undefined;
  }

  const indexedNode = nodeReference as Partial<NodeWithIndex>;
  return indexedNode.index;
}

/**
 * Merge two term lists into a single ordered list.
 *
 * @param firstTerms First term collection.
 * @param secondTerms Second term collection.
 * @returns Combined term collection.
 */
function mergeTermCollections(
  firstTerms: string[],
  secondTerms: string[],
): string[] {
  const mergedTerms = [...firstTerms];
  for (const termValue of secondTerms) {
    mergedTerms.push(termValue);
  }
  return mergedTerms;
}

/**
 * Fold a term collection into a summation expression.
 *
 * @param allTerms Term collection.
 * @returns Summation expression or fallback zero literal.
 */
function foldTermsIntoExpression(allTerms: string[]): string {
  if (allTerms.length === 0) {
    return SINGLE_TERM_FALLBACK;
  }

  return allTerms.join(' + ');
}

/**
 * Append generated state assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param sumExpression Generated sum expression.
 * @param biasValue Node bias.
 * @returns Void.
 */
function appendStateLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  sumExpression: string,
  biasValue: number,
): void {
  generationContext.bodyLines.push(
    `S[${nodeTraversalIndex}] = ${sumExpression} + ${biasValue};`,
  );
}

/**
 * Append generated activation assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param activationFunctionIndex Function table index.
 * @param maskValue Multiplicative mask.
 * @returns Void.
 */
function appendActivationLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  activationFunctionIndex: number,
  maskValue: number,
): void {
  const maskSuffix = buildMaskSuffix(maskValue);
  generationContext.bodyLines.push(
    `A[${nodeTraversalIndex}] = F[${activationFunctionIndex}](S[${nodeTraversalIndex}])${maskSuffix};`,
  );
}

/**
 * Build optional activation mask suffix for generated assignment line.
 *
 * @param maskValue Multiplicative mask value.
 * @returns Empty suffix for identity, otherwise multiplicative fragment.
 */
function buildMaskSuffix(maskValue: number): string {
  if (maskValue === MASK_MULTIPLIER_IDENTITY) {
    return '';
  }

  return ` * ${maskValue}`;
}

/**
 * Collect output node indexes from the output tail segment.
 *
 * @param generationContext Mutable generation context.
 * @returns Output indexes used for result array emission.
 */
function collectOutputIndexes(generationContext: GenerationContext): number[] {
  const outputIndexes: number[] = [];
  const startIndex =
    generationContext.standaloneProps.nodes.length -
    generationContext.standaloneProps.output;
  const endIndex = generationContext.standaloneProps.nodes.length;

  for (
    let nodeTraversalIndex = startIndex;
    nodeTraversalIndex < endIndex;
    nodeTraversalIndex++
  ) {
    const currentNode = generationContext.standaloneProps.nodes[
      nodeTraversalIndex
    ] as NodeWithIndex;
    if (typeof currentNode.index === 'number') {
      outputIndexes.push(currentNode.index);
    }
  }

  return outputIndexes;
}

/**
 * Append generated return line for output activations.
 *
 * @param generationContext Mutable generation context.
 * @param outputIndexes Output node indexes.
 * @returns Void.
 */
function appendOutputReturnLine(
  generationContext: GenerationContext,
  outputIndexes: number[],
): void {
  generationContext.bodyLines.push(
    `return [${formatOutputArrayValues(outputIndexes)}];`,
  );
}

/**
 * Format output activation selectors for generated return expression.
 *
 * @param outputIndexes Output node indexes.
 * @returns Comma-separated `A[index]` selector list.
 */
function formatOutputArrayValues(outputIndexes: number[]): string {
  const outputTerms: string[] = [];
  for (const outputIndex of outputIndexes) {
    outputTerms.push(`A[${outputIndex}]`);
  }
  return outputTerms.join(',');
}

/**
 * Assemble the final standalone IIFE source string.
 *
 * @param generationContext Mutable generation context.
 * @returns Final generated source string.
 */
function assembleStandaloneSource(
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
