# architecture/network/standalone

Output node discriminator used for standalone precondition checks.

## architecture/network/standalone/network.standalone.utils.types.ts

### OUTPUT_NODE_TYPE

Output node discriminator used for standalone precondition checks.

### SINGLE_TERM_FALLBACK

Fallback literal used when a node has no incoming terms.

### MASK_MULTIPLIER_IDENTITY

Multiplicative identity used to omit redundant mask expressions.

### INPUT_LOOP_LINE

Generated source line for copying external inputs into activation buffer.

### ACTIVATION_PRECISION_F32

Precision token selecting Float32 activation/state buffers.

### FLOAT32_ARRAY_TYPE

Typed-array constructor names used in generated source.

### FLOAT64_ARRAY_TYPE

Typed-array constructor names used in generated source.

### FUNCTION_PREFIX

Prefix token used when normalizing function sources.

### ARROW_TOKEN

Arrow token used during function-source normalization.

### FALLBACK_IDENTITY_BODY

Identity-function fallback body for invalid custom squash sources.

### COVERAGE_REPLACEMENT

Empty replacement used while stripping coverage artifacts.

### NO_OUTPUT_NODES_ERROR

Error message when attempting standalone generation without outputs.

### INVALID_INPUT_SIZE_ERROR_PREFIX

Input-size validation message fragments for generated activate guards.

### INVALID_INPUT_SIZE_ERROR_MIDDLE

Input-size validation message fragments for generated activate guards.

### ISTANBUL_IGNORE_BLOCK_REGEX

Regex stripping Istanbul ignore blocks from stringified functions.

### COVERAGE_COUNTER_REGEX

Regex stripping Istanbul counters from stringified functions.

### COVERAGE_CALL_REGEX

Regex stripping Istanbul function invocations from source snippets.

### SOURCE_MAP_REGEX

Regex stripping sourceMappingURL comments from generated snippets.

### STRAY_COMMA_OPEN_REGEX

Regex normalizing stray commas near opening parentheses.

### STRAY_COMMA_CLOSE_REGEX

Regex normalizing stray commas near closing parentheses.

### SOLITARY_SEMICOLON_REGEX

Regex removing solitary semicolon lines created by instrumentation.

### REPEATED_SEMICOLON_REGEX

Regex collapsing repeated semicolons.

### EMPTY_TOKEN_REGEX

Regex removing empty punctuation-only token lines.

### BUILTIN_ACTIVATION_SNIPPETS

Built-in activation snippets emitted as named JavaScript function declarations.

Values are intentionally compact so emitted standalone source remains deterministic and small.

### StandaloneSquashFunction

```ts
StandaloneSquashFunction(
  inputValue: number,
  derivate: boolean | undefined,
): number
```

Activation function shape used by standalone source generation helpers.

## architecture/network/standalone/network.standalone.utils.ts

Standalone forward pass code generator.

Purpose:
 Transforms a dynamic Network instance (object graph with Nodes / Connections / gating metadata)
 into a self-contained JavaScript function string that, when evaluated, returns an `activate(input)`
 function capable of performing forward propagation without the original library runtime.

Why generate code?
 - Deployment: Embed a compact, dependency‑free inference function in environments where bundling
   the full evolutionary framework is unnecessary (e.g. model cards, edge scripts, CI sanity checks).
 - Performance: Remove dynamic indirection (property lookups, virtual dispatch) by specializing
   the computation graph into straight‑line code and simple loops; JS engines can optimize this.
 - Readability: Emitted source is human-readable so users can inspect weighted sums and activations.

Features Supported:
 - Standard feed‑forward connections with optional gating (multiplicative modulation).
 - Single self-connection per node (handled as recurrent term S[i] * weight before activation).
 - Arbitrary activation functions: built‑in ones are emitted via canonical snippets; custom user
   functions are stringified and sanitized via stripCoverage(). Arrow or anonymous functions are
   normalized into named `function <name>(...)` forms for clarity and stable ordering.

Not Supported / Simplifications:
 - No dynamic dropout, noise injection, or stochastic depth—those would require runtime randomness.
 - Assumes all node indices are stable and sequential (enforced prior to generation).
 - Gradient / backprop logic intentionally omitted (forward inference only).

### generateStandalone

```ts
generateStandalone(
  net: default,
): string
```

Generate a standalone JavaScript source string that returns an `activate(input:number[])` function.

Implementation Steps:
 1. Validate presence of output nodes (must produce something observable).
 2. Assign stable sequential indices to nodes (used as array offsets in generated code).
 3. Collect initial activation/state values into typed array initializers for warm starting.
 4. For each non-input node, build a line computing S[i] (pre-activation sum with bias) and A[i]
    (post-activation output). Gating multiplies activation by gate activations; self-connection adds
    recurrent term S[i] * weight before activation.
 5. De-duplicate activation functions: each unique squash name is emitted once; references become
    indices into array F of function references for compactness.
 6. Emit an IIFE producing the activate function with internal arrays A (activations) and S (states).

Parameters:
- `net` - Network instance to snapshot.

Returns: Source string (ES5-compatible) – safe to eval in sandbox to obtain activate function.

## architecture/network/standalone/network.standalone.utils.loop.ts

### appendInputSeedLine

```ts
appendInputSeedLine(
  generationContext: StandaloneGenerationContext,
): void
```

Append the generated input-copy loop to the standalone body.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

### appendAllNodeComputationLines

```ts
appendAllNodeComputationLines(
  generationContext: StandaloneGenerationContext,
): void
```

Append compute lines for all non-input nodes.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

### appendOutputReturnLine

```ts
appendOutputReturnLine(
  generationContext: StandaloneGenerationContext,
  outputIndexes: number[],
): void
```

Append generated return line for output activations.

Parameters:
- `generationContext` - Mutable generation context.
- `outputIndexes` - Output node indexes.

Returns: Void.

### appendSingleNodeComputationLines

```ts
appendSingleNodeComputationLines(
  generationContext: StandaloneGenerationContext,
  nodeTraversalIndex: number,
): void
```

Append state and activation lines for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index currently being emitted.

Returns: Void.

### appendStateLine

```ts
appendStateLine(
  generationContext: StandaloneGenerationContext,
  nodeTraversalIndex: number,
  sumExpression: string,
  biasValue: number,
): void
```

Append generated state assignment line for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index.
- `sumExpression` - Generated sum expression.
- `biasValue` - Node bias.

Returns: Void.

### appendActivationLine

```ts
appendActivationLine(
  generationContext: StandaloneGenerationContext,
  nodeTraversalIndex: number,
  activationFunctionIndex: number,
  maskValue: number,
): void
```

Append generated activation assignment line for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index.
- `activationFunctionIndex` - Function table index.
- `maskValue` - Multiplicative mask.

Returns: Void.

### buildMaskSuffix

```ts
buildMaskSuffix(
  maskValue: number,
): string
```

Build optional activation mask suffix for generated assignment line.

Parameters:
- `maskValue` - Multiplicative mask value.

Returns: Empty suffix for identity, otherwise multiplicative fragment.

## architecture/network/standalone/network.standalone.utils.graph.ts

### buildNodeSumExpression

```ts
buildNodeSumExpression(
  currentNode: default,
  nodeTraversalIndex: number,
): string
```

Build the pre-activation sum expression for one node.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index.

Returns: String expression used for generated `S[index]` assignment.

### collectOutputIndexes

```ts
collectOutputIndexes(
  generationContext: StandaloneGenerationContext,
): number[]
```

Collect output node indexes from the output tail segment.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Output indexes used for result array emission.

### formatOutputArrayValues

```ts
formatOutputArrayValues(
  outputIndexes: number[],
): string
```

Format output activation selectors for generated return expression.

Parameters:
- `outputIndexes` - Output node indexes.

Returns: Comma-separated `A[index]` selector list.

### collectIncomingTerms

```ts
collectIncomingTerms(
  currentNode: default,
): string[]
```

Collect feed-forward inbound connection terms for a node.

Parameters:
- `currentNode` - Current node.

Returns: Weighted term expressions.

### collectSelfConnectionTerms

```ts
collectSelfConnectionTerms(
  currentNode: default,
  nodeTraversalIndex: number,
): string[]
```

Collect recurrent self-connection term for a node when present.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index used for self-state reference.

Returns: Zero or one recurrent term expressions.

### appendGateMultiplier

```ts
appendGateMultiplier(
  connectionTerm: string,
  gateNode: default | null,
): string
```

Append a gate activation multiplier to a connection term when a gate exists.

Parameters:
- `connectionTerm` - Base connection term.
- `gateNode` - Optional gate node.

Returns: Term with optional gate multiplier.

### getOptionalNodeIndex

```ts
getOptionalNodeIndex(
  nodeReference: default | null,
): number | undefined
```

Resolve optional generated node index from a node reference.

Parameters:
- `nodeReference` - Optional node reference.

Returns: Node index when available.

### mergeTermCollections

```ts
mergeTermCollections(
  firstTerms: string[],
  secondTerms: string[],
): string[]
```

Merge two term lists into a single ordered list.

Parameters:
- `firstTerms` - First term collection.
- `secondTerms` - Second term collection.

Returns: Combined term collection.

### foldTermsIntoExpression

```ts
foldTermsIntoExpression(
  allTerms: string[],
): string
```

Fold a term collection into a summation expression.

Parameters:
- `allTerms` - Term collection.

Returns: Summation expression or fallback zero literal.

## architecture/network/standalone/network.standalone.utils.setup.ts

### asStandaloneProps

```ts
asStandaloneProps(
  net: default,
): NetworkStandaloneProps
```

Cast a network instance to the internal standalone generation view.

Parameters:
- `net` - Network instance to cast.

Returns: Internal network properties used by the standalone generator.

### ensureOutputNodesExist

```ts
ensureOutputNodesExist(
  standaloneProps: NetworkStandaloneProps,
): void
```

Validate that the network has at least one output node.

Parameters:
- `standaloneProps` - Internal standalone network view.

Returns: Void.

### createGenerationContext

```ts
createGenerationContext(
  standaloneProps: NetworkStandaloneProps,
): StandaloneGenerationContext
```

Create a fresh generation context used across orchestration steps.

Parameters:
- `standaloneProps` - Internal standalone network view.

Returns: Initialized generation context.

### seedNodeIndexesAndState

```ts
seedNodeIndexesAndState(
  generationContext: StandaloneGenerationContext,
): void
```

Seed index, activation, and state arrays from network nodes.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

## architecture/network/standalone/network.standalone.utils.coverage.ts

### stripCoverage

```ts
stripCoverage(
  code: string,
): string
```

Remove instrumentation artifacts and formatting detritus from function sources.

Parameters:
- `code` - Source text potentially containing coverage wrappers.

Returns: Cleaned source text suitable for deterministic standalone emission.

## architecture/network/standalone/network.standalone.utils.finalize.ts

### assembleStandaloneSource

```ts
assembleStandaloneSource(
  generationContext: StandaloneGenerationContext,
): string
```

Assemble the final standalone IIFE source string.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Final generated source string.

### buildActivationArrayLiteral

```ts
buildActivationArrayLiteral(
  generationContext: StandaloneGenerationContext,
): string
```

Build deterministic activation function array literal by function index ordering.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Comma-separated activation function names.

### resolveActivationArrayType

```ts
resolveActivationArrayType(
  generationContext: StandaloneGenerationContext,
): string
```

Resolve typed-array constructor name based on configured activation precision.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Constructor name used in generated source.

### buildInputGuardLine

```ts
buildInputGuardLine(
  expectedInputSize: number,
): string
```

Build generated input length guard line.

Parameters:
- `expectedInputSize` - Required input vector size.

Returns: Guard statement line including trailing newline.

## architecture/network/standalone/network.standalone.utils.activation.ts

### resolveSquashName

```ts
resolveSquashName(
  currentNode: default,
  nodeTraversalIndex: number,
): string
```

Resolve a stable activation function name for emission.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index for anonymous-name fallback.

Returns: Activation function identifier.

### ensureActivationFunctionIndex

```ts
ensureActivationFunctionIndex(
  generationContext: StandaloneGenerationContext,
  squashName: string,
  squashFunction: StandaloneSquashFunction,
  nodeTraversalIndex: number,
): number
```

Ensure an activation function is registered and return its table index.

Parameters:
- `generationContext` - Mutable generation context.
- `squashName` - Activation function name.
- `squashFunction` - Activation function implementation.
- `nodeTraversalIndex` - Current node index for fallback naming.

Returns: Activation function index within generated `F` array.

### resolveActivationFunctionSource

```ts
resolveActivationFunctionSource(
  squashName: string,
  squashFunction: StandaloneSquashFunction,
  nodeTraversalIndex: number,
): string
```

Resolve emitted source for built-in or custom activation functions.

Parameters:
- `squashName` - Activation function name.
- `squashFunction` - Activation function implementation.
- `nodeTraversalIndex` - Current node index for fallback flow.

Returns: Named function source string.

### normalizeBuiltinSource

```ts
normalizeBuiltinSource(
  sourceCode: string,
  squashName: string,
): string
```

Normalize built-in activation source to a named function and strip coverage artifacts.

Parameters:
- `sourceCode` - Built-in function source.
- `squashName` - Required function name.

Returns: Cleaned named function source.

### normalizeCustomSource

```ts
normalizeCustomSource(
  sourceCode: string,
  squashName: string,
  nodeTraversalIndex: number,
): string
```

Normalize custom activation source with function/arrow/fallback handling.

Parameters:
- `sourceCode` - Custom function source.
- `squashName` - Required function name.
- `nodeTraversalIndex` - Current node index for traversal-context parity.

Returns: Cleaned named function source.

### ensureNamedFunctionSource

```ts
ensureNamedFunctionSource(
  sourceCode: string,
  squashName: string,
): string
```

Ensure generated function source starts with the required named signature.

Parameters:
- `sourceCode` - Function source.
- `squashName` - Required function name.

Returns: Named function source.

### convertArrowToNamedFunction

```ts
convertArrowToNamedFunction(
  sourceCode: string,
  squashName: string,
): string
```

Convert an arrow-function source string into a named function declaration source.

Parameters:
- `sourceCode` - Arrow-function source.
- `squashName` - Required function name.

Returns: Named function source.

### normalizeArrowParameters

```ts
normalizeArrowParameters(
  parameterSegment: string,
): string
```

Normalize arrow parameter segment into comma-separated parameter list content.

Parameters:
- `parameterSegment` - Raw arrow parameter segment.

Returns: Parameter list body (without surrounding parentheses).

### normalizeArrowBody

```ts
normalizeArrowBody(
  bodySegment: string,
): string
```

Normalize arrow body into a function-body block.

Parameters:
- `bodySegment` - Raw arrow body segment.

Returns: Function body block string.

### registerActivationFunction

```ts
registerActivationFunction(
  generationContext: StandaloneGenerationContext,
  squashName: string,
  functionSource: string,
): void
```

Register a function source and allocate its numeric index.

Parameters:
- `generationContext` - Mutable generation context.
- `squashName` - Activation function name.
- `functionSource` - Named function source to store.

Returns: Void.
