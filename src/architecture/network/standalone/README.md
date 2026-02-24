# architecture/network/standalone

## architecture/network/standalone/network.standalone.utils.types.ts

### network.standalone.utils.types

Output node discriminator used for standalone precondition checks.

### ACTIVATION_PRECISION_F32

### ARROW_TOKEN

### BUILTIN_ACTIVATION_SNIPPETS

### COVERAGE_CALL_REGEX

### COVERAGE_COUNTER_REGEX

### COVERAGE_REPLACEMENT

### EMPTY_TOKEN_REGEX

### FALLBACK_IDENTITY_BODY

### FLOAT32_ARRAY_TYPE

### FLOAT64_ARRAY_TYPE

### FUNCTION_PREFIX

### INPUT_LOOP_LINE

### INVALID_INPUT_SIZE_ERROR_MIDDLE

### INVALID_INPUT_SIZE_ERROR_PREFIX

### ISTANBUL_IGNORE_BLOCK_REGEX

### MASK_MULTIPLIER_IDENTITY

### NO_OUTPUT_NODES_ERROR

### OUTPUT_NODE_TYPE

### REPEATED_SEMICOLON_REGEX

### SINGLE_TERM_FALLBACK

### SOLITARY_SEMICOLON_REGEX

### SOURCE_MAP_REGEX

### StandaloneSquashFunction

`(inputValue: number, derivate: boolean | undefined) => number`

Activation function shape used by standalone source generation helpers.

### STRAY_COMMA_CLOSE_REGEX

### STRAY_COMMA_OPEN_REGEX

## architecture/network/standalone/network.standalone.utils.ts

### generateStandalone

`(net: import("C:/NeatapticTS/src/architecture/network").default) => string`

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

## architecture/network/standalone/network.standalone.utils.loop.ts

### appendActivationLine

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, nodeTraversalIndex: number, activationFunctionIndex: number, maskValue: number) => void`

Append generated activation assignment line for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index.
- `activationFunctionIndex` - Function table index.
- `maskValue` - Multiplicative mask.

Returns: Void.

### appendAllNodeComputationLines

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => void`

Append compute lines for all non-input nodes.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

### appendInputSeedLine

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => void`

Append the generated input-copy loop to the standalone body.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

### appendOutputReturnLine

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, outputIndexes: number[]) => void`

Append generated return line for output activations.

Parameters:
- `generationContext` - Mutable generation context.
- `outputIndexes` - Output node indexes.

Returns: Void.

### appendSingleNodeComputationLines

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, nodeTraversalIndex: number) => void`

Append state and activation lines for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index currently being emitted.

Returns: Void.

### appendStateLine

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, nodeTraversalIndex: number, sumExpression: string, biasValue: number) => void`

Append generated state assignment line for one node.

Parameters:
- `generationContext` - Mutable generation context.
- `nodeTraversalIndex` - Node index.
- `sumExpression` - Generated sum expression.
- `biasValue` - Node bias.

Returns: Void.

### buildMaskSuffix

`(maskValue: number) => string`

Build optional activation mask suffix for generated assignment line.

Parameters:
- `maskValue` - Multiplicative mask value.

Returns: Empty suffix for identity, otherwise multiplicative fragment.

## architecture/network/standalone/network.standalone.utils.graph.ts

### appendGateMultiplier

`(connectionTerm: string, gateNode: import("C:/NeatapticTS/src/architecture/node").default | null) => string`

Append a gate activation multiplier to a connection term when a gate exists.

Parameters:
- `connectionTerm` - Base connection term.
- `gateNode` - Optional gate node.

Returns: Term with optional gate multiplier.

### buildNodeSumExpression

`(currentNode: import("C:/NeatapticTS/src/architecture/node").default, nodeTraversalIndex: number) => string`

Build the pre-activation sum expression for one node.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index.

Returns: String expression used for generated `S[index]` assignment.

### collectIncomingTerms

`(currentNode: import("C:/NeatapticTS/src/architecture/node").default) => string[]`

Collect feed-forward inbound connection terms for a node.

Parameters:
- `currentNode` - Current node.

Returns: Weighted term expressions.

### collectOutputIndexes

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => number[]`

Collect output node indexes from the output tail segment.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Output indexes used for result array emission.

### collectSelfConnectionTerms

`(currentNode: import("C:/NeatapticTS/src/architecture/node").default, nodeTraversalIndex: number) => string[]`

Collect recurrent self-connection term for a node when present.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index used for self-state reference.

Returns: Zero or one recurrent term expressions.

### foldTermsIntoExpression

`(allTerms: string[]) => string`

Fold a term collection into a summation expression.

Parameters:
- `allTerms` - Term collection.

Returns: Summation expression or fallback zero literal.

### formatOutputArrayValues

`(outputIndexes: number[]) => string`

Format output activation selectors for generated return expression.

Parameters:
- `outputIndexes` - Output node indexes.

Returns: Comma-separated `A[index]` selector list.

### getOptionalNodeIndex

`(nodeReference: import("C:/NeatapticTS/src/architecture/node").default | null) => number | undefined`

Resolve optional generated node index from a node reference.

Parameters:
- `nodeReference` - Optional node reference.

Returns: Node index when available.

### mergeTermCollections

`(firstTerms: string[], secondTerms: string[]) => string[]`

Merge two term lists into a single ordered list.

Parameters:
- `firstTerms` - First term collection.
- `secondTerms` - Second term collection.

Returns: Combined term collection.

## architecture/network/standalone/network.standalone.utils.setup.ts

### asStandaloneProps

`(net: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkStandaloneProps`

Cast a network instance to the internal standalone generation view.

Parameters:
- `net` - Network instance to cast.

Returns: Internal network properties used by the standalone generator.

### createGenerationContext

`(standaloneProps: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkStandaloneProps) => import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext`

Create a fresh generation context used across orchestration steps.

Parameters:
- `standaloneProps` - Internal standalone network view.

Returns: Initialized generation context.

### ensureOutputNodesExist

`(standaloneProps: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkStandaloneProps) => void`

Validate that the network has at least one output node.

Parameters:
- `standaloneProps` - Internal standalone network view.

Returns: Void.

### seedNodeIndexesAndState

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => void`

Seed index, activation, and state arrays from network nodes.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Void.

## architecture/network/standalone/network.standalone.utils.coverage.ts

### stripCoverage

`(code: string) => string`

Remove instrumentation artifacts and formatting detritus from function sources.

Parameters:
- `code` - Source text potentially containing coverage wrappers.

Returns: Cleaned source text suitable for deterministic standalone emission.

## architecture/network/standalone/network.standalone.utils.finalize.ts

### assembleStandaloneSource

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => string`

Assemble the final standalone IIFE source string.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Final generated source string.

### buildActivationArrayLiteral

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => string`

Build deterministic activation function array literal by function index ordering.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Comma-separated activation function names.

### buildInputGuardLine

`(expectedInputSize: number) => string`

Build generated input length guard line.

Parameters:
- `expectedInputSize` - Required input vector size.

Returns: Guard statement line including trailing newline.

### resolveActivationArrayType

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext) => string`

Resolve typed-array constructor name based on configured activation precision.

Parameters:
- `generationContext` - Mutable generation context.

Returns: Constructor name used in generated source.

## architecture/network/standalone/network.standalone.utils.activation.ts

### convertArrowToNamedFunction

`(sourceCode: string, squashName: string) => string`

Convert an arrow-function source string into a named function declaration source.

Parameters:
- `sourceCode` - Arrow-function source.
- `squashName` - Required function name.

Returns: Named function source.

### ensureActivationFunctionIndex

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, squashName: string, squashFunction: import("C:/NeatapticTS/src/architecture/network/standalone/network.standalone.utils.types").StandaloneSquashFunction, nodeTraversalIndex: number) => number`

Ensure an activation function is registered and return its table index.

Parameters:
- `generationContext` - Mutable generation context.
- `squashName` - Activation function name.
- `squashFunction` - Activation function implementation.
- `nodeTraversalIndex` - Current node index for fallback naming.

Returns: Activation function index within generated `F` array.

### ensureNamedFunctionSource

`(sourceCode: string, squashName: string) => string`

Ensure generated function source starts with the required named signature.

Parameters:
- `sourceCode` - Function source.
- `squashName` - Required function name.

Returns: Named function source.

### normalizeArrowBody

`(bodySegment: string) => string`

Normalize arrow body into a function-body block.

Parameters:
- `bodySegment` - Raw arrow body segment.

Returns: Function body block string.

### normalizeArrowParameters

`(parameterSegment: string) => string`

Normalize arrow parameter segment into comma-separated parameter list content.

Parameters:
- `parameterSegment` - Raw arrow parameter segment.

Returns: Parameter list body (without surrounding parentheses).

### normalizeBuiltinSource

`(sourceCode: string, squashName: string) => string`

Normalize built-in activation source to a named function and strip coverage artifacts.

Parameters:
- `sourceCode` - Built-in function source.
- `squashName` - Required function name.

Returns: Cleaned named function source.

### normalizeCustomSource

`(sourceCode: string, squashName: string, nodeTraversalIndex: number) => string`

Normalize custom activation source with function/arrow/fallback handling.

Parameters:
- `sourceCode` - Custom function source.
- `squashName` - Required function name.
- `nodeTraversalIndex` - Current node index for traversal-context parity.

Returns: Cleaned named function source.

### registerActivationFunction

`(generationContext: import("C:/NeatapticTS/src/architecture/network/network.types").StandaloneGenerationContext, squashName: string, functionSource: string) => void`

Register a function source and allocate its numeric index.

Parameters:
- `generationContext` - Mutable generation context.
- `squashName` - Activation function name.
- `functionSource` - Named function source to store.

Returns: Void.

### resolveActivationFunctionSource

`(squashName: string, squashFunction: import("C:/NeatapticTS/src/architecture/network/standalone/network.standalone.utils.types").StandaloneSquashFunction, nodeTraversalIndex: number) => string`

Resolve emitted source for built-in or custom activation functions.

Parameters:
- `squashName` - Activation function name.
- `squashFunction` - Activation function implementation.
- `nodeTraversalIndex` - Current node index for fallback flow.

Returns: Named function source string.

### resolveSquashName

`(currentNode: import("C:/NeatapticTS/src/architecture/node").default, nodeTraversalIndex: number) => string`

Resolve a stable activation function name for emission.

Parameters:
- `currentNode` - Current node.
- `nodeTraversalIndex` - Node index for anonymous-name fallback.

Returns: Activation function identifier.
