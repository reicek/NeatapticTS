import type Network from '../network';

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
const stripCoverage = (code: string): string => {
  code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, ''); // /* istanbul ignore ... */ blocks
  code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, ''); // counters like cov_xyz().s[3]++
  code = code.replace(/cov_[\w$]+\(\)/g, ''); // bare cov_ calls
  code = code.replace(/^\s*\/\/ # sourceMappingURL=.*\s*$/gm, ''); // source maps
  code = code.replace(/\(\s*,\s*/g, '( '); // normalize stray comma spacing
  code = code.replace(/\s*,\s*\)/g, ' )');
  code = code.trim();
  code = code.replace(/^\s*;\s*$/gm, ''); // solitary semicolons
  code = code.replace(/;{2,}/g, ';'); // collapse repeated semicolons
  code = code.replace(/^\s*[,;]?\s*$/gm, ''); // leftover empty tokens
  return code;
};

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
  // 1. Structural validation: ensure at least one output node exists.
  if (!(net as any).nodes.some((nodeRef: any) => nodeRef.type === 'output')) {
    throw new Error(
      'Cannot create standalone function: network has no output nodes.'
    );
  }
  /** Map of activation function name -> emitted source string (deduplication). */
  const emittedActivationSource: Record<string, string> = {};
  /** Ordered list of activation function source strings (in emission order). */
  const activationFunctionSources: string[] = [];
  /** Activation function name -> index in F array (for compact referencing). */
  const activationFunctionIndexMap: Record<string, number> = {};
  /** Counter allocating the next function index. */
  let nextActivationFunctionIndex = 0;
  /** Initial activation values (A array seed). */
  const initialActivations: number[] = [];
  /** Initial state (pre-activation sums) values (S array seed). */
  const initialStates: number[] = [];
  /** Body lines comprising the activate(input) function. */
  const bodyLines: string[] = [];
  /** Built-in activation implementations (canonical, readable forms). */
  const builtinActivationSnippets: Record<string, string> = {
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
    selu:
      'function selu(x){ var a=1.6732632423543772,s=1.0507009873554805; var fx=x>0?x:a*Math.exp(x)-a; return fx*s; }',
    softplus:
      'function softplus(x){ if(x>30)return x; if(x<-30)return Math.exp(x); return Math.max(0,x)+Math.log(1+Math.exp(-Math.abs(x))); }',
    swish: 'function swish(x){ var s=1/(1+Math.exp(-x)); return x*s; }',
    gelu:
      'function gelu(x){ var cdf=0.5*(1.0+Math.tanh(Math.sqrt(2.0/Math.PI)*(x+0.044715*Math.pow(x,3)))); return x*cdf; }',
    mish:
      'function mish(x){ var sp_x; if(x>30){sp_x=x;}else if(x<-30){sp_x=Math.exp(x);}else{sp_x=Math.log(1+Math.exp(x));} var tanh_sp_x=Math.tanh(sp_x); return x*tanh_sp_x; }',
  };

  // 2. Assign stable indices & collect runtime state seeds.
  (net as any).nodes.forEach((node: any, nodeIndex: number) => {
    node.index = nodeIndex;
    initialActivations.push(node.activation);
    initialStates.push(node.state);
  });

  // 3. Emit input seeding loop (direct copy of provided input into A[0..inputSize-1]).
  bodyLines.push('for(var i = 0; i < input.length; i++) A[i] = input[i];');
  // 4. Build computational body for each non-input node.
  for (
    let nodeIndex = (net as any).input;
    nodeIndex < (net as any).nodes.length;
    nodeIndex++
  ) {
    const node: any = (net as any).nodes[nodeIndex];
    const squashFn: any = node.squash;
    const squashName = squashFn.name || `anonymous_squash_${nodeIndex}`;
    // Activation function emission (deduplicate by name).
    if (!(squashName in emittedActivationSource)) {
      let functionSource: string;
      if (builtinActivationSnippets[squashName]) {
        functionSource = builtinActivationSnippets[squashName];
        // Guarantee explicit named function signature (normalize just in case snippet differs).
        if (!functionSource.startsWith(`function ${squashName}`)) {
          functionSource = `function ${squashName}${functionSource.substring(
            functionSource.indexOf('(')
          )}`;
        }
        functionSource = stripCoverage(functionSource);
      } else {
        // Attempt to stringify custom activation; fallback to identity if unparsable.
        functionSource = squashFn.toString();
        functionSource = stripCoverage(functionSource);
        if (functionSource.startsWith('function')) {
          functionSource = `function ${squashName}${functionSource.substring(
            functionSource.indexOf('(')
          )}`;
        } else if (functionSource.includes('=>')) {
          // Arrow function: treat substring from first '(' as params.
          functionSource = `function ${squashName}${functionSource.substring(
            functionSource.indexOf('(')
          )}`;
        } else {
          functionSource = `function ${squashName}(x){ return x; }`;
        }
      }
      emittedActivationSource[squashName] = functionSource;
      activationFunctionSources.push(functionSource);
      activationFunctionIndexMap[squashName] = nextActivationFunctionIndex++;
    }
    const activationFunctionIndex = activationFunctionIndexMap[squashName];
    /** Weighted incoming terms (strings) assembled for nodeIndex. */
    const incomingTerms: string[] = [];
    // Standard feed-forward inbound connections.
    for (const connection of node.connections.in) {
      if (typeof connection.from.index === 'undefined') continue; // Skip malformed edge.
      let term = `A[${connection.from.index}] * ${connection.weight}`;
      // Gating multiplies the signal by the gate node activation (multiplicative modulation).
      if (connection.gater && typeof connection.gater.index !== 'undefined') {
        term += ` * A[${connection.gater.index}]`;
      }
      incomingTerms.push(term);
    }
    // Optional self-connection (recurrent contribution from prior state).
    if (node.connections.self.length > 0) {
      const selfConn = node.connections.self[0];
      let term = `S[${nodeIndex}] * ${selfConn.weight}`;
      if (selfConn.gater && typeof selfConn.gater.index !== 'undefined') {
        term += ` * A[${selfConn.gater.index}]`;
      }
      incomingTerms.push(term);
    }
    /** Summation expression (0 if no inbound edges). */
    const sumExpression =
      incomingTerms.length > 0 ? incomingTerms.join(' + ') : '0';
    bodyLines.push(`S[${nodeIndex}] = ${sumExpression} + ${node.bias};`);
    /** Optional multiplicative mask (e.g., dropout mask captured previously). */
    const maskValue =
      typeof node.mask === 'number' && node.mask !== 1 ? node.mask : 1;
    bodyLines.push(
      `A[${nodeIndex}] = F[${activationFunctionIndex}](S[${nodeIndex}])${
        maskValue !== 1 ? ` * ${maskValue}` : ''
      };`
    );
  }
  // 5. Gather output indices (tail section of node array).
  const outputIndices: number[] = [];
  for (
    let nodeIndex = (net as any).nodes.length - (net as any).output;
    nodeIndex < (net as any).nodes.length;
    nodeIndex++
  ) {
    if (typeof ((net as any).nodes[nodeIndex] as any)?.index !== 'undefined') {
      outputIndices.push(((net as any).nodes[nodeIndex] as any).index);
    }
  }
  bodyLines.push(
    `return [${outputIndices.map((idx) => `A[${idx}]`).join(',')}];`
  );
  // 6. Assemble final source with deterministic activation function ordering by index.
  const activationArrayLiteral = Object.entries(activationFunctionIndexMap)
    .sort(([, a], [, b]) => a - b)
    .map(([name]) => name)
    .join(',');
  const activationArrayType =
    (net as any)._activationPrecision === 'f32'
      ? 'Float32Array'
      : 'Float64Array';
  let generatedSource = '';
  generatedSource += `(function(){\n`;
  generatedSource += `${activationFunctionSources.join('\n')}\n`;
  generatedSource += `var F = [${activationArrayLiteral}];\n`;
  generatedSource += `var A = new ${activationArrayType}([${initialActivations.join(
    ','
  )}]);\n`;
  generatedSource += `var S = new ${activationArrayType}([${initialStates.join(
    ','
  )}]);\n`;
  generatedSource += `function activate(input){\n`;
  generatedSource += `if (!input || input.length !== ${
    (net as any).input
  }) { throw new Error('Invalid input size. Expected ${
    (net as any).input
  }, got ' + (input ? input.length : 'undefined')); }\n`;
  generatedSource += bodyLines.join('\n');
  generatedSource += `}\n`;
  generatedSource += `return activate;\n})();`;
  return generatedSource;
}
