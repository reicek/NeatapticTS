/** Output node type discriminator checked during standalone precondition validation to identify activation output targets. */
export const OUTPUT_NODE_TYPE = 'output';

/** Fallback zero literal emitted into generated source when a node has no incoming weighted terms to sum. */
export const SINGLE_TERM_FALLBACK = '0';

/** Multiplicative identity value used to detect and omit redundant gating mask expressions from generated standalone source. */
export const MASK_MULTIPLIER_IDENTITY = 1;

/** Generated source line that copies the caller-supplied input array into the typed activation buffer at inference time. */
export const INPUT_LOOP_LINE =
  'for(var inputIndex = 0; inputIndex < input.length; inputIndex++) A[inputIndex] = input[inputIndex];';

/** Precision token that selects Float32 typed-array activation and state buffers in the generated standalone function. */
export const ACTIVATION_PRECISION_F32 = 'f32';
/** Precision token that selects float16-backed Uint16 storage buffers for lower-memory standalone inference functions. */
export const ACTIVATION_PRECISION_F16 = 'f16';

/** Float32Array constructor name emitted into generated standalone source for single-precision activation buffers. */
export const FLOAT32_ARRAY_TYPE = 'Float32Array';
/** Float64Array constructor name emitted into generated standalone source for double-precision activation buffers. */
export const FLOAT64_ARRAY_TYPE = 'Float64Array';
/** Uint16Array constructor name emitted into generated standalone source for float16-backed state storage buffers. */
export const UINT16_ARRAY_TYPE = 'Uint16Array';

/** Prefix token detected when normalizing stringified function sources before stripping instrumentation artifacts. */
export const FUNCTION_PREFIX = 'function';
/** Arrow token detected during function-source normalization for stripping instrumentation from arrow-style squash functions. */
export const ARROW_TOKEN = '=>';
/** Identity-function fallback body injected when a custom squash source cannot be normalized to a valid form. */
export const FALLBACK_IDENTITY_BODY = '(x){ return x; }';

/** Empty string replacement substituted when stripping Istanbul coverage instrumentation artifacts from function source. */
export const COVERAGE_REPLACEMENT = '';

/** Error thrown when standalone generation is attempted on a network that has no output nodes to emit. */
export const NO_OUTPUT_NODES_ERROR =
  'Cannot create standalone function: network has no output nodes.';
/** Input-size validation prefix fragment emitted by the generated standalone activate guard at inference time. */
export const INVALID_INPUT_SIZE_ERROR_PREFIX = 'Invalid input size. Expected ';
/** Input-size validation middle fragment joining expected and actual counts in the generated activate guard message. */
export const INVALID_INPUT_SIZE_ERROR_MIDDLE = ', got ';

/** Regex that strips Istanbul ignore-hint block comments from stringified activation functions before standalone emission. */
export const ISTANBUL_IGNORE_BLOCK_REGEX =
  /\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g;
/** Regex that strips Istanbul statement, function, and branch counter increments from stringified activation source. */
export const COVERAGE_COUNTER_REGEX =
  /cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g;
/** Regex that strips bare Istanbul cov_ function invocations left behind after counter removal. */
export const COVERAGE_CALL_REGEX = /cov_[\w$]+\(\)/g;
/** Regex that strips sourceMappingURL comments from generated code snippets to keep standalone output clean. */
export const SOURCE_MAP_REGEX = /^\s*\/\/\s*# sourceMappingURL=.*\s*$/gm;
/** Regex that normalizes stray commas adjacent to opening parentheses created by coverage stripping. */
export const STRAY_COMMA_OPEN_REGEX = /\(\s*,\s*/g;
/** Regex that normalizes stray commas adjacent to closing parentheses created by coverage stripping. */
export const STRAY_COMMA_CLOSE_REGEX = /\s*,\s*\)/g;
/** Regex that removes solitary semicolon lines left behind after Istanbul instrumentation removal passes. */
export const SOLITARY_SEMICOLON_REGEX = /^\s*;\s*$/gm;
/** Regex that collapses consecutive double-semicolons produced as coverage-stripping side effects in generated standalone source. */
export const REPEATED_SEMICOLON_REGEX = /;{2,}/g;
/** Regex that removes lines containing only punctuation tokens left behind by instrumentation cleanup passes. */
export const EMPTY_TOKEN_REGEX = /^\s*[,;]+\s*$/gm;

/**
 * Built-in activation function snippets emitted as named JavaScript declarations into self-contained standalone inference functions.
 *
 * Values are intentionally compact so emitted standalone source remains deterministic and small.
 */
export const BUILTIN_ACTIVATION_SNIPPETS: Record<string, string> = {
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

/** Activation function signature expected by standalone source generation helpers when resolving custom squash callables. */
export type StandaloneSquashFunction = (
  inputValue: number,
  derivate?: boolean,
) => number;
