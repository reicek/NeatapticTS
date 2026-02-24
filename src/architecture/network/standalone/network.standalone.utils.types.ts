/** Output node discriminator used for standalone precondition checks. */
export const OUTPUT_NODE_TYPE = 'output';

/** Fallback literal used when a node has no incoming terms. */
export const SINGLE_TERM_FALLBACK = '0';

/** Multiplicative identity used to omit redundant mask expressions. */
export const MASK_MULTIPLIER_IDENTITY = 1;

/** Generated source line for copying external inputs into activation buffer. */
export const INPUT_LOOP_LINE =
  'for(var inputIndex = 0; inputIndex < input.length; inputIndex++) A[inputIndex] = input[inputIndex];';

/** Precision token selecting Float32 activation/state buffers. */
export const ACTIVATION_PRECISION_F32 = 'f32';

/** Typed-array constructor names used in generated source. */
export const FLOAT32_ARRAY_TYPE = 'Float32Array';
/** Typed-array constructor names used in generated source. */
export const FLOAT64_ARRAY_TYPE = 'Float64Array';

/** Prefix token used when normalizing function sources. */
export const FUNCTION_PREFIX = 'function';
/** Arrow token used during function-source normalization. */
export const ARROW_TOKEN = '=>';
/** Identity-function fallback body for invalid custom squash sources. */
export const FALLBACK_IDENTITY_BODY = '(x){ return x; }';

/** Empty replacement used while stripping coverage artifacts. */
export const COVERAGE_REPLACEMENT = '';

/** Error message when attempting standalone generation without outputs. */
export const NO_OUTPUT_NODES_ERROR =
  'Cannot create standalone function: network has no output nodes.';
/** Input-size validation message fragments for generated activate guards. */
export const INVALID_INPUT_SIZE_ERROR_PREFIX = 'Invalid input size. Expected ';
/** Input-size validation message fragments for generated activate guards. */
export const INVALID_INPUT_SIZE_ERROR_MIDDLE = ', got ';

/** Regex stripping Istanbul ignore blocks from stringified functions. */
export const ISTANBUL_IGNORE_BLOCK_REGEX =
  /\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g;
/** Regex stripping Istanbul counters from stringified functions. */
export const COVERAGE_COUNTER_REGEX =
  /cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g;
/** Regex stripping Istanbul function invocations from source snippets. */
export const COVERAGE_CALL_REGEX = /cov_[\w$]+\(\)/g;
/** Regex stripping sourceMappingURL comments from generated snippets. */
export const SOURCE_MAP_REGEX = /^\s*\/\/ # sourceMappingURL=.*\s*$/gm;
/** Regex normalizing stray commas near opening parentheses. */
export const STRAY_COMMA_OPEN_REGEX = /\(\s*,\s*/g;
/** Regex normalizing stray commas near closing parentheses. */
export const STRAY_COMMA_CLOSE_REGEX = /\s*,\s*\)/g;
/** Regex removing solitary semicolon lines created by instrumentation. */
export const SOLITARY_SEMICOLON_REGEX = /^\s*;\s*$/gm;
/** Regex collapsing repeated semicolons. */
export const REPEATED_SEMICOLON_REGEX = /;{2,}/g;
/** Regex removing empty punctuation-only token lines. */
export const EMPTY_TOKEN_REGEX = /^\s*[,;]?\s*$/gm;

/**
 * Built-in activation snippets emitted as named JavaScript function declarations.
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

/** Activation function shape used by standalone source generation helpers. */
export type StandaloneSquashFunction = (
  inputValue: number,
  derivate?: boolean,
) => number;
