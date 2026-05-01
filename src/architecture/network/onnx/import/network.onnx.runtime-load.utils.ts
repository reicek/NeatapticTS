import Layer from '../../../layer/layer';
import Network from '../../network';
import type {
  OnnxPerceptronBuildContext,
  OnnxPerceptronSizeValidationContext,
  OnnxRuntimeFactories,
  OnnxRuntimeLayerFactory,
  OnnxRuntimeLayerModule,
  OnnxRuntimePerceptronFactory,
} from './network.onnx.runtime-load.types';
import { NetworkOnnxPerceptronSizeValidationError } from '../network.onnx.errors';

/** Minimum number of layer-size values needed for input/output perceptron construction. */
const MINIMUM_PERCEPTRON_SIZE_COUNT = 2;

/** Error message used when ONNX architecture metadata omits input/output sizes. */
const PERCEPTRON_SIZE_VALIDATION_ERROR_MESSAGE =
  'ONNX import requires at least input and output sizes';

/** Property key used for LSTM layer constructor wiring in runtime layer module. */
const LSTM_LAYER_MODULE_KEY = 'lstm';

/** Property key used for GRU layer constructor wiring in runtime layer module. */
const GRU_LAYER_MODULE_KEY = 'gru';

/** Index of the perceptron input size in the layer-size list. */
const INPUT_SIZE_INDEX = 0;

/** Inclusive start index of hidden-layer size values in the layer-size list. */
const HIDDEN_LAYER_SLICE_START_INDEX = 1;

/** Exclusive end offset for hidden-layer size extraction (exclude output layer). */
const HIDDEN_LAYER_SLICE_END_OFFSET = -1;

/** Fallback output size used only when defensive extraction receives malformed input. */
const OUTPUT_SIZE_FALLBACK_COUNT = 0;

/**
 * Resolve runtime factories used by ONNX import orchestration.
 *
 * @returns Perceptron factory and layer module object.
 */
export function loadRuntimeFactories(): OnnxRuntimeFactories {
  // Step 1: Resolve the perceptron factory used by import reconstruction.
  const perceptronFactory = createPerceptronFactory();

  // Step 2: Resolve runtime recurrent-layer constructors.
  const layerModule = createRuntimeLayerModule();

  // Step 3: Fold runtime dependencies into one transport object.
  return foldRuntimeFactories(perceptronFactory, layerModule);
}

/**
 * Create an ONNX import network factory from modern static constructors.
 *
 * @returns Perceptron-compatible factory function.
 */
function createPerceptronFactory(): OnnxRuntimePerceptronFactory {
  return (...sizes: number[]): Network => {
    // Step 1: Validate incoming architecture size data.
    const validationContext = createPerceptronSizeValidationContext(sizes);
    validatePerceptronSizes(validationContext);

    // Step 2: Build a tiny context for deterministic size extraction.
    const buildContext = createPerceptronBuildContext(sizes);

    // Step 3: Fold context into a concrete perceptron network instance.
    return buildPerceptronNetwork(buildContext);
  };
}

/**
 * Create the runtime layer-module wiring used by ONNX import orchestrators.
 *
 * @returns Runtime recurrent-layer module object.
 */
function createRuntimeLayerModule(): OnnxRuntimeLayerModule {
  // Step 1: Resolve the LSTM runtime constructor.
  const lstmFactory = resolveLayerFactory(LSTM_LAYER_MODULE_KEY);

  // Step 2: Resolve the GRU runtime constructor.
  const gruFactory = resolveLayerFactory(GRU_LAYER_MODULE_KEY);

  // Step 3: Fold constructors into a runtime layer module.
  return foldRuntimeLayerModule(lstmFactory, gruFactory);
}

/**
 * Fold runtime perceptron and layer module into a transport payload.
 *
 * @param perceptronFactory Perceptron factory function.
 * @param layerModule Runtime recurrent-layer constructors.
 * @returns Runtime factories payload.
 */
function foldRuntimeFactories(
  perceptronFactory: OnnxRuntimePerceptronFactory,
  layerModule: OnnxRuntimeLayerModule,
): OnnxRuntimeFactories {
  return {
    perceptronFactory,
    layerModule,
  };
}

/**
 * Fold LSTM/GRU factories into a runtime layer module payload.
 *
 * @param lstmFactory Runtime LSTM layer factory.
 * @param gruFactory Runtime GRU layer factory.
 * @returns Runtime layer module.
 */
function foldRuntimeLayerModule(
  lstmFactory: OnnxRuntimeLayerFactory,
  gruFactory: OnnxRuntimeLayerFactory,
): OnnxRuntimeLayerModule {
  return {
    [LSTM_LAYER_MODULE_KEY]: lstmFactory,
    [GRU_LAYER_MODULE_KEY]: gruFactory,
  };
}

/**
 * Resolve one runtime layer factory by module key.
 *
 * @param layerKey Runtime layer key.
 * @returns Matching layer factory.
 */
function resolveLayerFactory(
  layerKey: keyof OnnxRuntimeLayerModule,
): OnnxRuntimeLayerFactory {
  if (layerKey === LSTM_LAYER_MODULE_KEY) {
    return Layer.lstm;
  }

  return Layer.gru;
}

/**
 * Build perceptron-size validation context.
 *
 * @param sizes Layer-size payload.
 * @returns Validation context.
 */
function createPerceptronSizeValidationContext(
  sizes: number[],
): OnnxPerceptronSizeValidationContext {
  return {
    sizes,
    minimumSizeCount: MINIMUM_PERCEPTRON_SIZE_COUNT,
    errorMessage: PERCEPTRON_SIZE_VALIDATION_ERROR_MESSAGE,
  };
}

/**
 * Validate perceptron size-list constraints.
 *
 * @param validationContext Validation context.
 * @returns Nothing. Throws on invalid size-list.
 */
function validatePerceptronSizes(
  validationContext: OnnxPerceptronSizeValidationContext,
): void {
  if (validationContext.sizes.length >= validationContext.minimumSizeCount) {
    return;
  }

  throw new NetworkOnnxPerceptronSizeValidationError(
    validationContext.errorMessage,
  );
}

/**
 * Build perceptron-network construction context.
 *
 * @param sizes Layer-size payload.
 * @returns Build context.
 */
function createPerceptronBuildContext(
  sizes: number[],
): OnnxPerceptronBuildContext {
  return {
    sizes,
    inputIndex: INPUT_SIZE_INDEX,
    hiddenSliceStartIndex: HIDDEN_LAYER_SLICE_START_INDEX,
    hiddenSliceEndOffset: HIDDEN_LAYER_SLICE_END_OFFSET,
    outputFallbackCount: OUTPUT_SIZE_FALLBACK_COUNT,
  };
}

/**
 * Build a perceptron network from size-extraction context.
 *
 * @param buildContext Perceptron build context.
 * @returns Reconstructed network instance.
 */
function buildPerceptronNetwork(
  buildContext: OnnxPerceptronBuildContext,
): Network {
  // Step 1: Resolve input layer width.
  const inputCount = buildContext.sizes[buildContext.inputIndex];

  // Step 2: Resolve hidden layer widths.
  const hiddenLayerSizes = buildContext.sizes.slice(
    buildContext.hiddenSliceStartIndex,
    buildContext.hiddenSliceEndOffset,
  );

  // Step 3: Resolve output layer width. at(-1) is always defined: validation ensures ≥2 sizes.
  const outputCount = buildContext.sizes.at(buildContext.hiddenSliceEndOffset)!;

  // Step 4: Fold widths into MLP creation.
  return Network.createMLP(inputCount, hiddenLayerSizes, outputCount);
}
