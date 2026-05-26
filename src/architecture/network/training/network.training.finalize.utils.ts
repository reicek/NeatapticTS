import * as methods from '../../../methods/methods';
import { config } from '../../../config';
import type Network from '../../network/network';
import type {
  CostFunction,
  CostFunctionOrObject,
  OptimizerConfigBase,
  PlateauSmoothingConfig,
  PlateauSmoothingState,
  PrimarySmoothingState,
  TrainingConnectionInternals as ConnectionInternals,
  TrainingNetworkInternals as NetworkInternals,
  TrainingNodeInternals as NodeInternals,
  TrainingOptions,
} from '../network.types';
import { trainSetCore } from './network.training.loop.utils';
import {
  ALLOWED_OPTIMIZERS,
  buildMonitoredSmoothingConfig,
  resolveEmaAlpha,
} from './network.training.utils.types';
import {
  computeMonitoredError,
  computePlateauMetric,
} from './network.training.smoothing.utils';
import {
  NetworkTrainingAccumulationStepsError,
  NetworkTrainingBatchSizeError,
  NetworkTrainingDatasetCompatibilityError,
  NetworkTrainingDropoutRangeError,
  NetworkTrainingInvalidCostFunctionError,
  NetworkTrainingInvalidOptimizerOptionError,
  NetworkTrainingNestedLookaheadError,
  NetworkTrainingStoppingConditionRequiredError,
  NetworkTrainingUnknownLookaheadBaseTypeError,
  NetworkTrainingUnknownOptimizerTypeError,
} from './network.training.errors';

type NumberRingBuffer = {
  push: (value: number) => void;
  values: () => number[];
};

type TrainingSmoothingBundle = {
  monitoredSmoothingConfig: ReturnType<typeof buildMonitoredSmoothingConfig>;
  plateauBuffer: NumberRingBuffer;
  plateauSmoothingConfig: PlateauSmoothingConfig;
  plateauSmoothingState: PlateauSmoothingState;
  primarySmoothingState: PrimarySmoothingState;
  recentErrorsBuffer: NumberRingBuffer;
};

type EarlyStopState = {
  bestError: number;
  noImproveCount: number;
};

type ResolvedTrainingRuntimeOptions = {
  accumulationReduction: 'average' | 'sum';
  accumulationSteps: number;
  baseRate: number;
  batchSize: number;
  cost: CostFunction | CostFunctionOrObject;
  dropout: number;
  iterations: number;
  momentum: number;
  targetError: number;
};

type TrainingIterationResult = {
  monitoredError: number;
  plateauError: number;
};

const createNumberRingBuffer = (capacity: number): NumberRingBuffer => {
  const buffer: number[] = new Array(capacity);
  let count = 0;
  let writeIndex = 0;

  return {
    push(value: number): void {
      if (capacity === 1) {
        buffer[0] = value;
        count = 1;
        writeIndex = 0;
        return;
      }

      buffer[writeIndex] = value;
      writeIndex = (writeIndex + 1) % capacity;
      if (count < capacity) count++;
    },
    values(): number[] {
      if (count < capacity) {
        return buffer.slice(0, count);
      }

      const orderedValues = new Array(count);
      for (let sampleIndex = 0; sampleIndex < count; sampleIndex++) {
        orderedValues[sampleIndex] =
          buffer[(writeIndex + sampleIndex) % capacity];
      }

      return orderedValues;
    },
  };
};

const configureGradientClipping = (
  internalNet: NetworkInternals,
  gradientClipConfig: TrainingOptions['gradientClip'],
): void => {
  if (gradientClipConfig) {
    if (gradientClipConfig.mode) {
      internalNet._currentGradClip = {
        mode: gradientClipConfig.mode,
        maxNorm: gradientClipConfig.maxNorm,
        percentile: gradientClipConfig.percentile,
      };
    } else if (typeof gradientClipConfig.maxNorm === 'number') {
      internalNet._currentGradClip = {
        mode: 'norm',
        maxNorm: gradientClipConfig.maxNorm,
      };
    } else if (typeof gradientClipConfig.percentile === 'number') {
      internalNet._currentGradClip = {
        mode: 'percentile',
        percentile: gradientClipConfig.percentile,
      };
    }
    internalNet._gradClipSeparateBias = !!gradientClipConfig.separateBias;
    return;
  }

  internalNet._currentGradClip = undefined;
  internalNet._gradClipSeparateBias = false;
};

const configureMixedPrecision = (
  net: Network,
  internalNet: NetworkInternals,
  mixedPrecision: TrainingOptions['mixedPrecision'],
): void => {
  if (mixedPrecision) {
    const mixedPrecisionConfig =
      mixedPrecision === true ? { lossScale: 1024 } : mixedPrecision;
    internalNet._mixedPrecision.enabled = true;
    internalNet._mixedPrecision.lossScale =
      mixedPrecisionConfig.lossScale || 1024;
    const dynamicConfig = mixedPrecisionConfig.dynamic || {};
    internalNet._mixedPrecisionState.minLossScale = dynamicConfig.minScale || 1;
    internalNet._mixedPrecisionState.maxLossScale =
      dynamicConfig.maxScale || 65536;
    internalNet._mpIncreaseEvery =
      dynamicConfig.increaseEvery ||
      dynamicConfig.stableStepsForIncrease ||
      200;
    net.connections.forEach((connection) => {
      const connectionInternal = connection as unknown as ConnectionInternals;
      connectionInternal._fp32Weight = connection.weight;
    });
    net.nodes.forEach((node) => {
      if (node.type !== 'input') {
        const nodeInternal = node as unknown as NodeInternals;
        nodeInternal._fp32Bias = node.bias;
      }
    });
    return;
  }

  internalNet._mixedPrecision.enabled = false;
  internalNet._mixedPrecision.lossScale = 1;
  internalNet._mpIncreaseEvery = 200;
};

const resolveOptimizerConfig = (
  optimizer: TrainingOptions['optimizer'],
): OptimizerConfigBase | undefined => {
  if (typeof optimizer === 'undefined') {
    return undefined;
  }

  let optimizerConfig: OptimizerConfigBase;
  if (typeof optimizer === 'string') {
    optimizerConfig = { type: optimizer.toLowerCase() };
  } else if (typeof optimizer === 'object' && optimizer !== null) {
    optimizerConfig = { ...optimizer };
    if (typeof optimizerConfig.type === 'string') {
      optimizerConfig.type = optimizerConfig.type.toLowerCase();
    }
  } else {
    throw new NetworkTrainingInvalidOptimizerOptionError(
      'Invalid optimizer option; must be string or object',
    );
  }

  if (!ALLOWED_OPTIMIZERS.has(optimizerConfig.type)) {
    throw new NetworkTrainingUnknownOptimizerTypeError(
      `Unknown optimizer type: ${optimizerConfig.type}`,
    );
  }

  if (optimizerConfig.type === 'lookahead') {
    if (!optimizerConfig.baseType) optimizerConfig.baseType = 'adam';
    if (optimizerConfig.baseType === 'lookahead') {
      throw new NetworkTrainingNestedLookaheadError(
        'Nested lookahead (baseType lookahead) is not supported',
      );
    }
    if (!ALLOWED_OPTIMIZERS.has(optimizerConfig.baseType)) {
      throw new NetworkTrainingUnknownLookaheadBaseTypeError(
        `Unknown baseType for lookahead: ${optimizerConfig.baseType}`,
      );
    }
    optimizerConfig.la_k = optimizerConfig.la_k || 5;
    optimizerConfig.la_alpha = optimizerConfig.la_alpha ?? 0.5;
  }

  return optimizerConfig;
};

const validateTrainingOptions = (
  net: Network,
  set: { input: number[]; output: number[] }[],
  options: TrainingOptions,
  cost: CostFunction | CostFunctionOrObject,
  dropout: number,
  batchSize: number,
  accumulationSteps: number,
): void => {
  if (
    !set ||
    set.length === 0 ||
    set[0].input.length !== net.input ||
    set[0].output.length !== net.output
  ) {
    throw new NetworkTrainingDatasetCompatibilityError(
      'Dataset is invalid or dimensions do not match network input/output size!',
    );
  }

  if (
    typeof options.iterations === 'undefined' &&
    typeof options.error === 'undefined'
  ) {
    if (config.warnings) {
      console.warn('Missing `iterations` or `error` option.');
    }

    throw new NetworkTrainingStoppingConditionRequiredError(
      'Missing `iterations` or `error` option. Training requires a stopping condition.',
    );
  }

  if (config.warnings) {
    if (typeof options.rate === 'undefined') {
      console.warn('Missing `rate` option');
      console.warn('Missing `rate` option, using default learning rate 0.3.');
    }

    if (typeof options.iterations === 'undefined') {
      console.warn(
        'Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met.',
      );
    }
  }

  if (
    typeof cost !== 'function' &&
    !(
      typeof cost === 'object' &&
      cost !== null &&
      (typeof (cost as CostFunctionOrObject).fn === 'function' ||
        typeof (cost as CostFunctionOrObject).calculate === 'function')
    )
  ) {
    throw new NetworkTrainingInvalidCostFunctionError(
      'Invalid cost function provided to Network.train.',
    );
  }

  if (!Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    throw new NetworkTrainingDropoutRangeError('dropout must be in [0,1)');
  }

  if (batchSize > set.length) {
    throw new NetworkTrainingBatchSizeError(
      'Batch size cannot be larger than the dataset length.',
    );
  }

  if (accumulationSteps < 1 || !Number.isFinite(accumulationSteps)) {
    throw new NetworkTrainingAccumulationStepsError(
      'accumulationSteps must be >=1',
    );
  }
};

const buildSmoothingConfig = (
  options: TrainingOptions,
): TrainingSmoothingBundle => {
  const movingAverageWindow = Math.max(1, options.movingAverageWindow || 1);
  const movingAverageType = options.movingAverageType || 'sma';
  const emaAlpha =
    movingAverageType === 'ema'
      ? resolveEmaAlpha(movingAverageWindow, options.emaAlpha)
      : undefined;
  const plateauWindow = Math.max(
    1,
    options.plateauMovingAverageWindow || movingAverageWindow,
  );
  const plateauType = options.plateauMovingAverageType || movingAverageType;
  const plateauEmaAlpha =
    plateauType === 'ema'
      ? resolveEmaAlpha(plateauWindow, options.plateauEmaAlpha)
      : undefined;

  return {
    monitoredSmoothingConfig: buildMonitoredSmoothingConfig(
      movingAverageType,
      movingAverageWindow,
      emaAlpha,
      options.trimmedRatio,
    ),
    plateauBuffer: createNumberRingBuffer(plateauWindow),
    plateauSmoothingConfig: {
      type: plateauType,
      window: plateauWindow,
      emaAlpha: plateauEmaAlpha,
    },
    plateauSmoothingState: {
      plateauEmaValue: undefined,
    },
    primarySmoothingState: {
      emaValue: undefined,
      adaptiveBaseEmaValue: undefined,
      adaptiveEmaValue: undefined,
    },
    recentErrorsBuffer: createNumberRingBuffer(movingAverageWindow),
  };
};

const applyTrainingCallbacks = (
  iteration: number,
  finalError: number,
  plateauError: number,
  options: TrainingOptions,
  internalNet: NetworkInternals,
  net: Network,
): void => {
  if (typeof options.metricsHook === 'function') {
    try {
      options.metricsHook({
        iteration,
        error: finalError,
        plateauError,
        gradNorm: internalNet._lastGradNorm as number,
      });
    } catch {
      // Intentionally ignore errors from user-provided metrics hook.
    }
  }

  if (options.checkpoint && typeof options.checkpoint.save === 'function') {
    if (options.checkpoint.last) {
      try {
        options.checkpoint.save({
          type: 'last',
          iteration,
          error: finalError,
          network: net.toJSON(),
        });
      } catch {
        // Intentionally ignore errors from user-provided checkpoint callback.
      }
    }

    if (
      options.checkpoint.best &&
      (finalError < internalNet._checkpointBestError! ||
        internalNet._checkpointBestError == null)
    ) {
      internalNet._checkpointBestError = finalError;

      try {
        options.checkpoint.save({
          type: 'best',
          iteration,
          error: finalError,
          network: net.toJSON(),
        });
      } catch {
        // Intentionally ignore errors from user-provided checkpoint callback.
      }
    }
  }

  if (
    options.schedule &&
    options.schedule.iterations &&
    iteration % options.schedule.iterations === 0
  ) {
    try {
      options.schedule.function({ error: finalError, iteration });
    } catch {
      // Intentionally ignore errors from user-provided schedule callback.
    }
  }
};

const shouldEarlyStopNow = (
  finalError: number,
  state: EarlyStopState,
  options: TrainingOptions,
  targetError: number,
): boolean => {
  const earlyStopPatience = options.earlyStopPatience || undefined;
  const earlyStopMinDelta = options.earlyStopMinDelta || 0;

  if (finalError < state.bestError - earlyStopMinDelta) {
    state.bestError = finalError;
    state.noImproveCount = 0;
  } else if (earlyStopPatience) {
    state.noImproveCount++;
  }

  return (
    (earlyStopPatience && state.noImproveCount >= earlyStopPatience) ||
    finalError <= targetError
  );
};

const resolveTrainingRuntimeOptions = (
  options: TrainingOptions,
): ResolvedTrainingRuntimeOptions => ({
  accumulationReduction:
    options.accumulationReduction === 'sum' ? 'sum' : 'average',
  accumulationSteps: options.accumulationSteps || 1,
  baseRate: options.rate ?? 0.3,
  batchSize: options.batchSize || 1,
  cost: options.cost ?? methods.Cost.mse,
  dropout: options.dropout ?? 0,
  iterations: options.iterations ?? Number.MAX_SAFE_INTEGER,
  momentum: options.momentum || 0,
  targetError: options.error ?? -Infinity,
});

const runTrainingIteration = (
  net: Network,
  internalNet: NetworkInternals,
  set: { input: number[]; output: number[] }[],
  iteration: number,
  options: TrainingOptions,
  runtimeOptions: ResolvedTrainingRuntimeOptions,
  optimizerConfig: OptimizerConfigBase | undefined,
  smoothing: TrainingSmoothingBundle,
): TrainingIterationResult => {
  internalNet._maybePrune!((internalNet._globalEpoch || 0) + iteration);

  const trainError = trainSetCore(
    net,
    set,
    runtimeOptions.batchSize,
    runtimeOptions.accumulationSteps,
    runtimeOptions.baseRate,
    runtimeOptions.momentum,
    {},
    runtimeOptions.cost,
    optimizerConfig,
  );

  smoothing.recentErrorsBuffer.push(trainError);
  const monitoredError = computeMonitoredError(
    trainError,
    smoothing.recentErrorsBuffer.values(),
    smoothing.monitoredSmoothingConfig,
    smoothing.primarySmoothingState,
  );

  smoothing.plateauBuffer.push(trainError);
  const plateauError = computePlateauMetric(
    trainError,
    smoothing.plateauBuffer.values(),
    smoothing.plateauSmoothingConfig,
    smoothing.plateauSmoothingState,
  );

  return {
    monitoredError,
    plateauError,
  };
};

const finalizeTrainingRun = (
  net: Network,
  internalNet: NetworkInternals,
  performedIterations: number,
): void => {
  net.nodes.forEach((node) => {
    if (node.type === 'hidden') {
      node.mask = 1;
    }
  });
  net.dropout = 0;
  internalNet._globalEpoch =
    (internalNet._globalEpoch || 0) + performedIterations;
};

/**
 * Run the full training orchestration loop with smoothing, callbacks, and early stopping.
 *
 * @param net - Network instance to train.
 * @param set - Training dataset.
 * @param options - Training options.
 * @returns Final training summary including error, iteration count, and elapsed time.
 */
export const trainFinalizeCore = (
  net: Network,
  set: { input: number[]; output: number[] }[],
  options: TrainingOptions,
): { error: number; iterations: number; time: number } => {
  const internalNet = net as unknown as NetworkInternals;
  options = options ?? {};
  const runtimeOptions = resolveTrainingRuntimeOptions(options);
  internalNet._accumulationReduction = runtimeOptions.accumulationReduction;

  validateTrainingOptions(
    net,
    set,
    options,
    runtimeOptions.cost,
    runtimeOptions.dropout,
    runtimeOptions.batchSize,
    runtimeOptions.accumulationSteps,
  );

  configureGradientClipping(internalNet, options.gradientClip);
  configureMixedPrecision(net, internalNet, options.mixedPrecision);

  const optimizerConfig = resolveOptimizerConfig(options.optimizer);
  const start = Date.now();
  let finalError = Infinity;
  const smoothing = buildSmoothingConfig(options);
  const earlyStopState: EarlyStopState = {
    bestError: Infinity,
    noImproveCount: 0,
  };

  net.dropout = runtimeOptions.dropout;
  let performedIterations = 0;

  for (let iteration = 1; iteration <= runtimeOptions.iterations; iteration++) {
    const iterationResult = runTrainingIteration(
      net,
      internalNet,
      set,
      iteration,
      options,
      runtimeOptions,
      optimizerConfig,
      smoothing,
    );

    performedIterations = iteration;
    finalError = iterationResult.monitoredError;

    applyTrainingCallbacks(
      iteration,
      finalError,
      iterationResult.plateauError,
      options,
      internalNet,
      net,
    );

    if (
      shouldEarlyStopNow(
        finalError,
        earlyStopState,
        options,
        runtimeOptions.targetError,
      )
    ) {
      break;
    }
  }

  finalizeTrainingRun(net, internalNet, performedIterations);

  return {
    error: finalError,
    iterations: performedIterations,
    time: Date.now() - start,
  };
};
