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

  options = options ?? {};
  if (
    typeof options.iterations === 'undefined' &&
    typeof options.error === 'undefined'
  ) {
    if (config.warnings)
      console.warn('Missing `iterations` or `error` option.');
    throw new NetworkTrainingStoppingConditionRequiredError(
      'Missing `iterations` or `error` option. Training requires a stopping condition.',
    );
  }
  if (config.warnings) {
    if (typeof options.rate === 'undefined') {
      console.warn('Missing `rate` option');
      console.warn('Missing `rate` option, using default learning rate 0.3.');
    }
    if (typeof options.iterations === 'undefined')
      console.warn(
        'Missing `iterations` option. Training will run potentially indefinitely until `error` threshold is met.',
      );
  }

  const targetError = options.error ?? -Infinity;
  const cost = options.cost ?? methods.Cost.mse;
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

  const baseRate = options.rate ?? 0.3;
  const dropout = options.dropout ?? 0;
  if (!Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    throw new NetworkTrainingDropoutRangeError('dropout must be in [0,1)');
  }

  const momentum = options.momentum || 0;
  const batchSize = options.batchSize || 1;
  if (batchSize > set.length) {
    throw new NetworkTrainingBatchSizeError(
      'Batch size cannot be larger than the dataset length.',
    );
  }

  const accumulationSteps = options.accumulationSteps || 1;
  internalNet._accumulationReduction =
    options.accumulationReduction === 'sum' ? 'sum' : 'average';
  if (accumulationSteps < 1 || !Number.isFinite(accumulationSteps)) {
    throw new NetworkTrainingAccumulationStepsError(
      'accumulationSteps must be >=1',
    );
  }

  if (options.gradientClip) {
    const gradientClipConfig = options.gradientClip;
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
  } else {
    internalNet._currentGradClip = undefined;
    internalNet._gradClipSeparateBias = false;
  }

  if (options.mixedPrecision) {
    const mixedPrecisionConfig =
      options.mixedPrecision === true
        ? { lossScale: 1024 }
        : options.mixedPrecision;
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
  } else {
    internalNet._mixedPrecision.enabled = false;
    internalNet._mixedPrecision.lossScale = 1;
    internalNet._mpIncreaseEvery = 200;
  }

  let optimizerConfig: OptimizerConfigBase | undefined = undefined;
  if (typeof options.optimizer !== 'undefined') {
    if (typeof options.optimizer === 'string') {
      optimizerConfig = { type: options.optimizer.toLowerCase() };
    } else if (
      typeof options.optimizer === 'object' &&
      options.optimizer !== null
    ) {
      optimizerConfig = { ...options.optimizer };
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
  }

  const iterations = options.iterations ?? Number.MAX_SAFE_INTEGER;
  const start = Date.now();
  let finalError = Infinity;

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

  const monitoredSmoothingConfig = buildMonitoredSmoothingConfig(
    movingAverageType,
    movingAverageWindow,
    emaAlpha,
    options.trimmedRatio,
  );
  const plateauSmoothingConfig: PlateauSmoothingConfig = {
    type: plateauType,
    window: plateauWindow,
    emaAlpha: plateauEmaAlpha,
  };

  const earlyStopPatience = options.earlyStopPatience;
  const earlyStopMinDelta = options.earlyStopMinDelta || 0;
  let bestError = Infinity;
  let noImproveCount = 0;

  const recentErrorsCapacity = movingAverageWindow;
  const recentErrorsBuf: number[] = new Array(recentErrorsCapacity);
  let recentErrorsCount = 0;
  let recentErrorsWriteIdx = 0;
  const recentErrorsPush = (value: number): void => {
    if (recentErrorsCapacity === 1) {
      recentErrorsBuf[0] = value;
      recentErrorsCount = 1;
      recentErrorsWriteIdx = 0;
      return;
    }
    recentErrorsBuf[recentErrorsWriteIdx] = value;
    recentErrorsWriteIdx = (recentErrorsWriteIdx + 1) % recentErrorsCapacity;
    if (recentErrorsCount < recentErrorsCapacity) recentErrorsCount++;
  };
  const recentErrorsChrono = (): number[] => {
    if (recentErrorsCount === 0) return [];
    if (recentErrorsCount < recentErrorsCapacity) {
      return recentErrorsBuf.slice(0, recentErrorsCount);
    }
    const orderedErrors = new Array(recentErrorsCount);
    const startIndex = recentErrorsWriteIdx;
    for (let sampleIndex = 0; sampleIndex < recentErrorsCount; sampleIndex++) {
      orderedErrors[sampleIndex] =
        recentErrorsBuf[(startIndex + sampleIndex) % recentErrorsCapacity];
    }
    return orderedErrors;
  };

  const primarySmoothingState: PrimarySmoothingState = {
    emaValue: undefined,
    adaptiveBaseEmaValue: undefined,
    adaptiveEmaValue: undefined,
  };

  const plateauCapacity = plateauWindow;
  const plateauBuf: number[] = new Array(plateauCapacity);
  let plateauCount = 0;
  let plateauWriteIdx = 0;
  const plateauPush = (value: number): void => {
    if (plateauCapacity === 1) {
      plateauBuf[0] = value;
      plateauCount = 1;
      plateauWriteIdx = 0;
      return;
    }
    plateauBuf[plateauWriteIdx] = value;
    plateauWriteIdx = (plateauWriteIdx + 1) % plateauCapacity;
    if (plateauCount < plateauCapacity) plateauCount++;
  };
  const plateauChrono = (): number[] => {
    if (plateauCount === 0) return [];
    if (plateauCount < plateauCapacity) {
      return plateauBuf.slice(0, plateauCount);
    }
    const orderedErrors = new Array(plateauCount);
    const startIndex = plateauWriteIdx;
    for (let sampleIndex = 0; sampleIndex < plateauCount; sampleIndex++) {
      orderedErrors[sampleIndex] =
        plateauBuf[(startIndex + sampleIndex) % plateauCapacity];
    }
    return orderedErrors;
  };

  const plateauSmoothingState: PlateauSmoothingState = {
    plateauEmaValue: undefined,
  };

  net.dropout = dropout;
  let performedIterations = 0;

  for (let iteration = 1; iteration <= iterations; iteration++) {
    if (internalNet._maybePrune) {
      internalNet._maybePrune((internalNet._globalEpoch || 0) + iteration);
    }

    const trainError = trainSetCore(
      net,
      set,
      batchSize,
      accumulationSteps,
      baseRate,
      momentum,
      {},
      cost as CostFunction | CostFunctionOrObject,
      optimizerConfig,
    );

    performedIterations = iteration;
    recentErrorsPush(trainError);

    const recentErrors = recentErrorsChrono();
    const monitored = computeMonitoredError(
      trainError,
      recentErrors,
      monitoredSmoothingConfig,
      primarySmoothingState,
    );
    finalError = monitored;

    plateauPush(trainError);
    const plateauError = computePlateauMetric(
      trainError,
      plateauChrono(),
      plateauSmoothingConfig,
      plateauSmoothingState,
    );

    if (typeof options.metricsHook === 'function') {
      try {
        options.metricsHook({
          iteration,
          error: finalError,
          plateauError,
          gradNorm: internalNet._lastGradNorm ?? 0,
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
      if (options.checkpoint.best) {
        if (
          finalError < internalNet._checkpointBestError! ||
          internalNet._checkpointBestError == null
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

    if (finalError < bestError - earlyStopMinDelta) {
      bestError = finalError;
      noImproveCount = 0;
    } else if (earlyStopPatience) {
      noImproveCount++;
    }

    if (earlyStopPatience && noImproveCount >= earlyStopPatience) break;
    if (finalError <= targetError) break;
  }

  net.nodes.forEach((node) => {
    if (node.type === 'hidden') node.mask = 1;
  });
  net.dropout = 0;
  internalNet._globalEpoch =
    (internalNet._globalEpoch || 0) + performedIterations;

  return {
    error: finalError,
    iterations: performedIterations,
    time: Date.now() - start,
  };
};
