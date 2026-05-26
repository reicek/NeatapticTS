import { config } from '../../../config';
import type Network from '../../network/network';
import type {
  CostFunction,
  CostFunctionOrObject,
  OptimizerConfigBase,
  RegularizationConfig,
  TrainingNetworkInternals as NetworkInternals,
  TrainingNodeInternals as NodeInternals,
} from '../network.types';
import { applyGradientClippingCore } from './network.training.gradient-clip.utils';
import type {
  GradientClipRuntimeConfig,
  TrainingSample,
} from './network.training.utils.types';

/** Smallest magnitude that survives float16 subnormal storage. */
const FLOAT16_SUBNORMAL_MAGNITUDE_FLOOR = 2 ** -24;

/** Power-of-two loss-scale adjustments preserve exact rescaling steps. */
const LOSS_SCALE_ADJUSTMENT_FACTOR = 2;

const hasOnlyFiniteValues = (values: number[]): boolean =>
  values.every((value) => Number.isFinite(value));

type TrainingSampleProcessingResult = {
  batchSampleCountIncrement: number;
  errorContribution: number;
  processedSamplesIncrement: number;
};

const SKIPPED_TRAINING_SAMPLE_RESULT: TrainingSampleProcessingResult = {
  batchSampleCountIncrement: 0,
  errorContribution: 0,
  processedSamplesIncrement: 0,
};

const resolveCostFn = (
  costFunction: CostFunction | CostFunctionOrObject,
): ((target: number[], output: number[]) => number) => {
  if (typeof costFunction === 'function') {
    return costFunction;
  }

  if (
    typeof costFunction === 'object' &&
    costFunction !== null &&
    typeof (costFunction as CostFunctionOrObject).fn === 'function'
  ) {
    return (costFunction as CostFunctionOrObject).fn!;
  }

  if (
    typeof costFunction === 'object' &&
    costFunction !== null &&
    typeof (costFunction as CostFunctionOrObject).calculate === 'function'
  ) {
    return (costFunction as CostFunctionOrObject).calculate!;
  }

  return (): number => 0;
};

const propagateSampleNodes = (
  net: Network,
  outputNodes: NodeInternals[],
  target: number[],
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  optimizer?: OptimizerConfigBase,
): void => {
  const shouldDeferUpdate = Boolean(
    optimizer && optimizer.type && optimizer.type !== 'sgd',
  );

  for (let outIndex = 0; outIndex < outputNodes.length; outIndex++) {
    outputNodes[outIndex].propagate(
      currentRate,
      momentum,
      !shouldDeferUpdate,
      regularization,
      target[outIndex],
    );
  }

  for (
    let reverseIndex = net.nodes.length - 1;
    reverseIndex >= 0;
    reverseIndex--
  ) {
    const node = net.nodes[reverseIndex];
    if (node.type === 'output' || node.type === 'input') {
      continue;
    }

    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.propagate(
      currentRate,
      momentum,
      !shouldDeferUpdate,
      regularization,
    );
  }
};

const applyBatchBoundaryUpdate = (
  net: Network,
  internalNet: NetworkInternals,
  optimizer: OptimizerConfigBase | undefined,
  accumulationSteps: number,
  sampleIndex: number,
  setLength: number,
  currentRate: number,
  momentum: number,
  batchSampleCount: number,
): number => {
  if (!optimizer || !optimizer.type || optimizer.type === 'sgd') {
    return batchSampleCount;
  }

  internalNet._gradAccumMicroBatches++;
  const readyForStep =
    internalNet._gradAccumMicroBatches % accumulationSteps === 0 ||
    sampleIndex === setLength - 1;

  if (!readyForStep) {
    return 0;
  }

  internalNet._optimizerStep = (internalNet._optimizerStep || 0) + 1;
  const overflowDetected = detectMixedPrecisionOverflow(net, internalNet);

  if (overflowDetected) {
    zeroAccumulatedGradients(net);
    handleOverflow(internalNet);
    internalNet._lastGradNorm = 0;
    return 0;
  }

  if (internalNet._currentGradClip) {
    applyGradientClippingCore(
      net,
      internalNet._currentGradClip as GradientClipRuntimeConfig,
    );
  }

  if (
    accumulationSteps > 1 &&
    internalNet._accumulationReduction === 'average'
  ) {
    averageAccumulatedGradients(net, accumulationSteps);
  }

  const underflowDetected = detectMixedPrecisionUnderflow(net, internalNet);
  internalNet._lastGradNorm = applyOptimizerStep(
    net,
    optimizer,
    currentRate,
    momentum,
    internalNet,
  );

  if (internalNet._mixedPrecision.enabled) {
    maybeIncreaseLossScale(internalNet, underflowDetected);
  }

  return 0;
};

const processTrainingSample = (
  net: Network,
  sampleIndex: number,
  dataPoint: TrainingSample,
  computeError: (target: number[], output: number[]) => number,
  outputNodes: NodeInternals[],
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  optimizer?: OptimizerConfigBase,
): TrainingSampleProcessingResult => {
  const skipReason = resolveTrainingSampleSkipReason(net, dataPoint);

  if (skipReason) {
    warnSkippedTrainingSample(sampleIndex, skipReason);
    return SKIPPED_TRAINING_SAMPLE_RESULT;
  }

  return processFiniteTrainingSample(
    net,
    sampleIndex,
    dataPoint,
    computeError,
    outputNodes,
    currentRate,
    momentum,
    regularization,
    optimizer,
  );
};

const resolveTrainingSampleSkipReason = (
  net: Network,
  dataPoint: TrainingSample,
): string | undefined => {
  if (
    dataPoint.input.length !== net.input ||
    dataPoint.output.length !== net.output
  ) {
    return `has incorrect dimensions (input: ${dataPoint.input.length}/${net.input}, output: ${dataPoint.output.length}/${net.output}), skipping.`;
  }

  if (
    !hasOnlyFiniteValues(dataPoint.input) ||
    !hasOnlyFiniteValues(dataPoint.output)
  ) {
    return 'contains non-finite input or target values, skipping.';
  }

  return undefined;
};

const processFiniteTrainingSample = (
  net: Network,
  sampleIndex: number,
  dataPoint: TrainingSample,
  computeError: (target: number[], output: number[]) => number,
  outputNodes: NodeInternals[],
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  optimizer?: OptimizerConfigBase,
): TrainingSampleProcessingResult => {
  try {
    const output = activateTrainingSample(net, sampleIndex, dataPoint.input);

    if (!output) {
      return SKIPPED_TRAINING_SAMPLE_RESULT;
    }

    propagateSampleNodes(
      net,
      outputNodes,
      dataPoint.output,
      currentRate,
      momentum,
      regularization,
      optimizer,
    );

    return {
      batchSampleCountIncrement: 1,
      errorContribution: computeError(dataPoint.output, output),
      processedSamplesIncrement: 1,
    };
  } catch (error: unknown) {
    warnTrainingSampleFailure(sampleIndex, dataPoint.input, error);
    return SKIPPED_TRAINING_SAMPLE_RESULT;
  }
};

const activateTrainingSample = (
  net: Network,
  sampleIndex: number,
  input: number[],
): number[] | undefined => {
  const networkInternal = net as unknown as NetworkInternals;
  const output = networkInternal.activate(input, true);

  if (hasOnlyFiniteValues(output)) {
    return output;
  }

  warnSkippedTrainingSample(
    sampleIndex,
    'produced non-finite activation output, skipping.',
  );
  return undefined;
};

const warnSkippedTrainingSample = (
  sampleIndex: number,
  reason: string,
): void => {
  if (config.warnings) {
    console.warn(`Data point ${sampleIndex} ${reason}`);
  }
};

const warnTrainingSampleFailure = (
  sampleIndex: number,
  input: number[],
  error: unknown,
): void => {
  if (config.warnings) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.warn(
      `Error processing data point ${sampleIndex} (input: ${JSON.stringify(
        input,
      )}): ${errorMessage}. Skipping.`,
    );
  }
};

const shouldApplyBatchBoundaryUpdate = (
  batchSampleCount: number,
  sampleIndex: number,
  batchSize: number,
  setLength: number,
): boolean => {
  if (batchSampleCount <= 0) {
    return false;
  }

  return (sampleIndex + 1) % batchSize === 0 || sampleIndex === setLength - 1;
};

/**
 * Execute one dataset pass with mini-batching, accumulation, clipping, and optimizer updates.
 *
 * @param net - Network instance being trained.
 * @param set - Training sample set.
 * @param batchSize - Mini-batch size.
 * @param accumulationSteps - Micro-batches per optimizer step.
 * @param currentRate - Learning rate for this pass.
 * @param momentum - Momentum value used by propagation paths.
 * @param regularization - Regularization settings passed into propagation calls.
 * @param costFunction - Cost function or cost-function object.
 * @param optimizer - Optional optimizer configuration.
 * @returns Mean cost over processed samples.
 */
/**
 * Contract for trainSetCore.
 */
export const trainSetCore = (
  net: Network,
  set: TrainingSample[],
  batchSize: number,
  accumulationSteps: number,
  currentRate: number,
  momentum: number,
  regularization: RegularizationConfig,
  costFunction: CostFunction | CostFunctionOrObject,
  optimizer?: OptimizerConfigBase,
): number => {
  const internalNet = net as unknown as NetworkInternals;
  let cumulativeError = 0;
  let batchSampleCount = 0;
  internalNet._gradAccumMicroBatches = 0;
  let totalProcessedSamples = 0;
  const outputNodes = net.nodes
    .filter((node) => node.type === 'output')
    .map((node) => node as unknown as NodeInternals);
  const computeError = resolveCostFn(costFunction);

  for (let sampleIndex = 0; sampleIndex < set.length; sampleIndex++) {
    const sampleResult = processTrainingSample(
      net,
      sampleIndex,
      set[sampleIndex],
      computeError,
      outputNodes,
      currentRate,
      momentum,
      regularization,
      optimizer,
    );

    cumulativeError += sampleResult.errorContribution;
    batchSampleCount += sampleResult.batchSampleCountIncrement;
    totalProcessedSamples += sampleResult.processedSamplesIncrement;

    if (
      shouldApplyBatchBoundaryUpdate(
        batchSampleCount,
        sampleIndex,
        batchSize,
        set.length,
      )
    ) {
      batchSampleCount = applyBatchBoundaryUpdate(
        net,
        internalNet,
        optimizer,
        accumulationSteps,
        sampleIndex,
        set.length,
        currentRate,
        momentum,
        batchSampleCount,
      );
    }
  }

  if (internalNet._lastGradNorm == null) internalNet._lastGradNorm = 0;
  return totalProcessedSamples > 0
    ? cumulativeError / totalProcessedSamples
    : 0;
};

const detectMixedPrecisionOverflow = (
  net: Network,
  internalNet: NetworkInternals,
): boolean => {
  if (!internalNet._mixedPrecision.enabled) return false;
  if (internalNet._forceNextOverflow) {
    internalNet._forceNextOverflow = false;
    return true;
  }
  let overflow = false;
  net.nodes.forEach((node) => {
    const nodeInternal = node as unknown as NodeInternals;
    if (nodeInternal._fp32Bias !== undefined) {
      if (!Number.isFinite(nodeInternal.bias)) overflow = true;
    }
  });
  return overflow;
};

const detectMixedPrecisionUnderflow = (
  net: Network,
  internalNet: NetworkInternals,
): boolean => {
  if (!internalNet._mixedPrecision.enabled) return false;

  let hasFiniteNonZeroGradient = false;
  let maxScaledGradientMagnitude = 0;

  const observeGradient = (gradient: number | undefined): void => {
    if (typeof gradient !== 'number' || !Number.isFinite(gradient)) return;

    const absoluteGradient = Math.abs(gradient);
    if (absoluteGradient === 0) return;

    hasFiniteNonZeroGradient = true;
    maxScaledGradientMagnitude = Math.max(
      maxScaledGradientMagnitude,
      absoluteGradient * internalNet._mixedPrecision.lossScale,
    );
  };

  net.nodes.forEach((node) => {
    const nodeInternal = node as unknown as NodeInternals;

    nodeInternal.connections.in.forEach((connection) => {
      observeGradient(connection.totalDeltaWeight);
    });
    nodeInternal.connections.self.forEach((connection) => {
      observeGradient(connection.totalDeltaWeight);
    });
    observeGradient(nodeInternal.totalDeltaBias);
  });

  return (
    hasFiniteNonZeroGradient &&
    maxScaledGradientMagnitude < FLOAT16_SUBNORMAL_MAGNITUDE_FLOOR
  );
};

const zeroAccumulatedGradients = (net: Network): void => {
  net.nodes.forEach((node) => {
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.connections.in.forEach((connection) => {
      connection.totalDeltaWeight = 0;
    });
    nodeInternal.connections.self.forEach((connection) => {
      connection.totalDeltaWeight = 0;
    });
    if (typeof nodeInternal.totalDeltaBias === 'number') {
      nodeInternal.totalDeltaBias = 0;
    }
    nodeInternal.previousDeltaBias = 0;
  });
};

const averageAccumulatedGradients = (
  net: Network,
  accumulationSteps: number,
): void => {
  net.nodes.forEach((node) => {
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.connections.in.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        connection.totalDeltaWeight /= accumulationSteps;
      }
    });
    nodeInternal.connections.self.forEach((connection) => {
      if (typeof connection.totalDeltaWeight === 'number') {
        connection.totalDeltaWeight /= accumulationSteps;
      }
    });
    if (typeof nodeInternal.totalDeltaBias === 'number') {
      nodeInternal.totalDeltaBias /= accumulationSteps;
    }
  });
};

const applyOptimizerStep = (
  net: Network,
  optimizer: OptimizerConfigBase,
  currentRate: number,
  momentum: number,
  internalNet: NetworkInternals,
): number => {
  let sumSq = 0;
  net.nodes.forEach((node) => {
    if (node.type === 'input') return;
    const nodeInternal = node as unknown as NodeInternals;
    nodeInternal.applyBatchUpdatesWithOptimizer({
      type: optimizer.type,
      baseType: optimizer.baseType,
      beta1: optimizer.beta1,
      beta2: optimizer.beta2,
      eps: optimizer.eps,
      weightDecay: optimizer.weightDecay,
      momentum: optimizer.momentum ?? momentum,
      lrScale: currentRate,
      t: internalNet._optimizerStep,
      la_k: optimizer.la_k,
      la_alpha: optimizer.la_alpha,
    });
    nodeInternal.connections.in.forEach((connection) => {
      if (typeof connection.previousDeltaWeight === 'number') {
        sumSq +=
          connection.previousDeltaWeight * connection.previousDeltaWeight;
      }
    });
    nodeInternal.connections.self.forEach((connection) => {
      if (typeof connection.previousDeltaWeight === 'number') {
        sumSq +=
          connection.previousDeltaWeight * connection.previousDeltaWeight;
      }
    });
  });
  return Math.sqrt(sumSq);
};

const maybeIncreaseLossScale = (
  internalNet: NetworkInternals,
  underflowDetected: boolean,
): void => {
  if (underflowDetected) {
    internalNet._mixedPrecisionState.goodSteps = 0;
    internalNet._mixedPrecisionState.underflowCount =
      (internalNet._mixedPrecisionState.underflowCount || 0) + 1;
    internalNet._mixedPrecisionState.lastUnderflowStep =
      internalNet._optimizerStep;

    if (
      internalNet._mixedPrecision.lossScale <
      internalNet._mixedPrecisionState.maxLossScale
    ) {
      internalNet._mixedPrecision.lossScale = Math.min(
        internalNet._mixedPrecisionState.maxLossScale,
        internalNet._mixedPrecision.lossScale * LOSS_SCALE_ADJUSTMENT_FACTOR,
      );
      internalNet._mixedPrecisionState.scaleUpEvents =
        (internalNet._mixedPrecisionState.scaleUpEvents || 0) + 1;
    }

    return;
  }

  internalNet._mixedPrecisionState.goodSteps++;
  const increaseEvery = internalNet._mpIncreaseEvery || 200;
  if (
    internalNet._mixedPrecisionState.goodSteps >= increaseEvery &&
    internalNet._mixedPrecision.lossScale <
      internalNet._mixedPrecisionState.maxLossScale
  ) {
    internalNet._mixedPrecision.lossScale = Math.min(
      internalNet._mixedPrecisionState.maxLossScale,
      internalNet._mixedPrecision.lossScale * LOSS_SCALE_ADJUSTMENT_FACTOR,
    );
    internalNet._mixedPrecisionState.goodSteps = 0;
    internalNet._mixedPrecisionState.scaleUpEvents =
      (internalNet._mixedPrecisionState.scaleUpEvents || 0) + 1;
  }
};

const handleOverflow = (internalNet: NetworkInternals): void => {
  internalNet._mixedPrecisionState.badSteps++;
  internalNet._mixedPrecisionState.goodSteps = 0;
  internalNet._mixedPrecision.lossScale = Math.max(
    internalNet._mixedPrecisionState.minLossScale,
    Math.floor(
      internalNet._mixedPrecision.lossScale / LOSS_SCALE_ADJUSTMENT_FACTOR,
    ) || 1,
  );
  internalNet._mixedPrecisionState.overflowCount =
    (internalNet._mixedPrecisionState.overflowCount || 0) + 1;
  internalNet._mixedPrecisionState.scaleDownEvents =
    (internalNet._mixedPrecisionState.scaleDownEvents || 0) + 1;
  internalNet._lastOverflowStep = internalNet._optimizerStep;
};
