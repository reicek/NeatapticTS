import { config } from '../../../config';
import type Network from '../../network';
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
  const outputNodes = net.nodes.filter((node) => node.type === 'output');
  let computeError: (t: number[], o: number[]) => number;

  if (typeof costFunction === 'function') {
    computeError = costFunction;
  } else if (
    typeof costFunction === 'object' &&
    costFunction !== null &&
    typeof (costFunction as CostFunctionOrObject).fn === 'function'
  ) {
    computeError = (costFunction as CostFunctionOrObject).fn!;
  } else if (
    typeof costFunction === 'object' &&
    costFunction !== null &&
    typeof (costFunction as CostFunctionOrObject).calculate === 'function'
  ) {
    computeError = (costFunction as CostFunctionOrObject).calculate!;
  } else {
    computeError = (): number => 0;
  }

  for (let sampleIndex = 0; sampleIndex < set.length; sampleIndex++) {
    const dataPoint = set[sampleIndex];
    const input = dataPoint.input;
    const target = dataPoint.output;
    if (input.length !== net.input || target.length !== net.output) {
      if (config.warnings)
        console.warn(
          `Data point ${sampleIndex} has incorrect dimensions (input: ${input.length}/${net.input}, output: ${target.length}/${net.output}), skipping.`,
        );
      continue;
    }

    try {
      const networkInternal = net as unknown as NetworkInternals;
      const output = networkInternal.activate(input, true);
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        for (let outIndex = 0; outIndex < outputNodes.length; outIndex++) {
          const outputNodeInternal = outputNodes[
            outIndex
          ] as unknown as NodeInternals;
          outputNodeInternal.propagate(
            currentRate,
            momentum,
            false,
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
          if (node.type === 'output' || node.type === 'input') continue;
          const nodeInternal = node as unknown as NodeInternals;
          nodeInternal.propagate(currentRate, momentum, false, regularization);
        }
      } else {
        for (let outIndex = 0; outIndex < outputNodes.length; outIndex++) {
          const outputNodeInternal = outputNodes[
            outIndex
          ] as unknown as NodeInternals;
          outputNodeInternal.propagate(
            currentRate,
            momentum,
            true,
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
          if (node.type === 'output' || node.type === 'input') continue;
          const nodeInternal = node as unknown as NodeInternals;
          nodeInternal.propagate(currentRate, momentum, true, regularization);
        }
      }
      cumulativeError += computeError(target, output);
      batchSampleCount++;
      totalProcessedSamples++;
    } catch (error: unknown) {
      if (config.warnings) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        console.warn(
          `Error processing data point ${sampleIndex} (input: ${JSON.stringify(
            input,
          )}): ${errorMessage}. Skipping.`,
        );
      }
    }

    if (
      batchSampleCount > 0 &&
      ((sampleIndex + 1) % batchSize === 0 || sampleIndex === set.length - 1)
    ) {
      if (optimizer && optimizer.type && optimizer.type !== 'sgd') {
        internalNet._gradAccumMicroBatches++;
        const readyForStep =
          internalNet._gradAccumMicroBatches % accumulationSteps === 0 ||
          sampleIndex === set.length - 1;
        if (readyForStep) {
          internalNet._optimizerStep = (internalNet._optimizerStep || 0) + 1;
          const overflowDetected = detectMixedPrecisionOverflow(
            net,
            internalNet,
          );
          if (overflowDetected) {
            zeroAccumulatedGradients(net);
            if (internalNet._mixedPrecision.enabled) {
              handleOverflow(internalNet);
            }
            internalNet._lastGradNorm = 0;
          } else {
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
            internalNet._lastGradNorm = applyOptimizerStep(
              net,
              optimizer,
              currentRate,
              momentum,
              internalNet,
            );
            if (internalNet._mixedPrecision.enabled) {
              maybeIncreaseLossScale(internalNet);
            }
          }
        }
        batchSampleCount = 0;
      }
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
  if (accumulationSteps <= 1) return;
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

const maybeIncreaseLossScale = (internalNet: NetworkInternals): void => {
  internalNet._mixedPrecisionState.goodSteps++;
  const increaseEvery = internalNet._mpIncreaseEvery || 200;
  if (
    internalNet._mixedPrecisionState.goodSteps >= increaseEvery &&
    internalNet._mixedPrecision.lossScale <
      internalNet._mixedPrecisionState.maxLossScale
  ) {
    internalNet._mixedPrecision.lossScale *= 2;
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
    Math.floor(internalNet._mixedPrecision.lossScale / 2) || 1,
  );
  internalNet._mixedPrecisionState.overflowCount =
    (internalNet._mixedPrecisionState.overflowCount || 0) + 1;
  internalNet._mixedPrecisionState.scaleDownEvents =
    (internalNet._mixedPrecisionState.scaleDownEvents || 0) + 1;
  internalNet._lastOverflowStep = internalNet._optimizerStep;
};
