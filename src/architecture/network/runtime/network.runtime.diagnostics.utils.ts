import type Network from '../network';
import type Node from '../../node';

import type { NetworkRuntimeDiagnosticsInternals } from '../network.types';
import { NetworkRuntimeDropConnectProbabilityRangeError } from './network.runtime.errors';

type TrainingStatsSnapshot = {
  gradNorm: number;
  gradNormRaw: number;
  lossScale: number;
  optimizerStep: number;
  mp: {
    good: number;
    bad: number;
    overflowCount: number;
    scaleUps: number;
    scaleDowns: number;
    lastOverflowStep: number;
  };
};

/**
 * Runtime diagnostics and safety helpers for the public `Network` class.
 *
 * This chapter owns the public readers and small runtime controls that expose
 * training-health state, DropConnect policy, and dropout-mask cleanup without
 * changing the network topology itself.
 */

/**
 * Enable DropConnect with a probability in $[0,1)$.
 *
 * @param this Target network instance.
 * @param probability DropConnect probability.
 * @returns Nothing.
 */
export function enableDropConnect(this: Network, probability: number): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;

  if (probability < 0 || probability >= 1) {
    throw new NetworkRuntimeDropConnectProbabilityRangeError(
      'DropConnect probability must be in [0,1)',
    );
  }

  runtimeNetwork._dropConnectProb = probability;
}

/**
 * Disable DropConnect.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function disableDropConnect(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;
  runtimeNetwork._dropConnectProb = 0;
}

/**
 * Reset every dropout mask to `1`.
 *
 * This is useful after training so later inference does not inherit transient
 * node-level dropout state from a previous activation pass.
 *
 * @param this Target network instance.
 * @returns Nothing.
 */
export function resetDropoutMasks(this: Network): void {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;

  if (runtimeNetwork.layers && runtimeNetwork.layers.length > 0) {
    for (const layer of runtimeNetwork.layers) {
      resetMaskCollection(layer.nodes);
    }

    return;
  }

  resetMaskCollection(runtimeNetwork.nodes);

  /**
   * Reset one node collection in place.
   *
   * @param nodes Nodes whose masks should be restored.
   * @returns Nothing.
   */
  function resetMaskCollection(nodes: Node[]): void {
    for (const node of nodes) {
      if (typeof node.mask !== 'undefined') {
        node.mask = 1;
      }
    }
  }
}

/**
 * Read the last recorded raw gradient norm.
 *
 * @param this Target network instance.
 * @returns Last raw gradient norm.
 */
export function getRawGradientNorm(this: Network): number {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;
  return runtimeNetwork._lastRawGradNorm;
}

/**
 * Read the active mixed-precision loss scale.
 *
 * @param this Target network instance.
 * @returns Current loss scale.
 */
export function getLossScale(this: Network): number {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;
  return runtimeNetwork._mixedPrecision.lossScale;
}

/**
 * Read the last recorded gradient-clipping group count.
 *
 * @param this Target network instance.
 * @returns Last gradient-clipping group count.
 */
export function getLastGradClipGroupCount(this: Network): number {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;
  return runtimeNetwork._lastGradClipGroupCount;
}

/**
 * Read a consolidated training-health snapshot.
 *
 * @param this Target network instance.
 * @returns Training statistics snapshot.
 */
export function getTrainingStats(this: Network): TrainingStatsSnapshot {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;

  return {
    gradNorm: runtimeNetwork._lastGradNorm ?? 0,
    gradNormRaw: runtimeNetwork._lastRawGradNorm,
    lossScale: runtimeNetwork._mixedPrecision.lossScale,
    optimizerStep: runtimeNetwork._optimizerStep,
    mp: {
      good: runtimeNetwork._mixedPrecisionState.goodSteps,
      bad: runtimeNetwork._mixedPrecisionState.badSteps,
      overflowCount: runtimeNetwork._mixedPrecisionState.overflowCount ?? 0,
      scaleUps: runtimeNetwork._mixedPrecisionState.scaleUpEvents ?? 0,
      scaleDowns: runtimeNetwork._mixedPrecisionState.scaleDownEvents ?? 0,
      lastOverflowStep: runtimeNetwork._lastOverflowStep,
    },
  };
}
