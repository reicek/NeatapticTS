import type Network from '../network';
import type Node from '../../node';

import type {
  ActivationSchedule,
  ActivationSchedulingDiagnostics,
  NetworkRuntimeDiagnosticsInternals,
} from '../network.types';
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
 * training-health state, activation-ordering diagnostics, DropConnect policy,
 * and dropout-mask cleanup without changing the network topology itself.
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

/**
 * Read a human-friendly snapshot of the current activation-ordering contract.
 *
 * The snapshot explains whether activation is using a compiled schedule or a
 * fallback path, whether topology is currently dirty, and what callers should
 * do next when cycles or stale caches prevent the preferred schedule.
 *
 * @param this Target network instance.
 * @returns Activation scheduling diagnostics snapshot.
 */
export function getActivationSchedulingDiagnostics(
  this: Network,
): ActivationSchedulingDiagnostics {
  const runtimeNetwork = this as unknown as NetworkRuntimeDiagnosticsInternals;
  const cachedDiagnostics =
    runtimeNetwork._activationSchedulingDiagnostics ??
    createDefaultSchedulingDiagnostics(this, runtimeNetwork);

  if (!runtimeNetwork._topoDirty) {
    return cloneSchedulingDiagnostics(cachedDiagnostics);
  }

  return cloneSchedulingDiagnostics({
    ...cachedDiagnostics,
    topologyDirty: true,
    message: `${cachedDiagnostics.message} Topology is currently dirty, so the next activation will rebuild scheduling state before it runs.`,
    suggestions: appendSuggestion(
      cachedDiagnostics.suggestions,
      'Run activate() or noTraceActivate() after structural edits to refresh the compiled scheduling cache.',
    ),
  });
}

/**
 * Build a safe default diagnostics snapshot when no explicit scheduling record exists yet.
 *
 * @param network Target network instance.
 * @param runtimeNetwork Runtime diagnostics internals.
 * @returns Default scheduling diagnostics snapshot.
 */
function createDefaultSchedulingDiagnostics(
  network: Network,
  runtimeNetwork: NetworkRuntimeDiagnosticsInternals,
): ActivationSchedulingDiagnostics {
  const activationSchedule = runtimeNetwork._activationSchedule;

  if (activationSchedule) {
    return createCompiledSchedulingDiagnostics(network, activationSchedule);
  }

  return {
    topologyIntent: network.getTopologyIntent(),
    requestedMode: runtimeNetwork._enforceAcyclic ? 'acyclic' : 'recurrent',
    topologyDirty: Boolean(runtimeNetwork._topoDirty),
    executionPath: 'raw-node-order',
    issue: 'schedule-missing',
    message:
      'No compiled activation schedule is cached yet, so execution will use raw node order until topology is rebuilt.',
    inputNodeIds: network.inputNodeIds,
    outputNodeIds: network.outputNodeIds,
    stepCount: 0,
    recurrentComponentCount: 0,
    stateSemantics: null,
    cycleNodeIds: [],
    suggestions: [
      'Run activate() or noTraceActivate() to rebuild scheduling state after topology changes.',
    ],
  };
}

/**
 * Build the standard diagnostics snapshot for a compiled activation schedule.
 *
 * @param network Target network instance.
 * @param activationSchedule Cached compiled schedule.
 * @returns Scheduling diagnostics snapshot.
 */
function createCompiledSchedulingDiagnostics(
  network: Network,
  activationSchedule: ActivationSchedule,
): ActivationSchedulingDiagnostics {
  const recurrentComponentCount = activationSchedule.steps.filter(
    (activationStep) => activationStep.kind === 'recurrent-component',
  ).length;

  return {
    topologyIntent: network.getTopologyIntent(),
    requestedMode: activationSchedule.mode,
    topologyDirty: false,
    executionPath: 'compiled-schedule',
    issue: null,
    message:
      activationSchedule.mode === 'acyclic'
        ? 'Activation is using the compiled acyclic schedule with stable wave ordering.'
        : 'Activation is using the compiled recurrent schedule with explicit recurrent-component steps and carried recurrent state.',
    inputNodeIds: network.inputNodeIds,
    outputNodeIds: activationSchedule.outputNodeIds,
    stepCount: activationSchedule.steps.length,
    recurrentComponentCount,
    stateSemantics: activationSchedule.stateSemantics ?? null,
    cycleNodeIds: [],
    suggestions:
      activationSchedule.mode === 'recurrent'
        ? [
            'Call clear() before a new independent sequence when carried recurrent state should reset.',
          ]
        : [],
  };
}

/**
 * Clone the diagnostics snapshot so callers cannot mutate runtime state.
 *
 * @param diagnostics Diagnostics snapshot to clone.
 * @returns Detached diagnostics snapshot.
 */
function cloneSchedulingDiagnostics(
  diagnostics: ActivationSchedulingDiagnostics,
): ActivationSchedulingDiagnostics {
  return {
    ...diagnostics,
    inputNodeIds: [...diagnostics.inputNodeIds],
    outputNodeIds: [...diagnostics.outputNodeIds],
    cycleNodeIds: [...diagnostics.cycleNodeIds],
    suggestions: [...diagnostics.suggestions],
  };
}

/**
 * Append one suggestion string only when it is not already present.
 *
 * @param suggestions Existing suggestions.
 * @param suggestion Suggested next action.
 * @returns Updated suggestions list.
 */
function appendSuggestion(
  suggestions: string[],
  suggestion: string,
): string[] {
  if (suggestions.includes(suggestion)) {
    return [...suggestions];
  }

  return [...suggestions, suggestion];
}
