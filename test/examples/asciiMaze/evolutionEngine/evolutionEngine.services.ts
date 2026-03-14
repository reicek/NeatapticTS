import {
  engineState,
  setBaldwinPhaseDisabledFlag,
  setReducedTelemetryFlag,
  setTelemetryMinimalFlag,
} from './engineState';
import {
  EVOLUTION_ENGINE_INITIAL_LOGITS_RING_CAPACITY,
  EVOLUTION_ENGINE_MAX_LOGITS_RING_CAPACITY,
} from './evolutionEngine.constants';
import type { EngineState } from './engineState.types';
import type {
  EvolutionEngineFacadeRuntimeState,
  LogitsRingState,
} from './evolutionEngine.types';

const evolutionEngineFacadeRuntimeState: EvolutionEngineFacadeRuntimeState = {
  logitsRingCap: EVOLUTION_ENGINE_INITIAL_LOGITS_RING_CAPACITY,
  logitsRingShared: false,
  scratchLogitsRingW: 0,
};

/**
 * Return the shared engine singleton used by extracted engine modules.
 *
 * The public facade now depends on the same owner as the rest of the engine
 * boundary instead of creating a private duplicate singleton.
 *
 * @returns Shared engine state singleton.
 */
export const getEvolutionEngineSharedState = (): EngineState => engineState;

/**
 * Read the mutable facade-owned logits-ring runtime state.
 *
 * @returns Current ring-capacity, shared-mode, and write-cursor state.
 */
export const getEvolutionEngineFacadeRuntimeState =
  (): EvolutionEngineFacadeRuntimeState => {
    return evolutionEngineFacadeRuntimeState;
  };

/**
 * Apply the latest logits-ring runtime values returned by the evolution loop.
 *
 * @param updatedRingState - New ring-capacity, shared-mode, and write-cursor values.
 */
export const applyEvolutionEngineRingState = (
  updatedRingState: LogitsRingState,
): void => {
  evolutionEngineFacadeRuntimeState.logitsRingCap =
    updatedRingState.logitsRingCap;
  evolutionEngineFacadeRuntimeState.logitsRingShared =
    updatedRingState.logitsRingShared;
  evolutionEngineFacadeRuntimeState.scratchLogitsRingW =
    updatedRingState.scratchLogitsRingW;
};

/**
 * Reset the facade-owned logits-ring runtime state to its baseline defaults.
 */
export const resetEvolutionEngineRingState = (): void => {
  evolutionEngineFacadeRuntimeState.logitsRingCap =
    EVOLUTION_ENGINE_INITIAL_LOGITS_RING_CAPACITY;
  evolutionEngineFacadeRuntimeState.logitsRingShared = false;
  evolutionEngineFacadeRuntimeState.scratchLogitsRingW = 0;
};

/**
 * Read the hard maximum ring capacity used by the public facade.
 *
 * @returns Maximum ring capacity allowed for logits telemetry.
 */
export const getEvolutionEngineMaxLogitsRingCapacity = (): number => {
  return EVOLUTION_ENGINE_MAX_LOGITS_RING_CAPACITY;
};

/**
 * Apply telemetry and Baldwin-phase toggles derived from one normalized run request.
 *
 * @param reducedTelemetry - When true, keep only the essential telemetry metrics.
 * @param telemetryMinimal - When true, disable verbose telemetry capture.
 * @param disableBaldwinPhase - When true, skip the Baldwin refinement stage.
 */
export const configureEvolutionEngineToggles = (
  reducedTelemetry: boolean,
  telemetryMinimal: boolean,
  disableBaldwinPhase: boolean,
): void => {
  setReducedTelemetryFlag(reducedTelemetry);
  setTelemetryMinimalFlag(telemetryMinimal);
  setBaldwinPhaseDisabledFlag(disableBaldwinPhase);
};