import { NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY } from './neat.nge-evolution.constants';
import type {
  NgeEvolutionEpigeneticPriorInput,
  NgeEvolutionEpigeneticPriorResult,
} from './neat.nge-evolution.types';

const NO_APPLIED_DECAY = 0;

/**
 * Apply the optional birth-time epigenetic prior to one child parameter vector.
 *
 * When no two-parent reference is configured, the operator stays a strict no-op and
 * returns the original child vector reference without allocating any parameter shelves.
 * When configured, it deterministically blends both parent references, applies the
 * pending mutation delta, and adds the weak decay-scaled nudge toward that blend.
 *
 * @param input - Child parameter state, pending mutation delta, and optional parent reference vectors.
 * @returns The final child vector plus the resolved decay metadata for telemetry or tests.
 *
 * @example
 * ```ts
 * const result = applyNgeEvolutionEpigeneticPrior({
 *   childParameters: [1, 2],
 *   mutationDelta: [0.5, -0.25],
 *   reference: {
 *     firstParentParameters: [2, 4],
 *     secondParentParameters: [0, 6],
 *     blendedReference: [1, 5],
 *   },
 *   decay: 0.2,
 * });
 * ```
 */
export function applyNgeEvolutionEpigeneticPrior<
  ParameterVector extends readonly number[] = readonly number[],
>(
  input: NgeEvolutionEpigeneticPriorInput<ParameterVector>,
): NgeEvolutionEpigeneticPriorResult<ParameterVector> {
  if (input.reference === null) {
    return {
      outputParameters: input.childParameters,
      appliedDecay: NO_APPLIED_DECAY,
      referenceApplied: false,
      blendedReference: null,
    };
  }

  // Step 1: Resolve the weak two-parent anchor used by the birth-time nudge.
  const blendedReference = input.childParameters.map(
    (childParameter, parameterIndex) =>
      blendParentReferenceParameter(
        input.reference!.firstParentParameters,
        input.reference!.secondParentParameters,
        input.reference!.blendedReference,
        childParameter,
        parameterIndex,
      ),
  ) as unknown as ParameterVector;
  const appliedDecay = input.decay ?? NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY;

  // Step 2: Apply mutation plus the decay-scaled pull toward the blended anchor.
  const outputParameters = input.childParameters.map(
    (childParameter, parameterIndex) =>
      childParameter +
      (input.mutationDelta.at(parameterIndex) ?? 0) +
      appliedDecay * (blendedReference[parameterIndex]! - childParameter),
  ) as unknown as ParameterVector;

  return {
    outputParameters,
    appliedDecay,
    referenceApplied: true,
    blendedReference,
  };
}

function blendParentReferenceParameter(
  firstParentParameters: readonly number[],
  secondParentParameters: readonly number[],
  precomputedBlend: readonly number[],
  childParameter: number,
  parameterIndex: number,
): number {
  const fallbackParameter =
    precomputedBlend.at(parameterIndex) ?? childParameter;
  const firstParentParameter =
    firstParentParameters.at(parameterIndex) ?? fallbackParameter;
  const secondParentParameter =
    secondParentParameters.at(parameterIndex) ?? fallbackParameter;

  return (firstParentParameter + secondParentParameter) / 2;
}
