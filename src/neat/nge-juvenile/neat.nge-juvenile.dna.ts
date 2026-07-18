/**
 * NGE DNA governance translator for the grow-stabilize cycle.
 *
 * This module bridges the canonical NGE DNA envelope to the grow-stabilize
 * configuration surface. It reads optional `governance.growStabilize` overrides
 * from the DNA envelope and merges them with caller-supplied defaults, so
 * DNA-level governance knobs always take priority over runtime defaults
 * without hardcoding any values into the translator itself.
 *
 * ## Design rationale
 *
 * The translator is intentionally a thin, pure function:
 *
 * - It does not mutate the DNA envelope or the defaults object.
 * - It does not import or call any core algorithm code.
 * - It does not hardcode defaults; every fallback value comes from the
 *   caller-supplied `defaults` parameter.
 * - It preserves the `Partial<NgeGrowStabilizeConfig>` contract: fields absent
 *   from both sources are simply omitted from the result.
 *
 * This keeps the DNA → config boundary inspectable, testable, and free of
 * hidden side effects, which is critical for deterministic development
 * reproducibility (see `reproducibility-contracts` skill).
 *
 * ## Background reading
 *
 * - NGE DNA as a compact program rather than an explicit graph:
 *   see `nge-core-algorithm` skill.
 * - Governance overlays and morph policy knobs:
 *   see `plans/NGE_Grow_Stabilize_Cycle.plans.md` slice B7.
 *
 * ```mermaid
 * flowchart LR
 *   DNA["NgeDnaCanonicalEnvelope"] --> Gov{"governance.growStabilize?"}
 *   Gov -- present --> Merge["Merge defaults + DNA overrides"]
 *   Gov -- absent --> Passthrough["Return defaults"]
 *   Merge --> Config["Partial<NgeGrowStabilizeConfig>"]
 *   Passthrough --> Config
 * ```
 */

import type { NgeDnaCanonicalEnvelope } from '../nge-dna/neat.nge-dna.types';
import type { NgeGrowStabilizeConfig } from './neat.nge-juvenile.types';

/**
 * Governance overlay carried by a canonical NGE DNA envelope.
 *
 * Maps stage schedules, budgets, wiring-cost preferences, and morph policy
 * knobs to canonical NGE_DNA schema fields via the `growStabilize` shelf.
 * Each field is optional so DNA envelopes that do not specify governance
 * remain valid.
 *
 * @property growStabilize - Grow-stabilize config overrides sourced from DNA governance.
 */
export interface NgeDnaGovernance {
  /** Grow-stabilize config overrides sourced from DNA governance. */
  readonly growStabilize?: Partial<NgeGrowStabilizeConfig>;
}

/**
 * A canonical NGE DNA envelope with an optional governance overlay.
 *
 * The base `NgeDnaCanonicalEnvelope` does not declare a `governance` field.
 * This helper type extends it so the translator can safely access the
 * optional governance shelf that DNA envelopes may carry.
 */
type NgeDnaEnvelopeWithGovernance = NgeDnaCanonicalEnvelope & {
  /** Optional governance overlay for stage schedules, budgets, and morph policy. */
  readonly governance?: NgeDnaGovernance;
};

/**
 * Translate NGE DNA governance overrides into a grow-stabilize config.
 *
 * Reads the optional `governance.growStabilize` overlay on the DNA envelope
 * and merges it with caller-supplied defaults. DNA governance values take
 * priority over defaults for any field present in the governance block.
 * Fields absent from both sources are omitted from the result, preserving
 * the `Partial<NgeGrowStabilizeConfig>` contract.
 *
 * All fallback values come from the `defaults` parameter — the translator
 * never hardcodes config values. This ensures that the caller (typically the
 * lifecycle runner or a higher-level orchestrator) controls every default,
 * while DNA governance can override individual knobs without rewriting the
 * entire config.
 *
 * @param dna - Canonical NGE DNA envelope, optionally carrying a governance overlay.
 * @param defaults - Caller-supplied default values for grow-stabilize config fields.
 * @returns Merged grow-stabilize config with DNA governance overrides applied.
 *
 * @example
 * ```ts
 * const config = translateDnaToGrowStabilizeConfig(dna, {
 *   maxStructuralEditsPerStep: 5,
 *   maxNodes: 8_000,
 *   maxConnections: 32_000,
 *   maxEpisodicSlots: 15,
 *   moduleId: 'nge:runtime',
 * });
 * ```
 */
export function translateDnaToGrowStabilizeConfig(
  dna: NgeDnaCanonicalEnvelope,
  defaults: Partial<NgeGrowStabilizeConfig>,
): Partial<NgeGrowStabilizeConfig> {
  // Step 1: Safely extract the optional governance overlay from the DNA envelope.
  const dnaOverrides = extractGrowStabilizeGovernance(dna);

  // Step 2: If DNA provides no overrides, return a shallow copy of the defaults.
  if (dnaOverrides === undefined) {
    return { ...defaults };
  }

  // Step 3: Merge defaults with DNA overrides — DNA values take priority.
  return { ...defaults, ...dnaOverrides };
}

/**
 * Extract the optional `growStabilize` governance shelf from a DNA envelope.
 *
 * Returns `undefined` when the envelope carries no governance or when the
 * governance shelf has no `growStabilize` field. This keeps the merge logic
 * in the orchestrator clean and makes the governance access inspectable.
 *
 * @param dna - Canonical NGE DNA envelope, optionally carrying a governance overlay.
 * @returns The grow-stabilize overrides from DNA governance, or `undefined`.
 */
function extractGrowStabilizeGovernance(
  dna: NgeDnaCanonicalEnvelope,
): Partial<NgeGrowStabilizeConfig> | undefined {
  const governance = (dna as NgeDnaEnvelopeWithGovernance).governance;
  return governance?.growStabilize;
}
