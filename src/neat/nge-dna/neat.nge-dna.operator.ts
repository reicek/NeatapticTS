/**
 * Canonical envelope-to-Network operator for the NGE (Neuro-Genesis Engine)
 * developmental pipeline.
 *
 * This operator turns a canonical `NgeDnaCanonicalEnvelope` into a live,
 * trainable {@link Network}. It is the narrow entry point that closes the gap
 * between the genetic encoding stored in an envelope and the runtime phenotype
 * that NeatapticTS can activate, evaluate, and serialize.
 *
 * The operator reads `envelope.reproductionPolicy.modeIsEvolvable` to decide
 * whether the materialized `Network` should carry the NGE extension carrier.
 * When the flag is `true`, the bridge attaches the NGE descriptor and envelope
 * bags and maps NGE squashes to their real activation functions. When the flag
 * is `false` (the default), the bridge produces a plain classic-NEAT network
 * with the NGE extension omitted and every squash collapsed to `identity`.
 *
 * The activation path reuses the deterministic developmental pipeline owned by
 * {@link NGE_DNA}:
 *
 * 1. `NGE_DNA.buildVirtualPlan(seed)` — deterministic module placement plan.
 * 2. `NGE_DNA.realizePhenotype(plan, seed)` — CPPN-evaluated phenotype descriptor.
 * 3. `materializeNetworkFromPhenotype(..., { ngeEnabled })` — Network hydration.
 *
 * Because the same envelope and seed always produce the same virtual plan and
 * the same realized phenotype descriptor, two activations of the same DNA
 * produce the same `Network.toJSON()` payload regardless of global counter drift.
 *
 * ## Opt-in isolation
 *
 * Classic-NEAT consumers are not required to opt into NGE. When
 * `modeIsEvolvable` is omitted or set to `false`, the function behaves exactly
 * like a classic materialization path: the returned network has no extension bag,
 * no NGE-specific squashes, and no hidden state. This preserves backward
 * compatibility for genomes and checkpoints that do not yet participate in the
 * NGE developmental pipeline.
 *
 * ## Neuromodulation primitive scoping
 *
 * `ModulatorBroadcaster`, `EpisodicSlot`, and `GatingRouter` are represented in
 * the realized phenotype descriptor, but the current bridge maps them to static
 * squashes (`identity` for the first two, `sigmoid` for the router) rather than
 * to stateful runtime primitives. True runtime activation — memory slots,
 * broadcast-radius governance, and dynamic gating selection — requires a
 * dedicated primitive substrate that is intentionally out of scope for this
 * operator.
 *
 * ```mermaid
 * flowchart LR
 *   classDef primitive fill:#08131f,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef squash fill:#0f2233,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef future fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
 *
 *   MB[ModulatorBroadcaster]:::primitive -->|descriptor-only| id1[identity squash]:::squash
 *   ES[EpisodicSlot]:::primitive -->|descriptor-only| id2[identity squash]:::squash
 *   GR[GatingRouter]:::primitive -->|descriptor-only| sig[sigmoid squash]:::squash
 *   substrate[Stateful runtime substrate]:::future -->|extension point| MB
 *   substrate -->|extension point| ES
 *   substrate -->|extension point| GR
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   classDef env fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef plan fill:#0f2233,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef pheno fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef bridge fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
 *   classDef net fill:#08131f,stroke:#06d6a0,color:#bff7e6,stroke-width:1.5px;
 *
 *   envelope["NgeDnaCanonicalEnvelope"]:::env
 *   gate{"modeIsEvolvable"}:::plan
 *   planStep["buildVirtualPlan(seed)"]:::plan
 *   real["realizePhenotype(plan, seed)"]:::plan
 *   desc["NgeRealizedPhenotypeDescriptor"]:::pheno
 *   mat["materializeNetworkFromPhenotype(ngeEnabled)"]:::bridge
 *   net["Network"]:::net
 *
 *   envelope --> gate
 *   gate -->|true| on["ngeEnabled: true"]:::plan
 *   gate -->|false / omitted| off["ngeEnabled: false"]:::plan
 *   envelope --> planStep
 *   planStep --> real
 *   real --> desc
 *   desc --> mat
 *   on --> mat
 *   off --> mat
 *   mat --> net
 * ```
 *
 * Background reading: the original NEAT paper by
 * [Stanley and Miikkulainen (2002)](https://nn.cs.utexas.edu/?stanley:ec02), and
 * the evo-devo overview on
 * [Wikipedia — Evolutionary developmental biology](https://en.wikipedia.org/wiki/Evolutionary_developmental_biology).
 *
 * @see {@link materializeNetworkFromPhenotype}
 * @see {@link NGE_DNA}
 */
import Network from '../../architecture/network/network';

import { materializeNetworkFromPhenotype } from './neat.nge-dna.bridge';
import { NGE_DNA } from './neat.nge-dna';
import type { NgeDnaCanonicalEnvelope } from './neat.nge-dna.types';

/**
 * Internal marker for the descriptor-only neuromodulation-primitive scoping note.
 *
 * `ModulatorBroadcaster`, `EpisodicSlot`, and `GatingRouter` are represented in
 * the phenotype descriptor, but the current phenotype→Network bridge maps them to
 * static squashes (`identity` for the first two, `sigmoid` for the router). True
 * runtime stateful behavior — memory slots, broadcast-radius governance, and
 * dynamic gating selection — requires a dedicated primitive substrate and is
 * intentionally not implemented here.
 *
 * @internal
 */
export const NGE_NEUROMODULATION_BLOCKER_ID = 'DR-2026-06-27-05-NM';

/**
 * Materialize one runtime {@link Network} from a canonical NGE DNA envelope.
 *
 * This is the canonical operator that closes the loop between the DNA envelope
 * and the runtime network. It derives the `ngeEnabled` materialization hint from
 * `envelope.reproductionPolicy.modeIsEvolvable`, so the same envelope can produce
 * either an NGE-enabled network (with extension carrier) or a plain classic-NEAT
 * network (no NGE surface) depending on the policy.
 *
 * The default value of `modeIsEvolvable` is `false`; omitting the field or
 * leaving it `false` therefore yields a classic-NEAT-compatible `Network`.
 *
 * The function is deterministic: the same envelope and seed always produce the
 * same virtual plan, the same realized phenotype descriptor, and therefore the
 * same `Network.toJSON()` output regardless of global counter drift.
 *
 * @param envelope - Canonical NGE DNA envelope whose `reproductionPolicy.modeIsEvolvable`
 *   selects the NGE evolution path.
 * @param seed - Deterministic seed folded into the virtual plan and phenotype realization.
 * @returns A runtime Network whose topology mirrors the realized phenotype descriptor.
 * @throws {NGE_DNA_BridgeError} when the realized phenotype descriptor has zero modules.
 * @throws {NGE_DNA_SchemaError} when the envelope identity or fingerprint is invalid.
 *
 * @example
 * NGE-enabled activation:
 * ```ts
 * const dna = new NGE_DNA({
 *   moduleArchetypes: [...],
 *   rulePasses: [...],
 *   reproductionPolicy: { modeIsEvolvable: true },
 * });
 * const network = activateNgeNetworkFromEnvelope(dna.toCanonical(), 42);
 * network.activate([1, 2, 3]);
 * ```
 *
 * @example
 * Classic-NEAT-compatible activation (the default):
 * ```ts
 * const dna = new NGE_DNA({
 *   moduleArchetypes: [...],
 *   rulePasses: [...],
 *   // modeIsEvolvable defaults to false
 * });
 * const network = activateNgeNetworkFromEnvelope(dna.toCanonical(), 7);
 * // network has no NGE extension bag and identity squashes.
 * ```
 */
export function activateNgeNetworkFromEnvelope(
  envelope: NgeDnaCanonicalEnvelope,
  seed: number,
): Network {
  // Step 1: Rehydrate the deterministic DNA instance from the canonical envelope.
  const dna = NGE_DNA.fromCanonical(envelope);

  // Step 2: Build the deterministic virtual module plan for this seed.
  const plan = dna.buildVirtualPlan(seed);

  // Step 3: Realize the phenotype descriptor (modules + CPPN-evaluated edges).
  const descriptor = dna.realizePhenotype(plan, seed);

  // Step 4: Derive the NGE opt-in flag from the envelope reproduction policy.
  const ngeEnabled = envelope.reproductionPolicy.modeIsEvolvable === true;

  // Step 5: Materialize the runtime Network through the canonical bridge.
  return materializeNetworkFromPhenotype(envelope, plan, descriptor, {
    ngeEnabled,
  });
}
