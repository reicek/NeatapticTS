/**
 * Experimental Genesis EvoDevo (NGE) public namespace.
 *
 * This barrel exposes a narrow, unstable preview of NGE lifecycle primitives
 * for early integration and benchmarking. The sub-namespaces are not covered
 * by the stable public API contract and may change or be removed without a
 * major version bump.
 *
 * The four re-exported sub-surfaces are:
 *
 * - `adult` — adult-stage lifecycle staging and equilibrium transitions.
 * - `juvenile` — juvenile growth and development before equilibrium.
 * - `lifecycle` — the runner that sequences juvenile growth, adult staging,
 *   and assimilation write-back.
 * - `assimilation` — deterministic write-back of an equilibrium candidate
 *   into a realized phenotype.
 *
 * Genesis EvoDevo (NGE) takes its shape from evolutionary developmental
 * biology: form emerges from regulated developmental programs rather than a
 * fixed parts list. See Wikipedia contributors,
 * [Evolutionary developmental biology](https://en.wikipedia.org/wiki/Evolutionary_developmental_biology),
 * for background on the evo-devo ideas that inform the lifecycle framing.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef experimental fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
 *
 *   nge[nge namespace]:::experimental
 *   nge --> juvenile[juvenile growth]:::base
 *   nge --> lifecycle[lifecycle runner]:::accent
 *   nge --> adult[adult staging]:::base
 *   nge --> assimilation[assimilation write-back]:::base
 *   juvenile --> lifecycle
 *   adult --> lifecycle
 *   lifecycle --> assimilation
 * ```
 *
 * @example
 * ```ts
 * import { nge } from 'neataptic';
 *
 * // Access the lifecycle runner and adult helper from the experimental namespace.
 * // This surface is unstable and intended for early benchmarking only.
 * const { runNgeLifecycle } = nge.lifecycle;
 * const { advanceAdultState } = nge.adult;
 * console.log(typeof runNgeLifecycle, typeof advanceAdultState);
 * ```
 *
 * @experimental
 * @module neat/nge-experimental
 */
/**
 * Adult-stage NGE lifecycle staging and equilibrium transitions.
 *
 * This namespace is experimental and unstable; it is not covered by the stable
 * public API contract and may change without a major version bump.
 */
export * as adult from './nge-adult/neat.nge-adult';

/**
 * Juvenile growth and development namespace for NGE.
 *
 * This namespace is experimental and unstable; it is not covered by the stable
 * public API contract and may change without a major version bump.
 */
export * as juvenile from './nge-juvenile/neat.nge-juvenile';

/**
 * NGE lifecycle runner that sequences juvenile growth, adult staging, and
 * assimilation write-back.
 *
 * This namespace is experimental and unstable; it is not covered by the stable
 * public API contract and may change without a major version bump.
 */
export * as lifecycle from './neat.nge-lifecycle';

/**
 * Deterministic assimilation write-back of an equilibrium candidate into a
 * realized NGE phenotype.
 *
 * This namespace is experimental and unstable; it is not covered by the stable
 * public API contract and may change without a major version bump.
 */
export * as assimilation from './nge-assimilation/neat.nge-assimilation';
