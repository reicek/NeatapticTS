import type Network from '../../network';
import { materializeOffspringConnections } from './network.genetic.materialize.utils';
import {
  assignOffspringNodes,
  chooseOffspringConnectionGenes,
  createCrossoverContext,
  createNodeBuildContext,
} from './network.genetic.setup.utils';

/**
 * Genetic operator: NEAT‑style crossover (legacy merge operator removed).
 *
 * This module now focuses solely on producing recombinant offspring via {@link crossOver}.
 * The previous experimental `Network.merge` flow has been removed to reduce maintenance
 * surface area and avoid implying a misleading sequential-composition guarantee.
 *
 * Design notes:
 * - The implementation favors deterministic, inspectable orchestration at the top level.
 * - Gene-selection details are delegated to setup/materialization helpers so the public
 *   crossover API stays compact and predictable.
 * - The resulting offspring preserves the same input/output interface as both parents,
 *   which keeps downstream evaluation and training pipelines compatible.
 *
 * @module network.genetic
 */

/**
 * NEAT-inspired crossover between two parent networks producing a single offspring.
 *
 * Conceptual model:
 * - A "gene" corresponds to either a node choice at a structural index or a connection
 *   keyed by innovation identity.
 * - The offspring is assembled in two phases: node assignment first, then connection
 *   materialization constrained by available offspring endpoints.
 * - Fitness controls inheritance pressure unless `equal` is enabled, in which case both
 *   parents contribute symmetrically where possible.
 *
 * Simplifications relative to canonical NEAT:
 *  - Innovation ID is synthesized from (from.index, to.index) via Connection.innovationID instead of
 *    maintaining a global innovation number per mutation event.
 *  - Node alignment relies on current index ordering. This is weaker than historical innovation
 *    tracking, but adequate for many lightweight evolutionary experiments.
 *
 * Compatibility assumptions:
 * - Both parents must expose identical input/output counts.
 * - Parent node index ordering should represent comparable structural positions.
 * - Parent fitness scores are interpreted by setup helpers when deciding fitter-parent inheritance.
 *
 * High-level algorithm:
 *  1. Validate that parents have identical I/O dimensionality (required for compatibility).
 *  2. Decide offspring node array length:
 *       - If equal flag set or scores tied: random length in [minNodes, maxNodes].
 *       - Else: length of fitter parent.
 *  3. For each index up to chosen size, pick a node gene from parents per rules:
 *       - Input indices: always from parent1 (assumes identical input interface).
 *       - Output indices (aligned from end): randomly choose if both present else take existing.
 *       - Hidden indices: if both present pick randomly; else inherit from fitter (or either if equal).
 *  4. Reindex offspring nodes.
 *  5. Collect connections (standard + self) from each parent into maps keyed by innovationID capturing
 *     weight, enabled flag, and gater index.
 *  6. For overlapping genes (present in both), randomly choose one; if either disabled apply optional
 *     re-enable probability (reenableProb) to possibly re-activate.
 *  7. For disjoint/excess genes, inherit only from fitter parent (or both if equal flag set / scores tied).
 *  8. Materialize selected connection genes if their endpoints both exist in offspring; set weight & enabled state.
 *  9. Reattach gating if gater node exists in offspring.
 *
 * Enabled reactivation probability:
 *  - Parents may carry disabled connections; offspring may re-enable them with a probability derived
 *    from parent-specific _reenableProb (or default 0.25). This allows dormant structures to resurface.
 *
 * @param parentNetwork1 - First parent (ties resolved in its favor when scores equal and equal=false for some cases).
 * @param parentNetwork2 - Second parent.
 * @param equal - Force symmetric treatment regardless of fitness (true => node count random between sizes and both parents equally contribute disjoint genes).
 * @returns Offspring network instance.
 * @throws If input/output sizes differ.
 *
 * @example
 * ```ts
 * const offspring = crossOver(parentA, parentB);
 * offspring.mutate();
 * ```
 */
export function crossOver(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal = false,
): Network {
  // Step 1: Build immutable crossover context.
  const crossoverContext = createCrossoverContext(
    parentNetwork1,
    parentNetwork2,
    equal,
  );

  // Step 2: Select and assign offspring node genes.
  const nodeBuildContext = createNodeBuildContext(crossoverContext);
  assignOffspringNodes(nodeBuildContext);

  // Step 3: Select and materialize offspring connection genes.
  const chosenGenes = chooseOffspringConnectionGenes(crossoverContext);
  materializeOffspringConnections(crossoverContext.offspring, chosenGenes);

  // Step 4: Return the finalized offspring.
  return crossoverContext.offspring;
}

export default { crossOver };
