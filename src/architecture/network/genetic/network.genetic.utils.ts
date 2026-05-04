/**
 * Network-level crossover boundary for recombining two compatible parent
 * graphs into one offspring.
 *
 * This chapter is the genetic shelf of `architecture/network/`: the place
 * where structure is inherited rather than mutated from scratch. It answers a
 * practical question that shows up during neuroevolution and testing alike: if
 * two networks already expose the same input and output contract, how should a
 * child network mix their node and connection genes without losing a runnable
 * topology?
 *
 * The helpers below split that answer into setup, genome-backed selection, and
 * materialization. Setup decides how large the offspring can be and which
 * parent has inheritance priority. The genome heredity boundary chooses
 * overlapping and disjoint genes from strict structural contracts.
 * Materialization turns the chosen genes back into a concrete `Network`
 * instance and restores gating when the chosen gater still exists. That keeps
 * the public `crossOver()` surface compact while the README can still teach the
 * full inheritance flow.
 *
 * ```mermaid
 * flowchart LR
 *   ParentA[Parent A] --> Context[Build crossover context]
 *   ParentB[Parent B] --> Context
 *   Context --> Nodes[Assign offspring nodes]
 *   Nodes --> Genes[Choose connection genes]
 *   Genes --> Offspring[Materialize offspring network]
 * ```
 *
 * Required teaching output: old crossover vs proper crossover.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   subgraph Old[Legacy mental model, approximation]
 *     direction TB
 *     oldA[Parent A connections]:::base --> oldKey[Key by endpoint index pair\nfrom index, to index]:::accent
 *     oldB[Parent B connections]:::base --> oldKey
 *     oldKey --> oldMatch["Match" if endpoints look the same]:::base
 *   end
 *
 *   subgraph Proper[Proper-NEAT alignment, historical markings]
 *     direction TB
 *     newA[Parent A connections]:::base --> innov[Key by innovation number\nconnection innovation]:::accent
 *     newB[Parent B connections]:::base --> innov
 *     innov --> newMatch[Match / disjoint / excess by innovation id]:::base
 *   end
 *
 *   Old --> offs[Offspring gene set]:::base
 *   Proper --> offs
 * ```
 *
 * This chapter now aligns inherited connection genes by preserved innovation
 * numbers instead of synthetic endpoint ids, which moves the runtime crossover
 * boundary much closer to canonical NEAT historical markings. The remaining
 * simplifications are now narrower: node inheritance still follows runtime slot
 * ordering, and recurrent-policy enforcement still happens later during
 * materialization.
 *
 * For compact background reading on the wider idea, see Wikipedia
 * contributors,
 * [Crossover (genetic algorithm)](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)).
 * The implementation here specializes that idea to graph-shaped neural
 * networks with gating and disabled genes.
 *
 * Example: create one offspring using ordinary fitness-biased inheritance.
 *
 * ```ts
 * const child = crossOver(parentA, parentB);
 * ```
 *
 * Example: force symmetric inheritance when you want experimentation rather
 * than fitter-parent bias.
 *
 * ```ts
 * const exploratoryChild = crossOver(parentA, parentB, true);
 * ```
 */
import type Network from '../../network/network';
import { inheritTemporalDescriptorExtensions } from '../network.temporal.extensions.utils';
import { materializeOffspringConnections } from './network.genetic.materialize.utils';
import {
  assignOffspringNodes,
  chooseOffspringConnectionGenes,
  createCrossoverContext,
  createNodeBuildContext,
} from './network.genetic.setup.utils';
import type { RandomGenerator } from './network.genetic.utils.types';

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
 * Current simplifications relative to canonical NEAT:
 *  - Node alignment still relies on current index ordering while the broader
 *    proper-NEAT lift keeps the public runtime `Network` surface stable.
 *  - Recurrent and self-connection legality is still finalized during
 *    materialization rather than by a separate genotype-first heredity layer.
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
 *  5. Delegate innovation-keyed connection-gene collection and inheritance
 *     choice to the genome heredity boundary.
 *  6. For matching genes (present in both parents with the same innovation),
 *     randomly choose one; if either copy is disabled, apply the explicit
 *     re-enable policy through the crossover RNG.
 *  7. For disjoint/excess genes, inherit only from the fitter parent (or from
 *     both parents when `equal` is enabled or scores tie).
 *  8. Rebuild the offspring node set from required IO nodes plus inherited
 *     gene identities, then materialize selected connection genes under the
 *     offspring topology intent.
 *  9. Reattach gating if gater node exists in offspring.
 *
 * Enabled reactivation probability:
 *  - Parents may carry disabled connections; offspring may re-enable them with a probability derived
 *    from parent-specific _reenableProb (or default 0.25). This allows dormant structures to resurface.
 *
 * @param parentNetwork1 First parent (ties resolved in its favor when scores equal and equal=false for some cases).
 * @param parentNetwork2 Second parent.
 * @param equal Force symmetric treatment regardless of fitness (true => node count random between sizes and both parents equally contribute disjoint genes).
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
  return runCrossOver(parentNetwork1, parentNetwork2, equal);
}

/**
 * Internal crossover entrypoint that uses an explicit RNG supplied by the NEAT evolve layer.
 *
 * The public `Network.crossOver()` surface stays convenience-oriented so standalone callers can
 * keep relying on the runtime-network fallback. The evolve controller uses this helper instead
 * when crossover randomness must stay in the same deterministic stream as parent selection.
 *
 * @param parentNetwork1 First parent network.
 * @param parentNetwork2 Second parent network.
 * @param equal Equal-treatment mode flag.
 * @param randomGenerator Explicit crossover RNG owned by the evolve controller.
 * @returns Offspring network instance.
 */
export function crossOverWithRandomGenerator(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal: boolean = false,
  randomGenerator: RandomGenerator,
): Network {
  return runCrossOver(parentNetwork1, parentNetwork2, equal, randomGenerator);
}

function runCrossOver(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal: boolean,
  randomGenerator?: RandomGenerator,
): Network {
  // Step 1: Build immutable crossover context.
  const crossoverContext = createCrossoverContext(
    parentNetwork1,
    parentNetwork2,
    equal,
    randomGenerator,
  );

  // Step 2: Select and assign offspring node genes.
  const nodeBuildContext = createNodeBuildContext(crossoverContext);
  assignOffspringNodes(nodeBuildContext);

  // Step 3: Select and materialize offspring connection genes.
  const chosenGenes = chooseOffspringConnectionGenes(crossoverContext);
  materializeOffspringConnections(crossoverContext.offspring, chosenGenes, [
    crossoverContext.parent1,
    crossoverContext.parent2,
  ]);

  // Step 4: Preserve any parent temporal descriptors that still match the offspring structure.
  inheritTemporalDescriptorExtensions(crossoverContext.offspring, [
    crossoverContext.parentNetwork1,
    crossoverContext.parentNetwork2,
  ]);

  // Step 5: Return the finalized offspring.
  return crossoverContext.offspring;
}

export default { crossOver, crossOverWithRandomGenerator };
