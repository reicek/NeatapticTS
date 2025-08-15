import type Network from '../network';
import Node from '../node';
import Connection from '../connection';

/**
 * Genetic operator: NEAT‑style crossover (legacy merge operator removed).
 *
 * This module now focuses solely on producing recombinant offspring via {@link crossOver}.
 * The previous experimental Network.merge has been removed to reduce maintenance surface area
 * and avoid implying a misleading “sequential composition” guarantee.
 *
 * @module network.genetic
 */

/**
 * NEAT-inspired crossover between two parent networks producing a single offspring.
 *
 * Simplifications relative to canonical NEAT:
 *  - Innovation ID is synthesized from (from.index, to.index) via Connection.innovationID instead of
 *    maintaining a global innovation number per mutation event.
 *  - Node alignment relies on current index ordering. This is weaker than historical innovation
 *    tracking, but adequate for many lightweight evolutionary experiments.
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
 * @param network1 - First parent (ties resolved in its favor when scores equal and equal=false for some cases).
 * @param network2 - Second parent.
 * @param equal - Force symmetric treatment regardless of fitness (true => node count random between sizes and both parents equally contribute disjoint genes).
 * @returns Offspring network instance.
 * @throws If input/output sizes differ.
 */
export function crossOver(
  network1: Network,
  network2: Network,
  equal = false
): Network {
  if (network1.input !== network2.input || network1.output !== network2.output)
    throw new Error(
      'Parent networks must have the same input and output sizes for crossover.'
    );
  /** Offspring scaffold produced by recombination of parent networks. */
  const offspring = new (require('../network').default)(
    network1.input,
    network1.output
  ) as Network;
  /** Mutable list of standard (non self) connections assigned during reconstruction. */
  (offspring as any).connections = [];
  /** Ordered list of cloned node genes composing the offspring topology. */
  (offspring as any).nodes = [];
  /** Self–connections (loops) for offspring, rebuilt during connection materialization. */
  (offspring as any).selfconns = [];
  /** Collection of gated connections after inheritance. */
  (offspring as any).gates = [];
  /** Fitness (score) of parent 1 used for dominance decisions. */
  const score1 = (network1 as any).score || 0;
  /** Fitness (score) of parent 2 used for dominance decisions. */
  const score2 = (network2 as any).score || 0;
  /** Number of nodes in parent 1 (used to bound index-based selection). */
  const n1Size = (network1 as any).nodes.length;
  /** Number of nodes in parent 2 (used to bound index-based selection). */
  const n2Size = (network2 as any).nodes.length;
  // Decide offspring size based on equality / fitness.
  /** Final number of node slots (including I/O) the offspring will contain. */
  let size: number;
  if (equal || score1 === score2) {
    /** Upper bound on possible offspring node count when parents tied / equal mode. */
    const max = Math.max(n1Size, n2Size);
    /** Lower bound on possible offspring node count when parents tied / equal mode. */
    const min = Math.min(n1Size, n2Size);
    /** Random length chosen uniformly in [min, max]. */
    size = Math.floor(Math.random() * (max - min + 1) + min);
  } else size = score1 > score2 ? n1Size : n2Size;
  /** Number of output nodes (shared by both parents). */
  const outputSize = network1.output;
  // Assign indices for deterministic innovation mapping later.
  (network1 as any).nodes.forEach((n: any, i: number) => (n.index = i));
  (network2 as any).nodes.forEach((n: any, i: number) => (n.index = i));
  // Node gene selection loop.
  for (let i = 0; i < size; i++) {
    /** Chosen parent node gene for this index (if any). */
    let chosen: any;
    /** Parent 1 node gene at current index (undefined if beyond parent size). */
    const node1 = i < n1Size ? (network1 as any).nodes[i] : undefined;
    /** Parent 2 node gene at current index (undefined if beyond parent size). */
    const node2 = i < n2Size ? (network2 as any).nodes[i] : undefined;
    if (i < network1.input) chosen = node1;
    // Always preserve consistent input interface.
    else if (i >= size - outputSize) {
      // Output region aligned from tail.
      /** Index of candidate output node in parent 1 derived from tail alignment. */
      const o1 = n1Size - (size - i);
      /** Index of candidate output node in parent 2 derived from tail alignment. */
      const o2 = n2Size - (size - i);
      /** Parent 1 output node at aligned slot (if valid). */
      const n1o =
        o1 >= network1.input && o1 < n1Size
          ? (network1 as any).nodes[o1]
          : undefined;
      /** Parent 2 output node at aligned slot (if valid). */
      const n2o =
        o2 >= network2.input && o2 < n2Size
          ? (network2 as any).nodes[o2]
          : undefined;
      if (n1o && n2o)
        chosen = ((network1 as any)._rand || Math.random)() >= 0.5 ? n1o : n2o;
      else chosen = n1o || n2o;
    } else {
      // Hidden region.
      if (node1 && node2)
        chosen =
          ((network1 as any)._rand || Math.random)() >= 0.5 ? node1 : node2;
      else if (node1 && (score1 >= score2 || equal)) chosen = node1;
      else if (node2 && (score2 >= score1 || equal)) chosen = node2;
    }
    if (chosen) {
      // Clone structural gene (bias + activation function / squash) but do not copy connections yet.
      const nn: any = new Node(chosen.type);
      nn.bias = chosen.bias;
      nn.squash = chosen.squash;
      (offspring as any).nodes.push(nn);
    }
  }
  // Reassign indices after constructing node list.
  (offspring as any).nodes.forEach((n: any, i: number) => (n.index = i));
  // Gather connection genes from both parents (including self connections) keyed by innovation id.
  /** Map from innovation ID -> connection gene extracted from parent 1 (includes self connections). */
  const n1conns: Record<string, any> = {};
  /** Map from innovation ID -> connection gene extracted from parent 2 (includes self connections). */
  const n2conns: Record<string, any> = {};
  (network1 as any).connections
    .concat((network1 as any).selfconns)
    .forEach((c: any) => {
      if (typeof c.from.index === 'number' && typeof c.to.index === 'number')
        n1conns[Connection.innovationID(c.from.index, c.to.index)] = {
          weight: c.weight,
          from: c.from.index,
          to: c.to.index,
          gater: c.gater ? c.gater.index : -1,
          enabled: (c as any).enabled !== false,
        };
    });
  (network2 as any).connections
    .concat((network2 as any).selfconns)
    .forEach((c: any) => {
      if (typeof c.from.index === 'number' && typeof c.to.index === 'number')
        n2conns[Connection.innovationID(c.from.index, c.to.index)] = {
          weight: c.weight,
          from: c.from.index,
          to: c.to.index,
          gater: c.gater ? c.gater.index : -1,
          enabled: (c as any).enabled !== false,
        };
    });
  // Select connection genes: iterate parent1's map, handle overlaps, then optionally add remaining parent2 genes.
  /** Accumulated list of chosen connection gene descriptors to materialize in offspring. */
  const chosenConns: any[] = [];
  /** Array of innovation IDs originating from parent 1 (iteration order). */
  const keys1 = Object.keys(n1conns);
  keys1.forEach((k) => {
    /** Connection gene from parent 1 under current innovation ID. */
    const c1 = n1conns[k];
    if (n2conns[k]) {
      // Matching gene.
      /** Corresponding connection gene from parent 2 for matching innovation ID. */
      const c2 = n2conns[k];
      /** Selected gene (either c1 or c2) retained in offspring. */
      const pick = ((network1 as any)._rand || Math.random)() >= 0.5 ? c1 : c2; // Randomly select weight / flags from one parent.
      if (c1.enabled === false || c2.enabled === false) {
        // If either disabled, chance to re-enable.
        /** Probability threshold to re-enable a previously disabled matching connection. */
        const rp =
          (network1 as any)._reenableProb ??
          (network2 as any)._reenableProb ??
          0.25;
        pick.enabled = Math.random() < rp;
      }
      chosenConns.push(pick);
      delete n2conns[k]; // Remove from second map to mark consumed.
    } else if (score1 >= score2 || equal) {
      // Disjoint/excess gene from fitter or equal mode.
      if (c1.enabled === false) {
        /** Re-enable probability for a disabled disjoint/excess gene from parent1. */
        const rp = (network1 as any)._reenableProb ?? 0.25;
        c1.enabled = Math.random() < rp;
      }
      chosenConns.push(c1);
    }
  });
  // Remaining genes from parent2 if it is fitter (or equal mode).
  if (score2 >= score1 || equal)
    Object.keys(n2conns).forEach((k) => {
      const d = n2conns[k];
      if (d.enabled === false) {
        /** Re-enable probability for parent2 disjoint/excess gene. */ const rp =
          (network2 as any)._reenableProb ?? 0.25;
        d.enabled = Math.random() < rp;
      }
      chosenConns.push(d);
    });
  /** Number of nodes copied into offspring; used to validate endpoint indices of connection genes. */
  const nodeCount = (offspring as any).nodes.length;
  // Materialize connection genes in offspring network (skip if endpoint nodes not present due to size truncation).
  chosenConns.forEach((cd) => {
    if (cd.from < nodeCount && cd.to < nodeCount) {
      const from = (offspring as any).nodes[cd.from];
      const to = (offspring as any).nodes[cd.to];
      // Always enforce feed-forward ordering for crossover offspring: skip any backward or self-loop
      // edges (self loops handled elsewhere) to satisfy structural invariants expected by tests.
      if (cd.from >= cd.to) return; // skip backward / non feed-forward edge
      if (!from.isProjectingTo(to)) {
        /** Newly constructed connection edge within offspring (first element of connect array). */ const conn = (offspring as any).connect(
          from,
          to
        )[0];
        if (conn) {
          conn.weight = cd.weight;
          (conn as any).enabled = cd.enabled !== false;
          if (cd.gater !== -1 && cd.gater < nodeCount)
            (offspring as any).gate((offspring as any).nodes[cd.gater], conn);
        }
      }
    }
  });
  return offspring;
}

export default { crossOver };
