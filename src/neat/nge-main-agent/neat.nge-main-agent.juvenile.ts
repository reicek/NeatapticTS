import type {
  NgeMainAgentAdult,
  NgeMainAgentEmbryo,
  NgeMainAgentJuvenile,
  NgeMainAgentLifecycleConfig,
} from './neat.nge-main-agent.types';
import type { NgeAssimilationCandidate } from '../nge-assimilation/neat.nge-assimilation.types';
import {
  writeInternalAssimilationPriors,
  type InternalAssimilationResult,
} from '../nge-assimilation/neat.nge-assimilation.internal';
import { DEFAULT_ASSIMILATION_WRITE_BACK_RATE } from '../nge-assimilation/neat.nge-assimilation.constants';

/**
 * Grow the embryo topology into a juvenile state while staying within budget.
 *
 * Juvenile growth is deterministic and never shrinks the network below the
 * embryo size. The growth factor is derived from the embryo's seed so that the
 * same embryo always produces the same juvenile.
 *
 * @param embryo - Embryo state to grow.
 * @param config - Lifecycle config with the tier budget cap.
 * @returns Juvenile state with node and edge counts within the tier budget.
 */
export function growJuvenileTopology(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentJuvenile {
  const growthFactor = 1 + Math.min(0.2, embryo.seed / 1_000_000);
  const nodeCount = Math.min(
    config.maxNodes,
    Math.max(embryo.nodeCount, Math.floor(embryo.nodeCount * growthFactor)),
  );
  const edgeCount = Math.min(
    config.maxEdges,
    Math.max(embryo.edgeCount, Math.floor(embryo.edgeCount * growthFactor)),
  );

  return {
    stage: 'juvenile',
    generation: embryo.generation,
    seed: config.seed,
    nodeCount,
    edgeCount,
    archetypes: embryo.archetypes,
    schemaVersion: embryo.schemaVersion,
  };
}

/**
 * Mature a juvenile or embryo state into an adult state.
 *
 * The transition is a deterministic stage change that preserves the input
 * topology. Adult pruning is handled separately by {@link pruneAdultTopology}.
 *
 * @param juvenile - Juvenile or embryo state to mature.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Adult state with the same topology as the input.
 */
export function transitionJuvenileToAdult(
  juvenile: NgeMainAgentJuvenile | NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult {
  return {
    stage: 'adult',
    generation: juvenile.generation,
    seed: config.seed,
    nodeCount: juvenile.nodeCount,
    edgeCount: juvenile.edgeCount,
    archetypes: juvenile.archetypes,
    schemaVersion: juvenile.schemaVersion,
  };
}

/**
 * Minimum number of juvenile generations between grow passes.
 *
 * The deterministic cooldown is anchored by {@link MIN_JUVENILE_GROW_COOLDOWN}
 * and varies slightly with the lifecycle seed so different lineages do not
 * lock-step through the same cadence.
 */
const MIN_JUVENILE_GROW_COOLDOWN = 2;

/**
 * Decay factor applied to internal assimilation updates during the juvenile
 * stage.
 *
 * A value below 1.0 keeps the write-back weak and self-limiting, preventing a
 * single equilibrium snapshot from dominating the juvenile genome.
 */
const DEFAULT_JUVENILE_ASSIMILATION_DECAY = 0.9;

/**
 * Result of a combined juvenile grow and internal assimilation pass.
 *
 * The result surfaces whether the topology actually grew this generation and
 * any structural priors that were weakly written back from the main agent's
 * own equilibrium candidate.
 *
 * @example
 * ```ts
 * const pass: JuvenileGrowPassResult = growJuvenileTopologyWithInternalAssimilation(
 *   embryo,
 *   config,
 *   candidate,
 * );
 * expect(pass.didGrow).toBe(true);
 * ```
 */
export interface JuvenileGrowPassResult {
  /** Juvenile state after the grow gate, with or without size change. */
  juvenile: NgeMainAgentJuvenile;
  /** True when the topology was allowed to grow this generation. */
  didGrow: boolean;
  /** Internal assimilation result when an equilibrium candidate is supplied. */
  assimilation: InternalAssimilationResult | null;
}

/**
 * Evaluate the deterministic hysteresis grow gate for a juvenile embryo.
 *
 * The gate opens every `MIN_JUVENILE_GROW_COOLDOWN + (seed % 3)` generations,
 * starting from generation 0. This creates a predictable but lineage-specific
 * cadence that avoids synchronized population-wide growth bursts.
 *
 * @param embryo - Embryo state carrying the current generation and seed.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns True when the juvenile is permitted to grow this generation.
 *
 * @example
 * ```ts
 * const canGrow = evaluateJuvenileGrowGate(embryo, { seed: 7, maxNodes: 64, maxEdges: 256 });
 * expect(canGrow).toBe(embryo.generation === 0 || embryo.generation % 4 === 0);
 * ```
 */
export function evaluateJuvenileGrowGate(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): boolean {
  const cooldown = MIN_JUVENILE_GROW_COOLDOWN + (config.seed % 3);
  return embryo.generation === 0 || embryo.generation % cooldown === 0;
}

/**
 * Grow the juvenile topology and, if available, assimilate internal priors.
 *
 * This orchestration applies the hysteresis grow gate first. When the gate is
 * open, {@link growJuvenileTopology} is used exactly as defined. When the gate is
 * closed, a minimal juvenile snapshot is produced so the stage contract is
 * preserved without forcing a topology change. If an equilibrium candidate is
 * provided, weak, decaying structural priors are written back to the main
 * agent's own genome.
 *
 * @param embryo - Embryo state to grow.
 * @param config - Lifecycle config with the tier budget cap and seed.
 * @param candidate - Optional equilibrium candidate from the main agent's own
 *   adult boundary. Enemy-derived fields are ignored by internal assimilation.
 * @returns Juvenile state plus grow-gate and internal-assimilation metadata.
 *
 * @example
 * ```ts
 * const pass = growJuvenileTopologyWithInternalAssimilation(embryo, config, candidate);
 * expect(pass.juvenile.stage).toBe('juvenile');
 * expect(pass.assimilation?.enemyWeightsIncorporated).toBe(false);
 * ```
 */
export function growJuvenileTopologyWithInternalAssimilation(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
  candidate?: NgeAssimilationCandidate,
): JuvenileGrowPassResult {
  const didGrow = evaluateJuvenileGrowGate(embryo, config);
  const juvenile = didGrow
    ? growJuvenileTopology(embryo, config)
    : buildMinimalJuvenileTopology(embryo, config);

  const assimilation =
    candidate === undefined
      ? null
      : writeInternalAssimilationPriors({
          candidate,
          writeBackRate: DEFAULT_ASSIMILATION_WRITE_BACK_RATE,
          decay: DEFAULT_JUVENILE_ASSIMILATION_DECAY,
        });

  return { juvenile, didGrow, assimilation };
}

function buildMinimalJuvenileTopology(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentJuvenile {
  return {
    stage: 'juvenile',
    generation: embryo.generation,
    seed: config.seed,
    nodeCount: embryo.nodeCount,
    edgeCount: embryo.edgeCount,
    archetypes: embryo.archetypes,
    schemaVersion: embryo.schemaVersion,
  };
}
