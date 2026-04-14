/**
 * NEAT Configuration Module
 *
 * Purpose:
 * -------
 * Provides utilities for instantiating and seeding NEAT (NeuroEvolution of Augmenting Topologies)
 * instances with standardized configuration and optional initial populations.
 *
 * This module encapsulates:
 *  - NEAT driver creation with opinionated defaults (elitism, provenance, mutation operators)
 *  - Population seeding with defensive cloning and best-effort error handling
 *  - Configuration normalization and validation
 *
 * ES2023 Policy:
 * -------------
 * - Uses nullish coalescing `??` for default values (never `||`)
 * - Descriptive variable names (no short identifiers like `i`, `c`, `p`)
 * - Optional chaining `?.` for safe property access
 * - Spread operator for array/object operations
 *
 * @module evolutionEngine/neatConfiguration
 */

import { Neat, Network, methods } from '../../../src/neataptic';

/**
 * NEAT configuration object shape for type safety.
 */
export interface NeatConfig {
  popSize?: number;
  mutation?: unknown[];
  allowRecurrent?: boolean;
  elitism?: number;
  provenance?: number;
  mutationRate?: number;
  mutationAmount?: number;
  selection?: unknown;
  equal?: boolean;
  clear?: boolean;
  popsize?: number;
  network?: Network;
  fitnessFunction?: (network: Network) => number;
  [key: string]: unknown;
}

/**
 * Create a NEAT instance with normalized configuration and opinionated defaults.
 *
 * This function encapsulates the NEAT driver instantiation logic with defensive
 * configuration normalization. It applies sensible defaults for population size,
 * mutation operators, elitism, provenance, and various evolutionary strategies.
 *
 * Configuration Philosophy:
 *  - Prefer explicit defaults over implicit framework defaults
 *  - Use nullish coalescing `??` for clarity (avoid falsy semantics)
 *  - Compute derived settings (elitism, provenance) from population size
 *  - Enable modern features by default (adaptive mutation, multi-objective, novelty)
 *
 * Default Constants (from EvolutionEngine):
 *  - Population size: 150
 *  - Elitism fraction: 0.05 (top 5% preserved)
 *  - Provenance fraction: 0.1 (top 10% used for breeding)
 *  - Mutation rate: 0.3
 *  - Mutation amount: 0.05
 *  - Min hidden nodes: 0
 *  - Target species: 8
 *  - Entropy range: [0.40, 0.60]
 *  - Adaptive smooth factor: 0.90
 *
 * Parameters:
 * @param inputCount - Number of input nodes for the network topology
 * @param outputCount - Number of output nodes for the network topology
 * @param fitnessCallback - Function to evaluate network fitness (net: Network) => number
 * @param cfg - Optional configuration bag with the following supported properties:
 *  - popSize: Population size (default: 150)
 *  - mutation: Array of mutation operators (default: comprehensive set including LSTM)
 *  - allowRecurrent: Enable recurrent connections (default: true)
 *  - network: Optional builder-backed seed network used as the NEAT base graph
 *  - adaptiveMutation: Adaptive mutation config (default: enabled with 'twoTier' strategy)
 *  - multiObjective: Multi-objective config (default: enabled with 'nodes' metric)
 *  - telemetry: Telemetry config (default: enabled with all metrics)
 *  - lineageTracking: Enable lineage tracking (default: false)
 *  - novelty: Novelty search config (default: enabled with 0.15 blend factor)
 *  - targetSpecies: Target number of species (default: 8)
 *  - adaptiveTargetSpecies: Adaptive species targeting config (default: enabled)
 *
 * @returns Configured Neat instance ready for evolution
 *
 * @example
 * // Create a NEAT instance with custom population size and disabled lineage tracking
 * const neat = createNeat(10, 4, fitnessFn, { popSize: 200, lineageTracking: false });
 *
 * @example
 * // Create with default configuration
 * const neat = createNeat(10, 4, fitnessFn);
 */
export const createNeat = (
  inputCount: number,
  outputCount: number,
  fitnessCallback: (net: Network) => number,
  cfg?: NeatConfig,
): Neat => {
  const NeatNetworkConstructor = Neat as unknown as new (
    input: number,
    output: number,
    fitness: (network: Network) => unknown,
    options: NeatConfig,
  ) => Neat;

  // Default constants (extracted from EvolutionEngine static fields)
  const DEFAULT_POPSIZE = 150;
  const DEFAULT_ELITISM_FRACTION = 0.05;
  const DEFAULT_PROVENANCE_FRACTION = 0.1;
  const DEFAULT_MUTATION_RATE = 0.3;
  const DEFAULT_MUTATION_AMOUNT = 0.05;
  const DEFAULT_MIN_HIDDEN = 0;
  const DEFAULT_TARGET_SPECIES = 8;
  const DEFAULT_ENTROPY_RANGE: [number, number] = [0.4, 0.6];
  const DEFAULT_ADAPTIVE_SMOOTH = 0.9;

  // Step 1: Normalize configuration bag and derive primary numeric settings.
  const conf = cfg ?? {};
  const allowRecurrent = conf.allowRecurrent !== false;
  const popSize: number = Number.isFinite(conf.popSize)
    ? (conf.popSize as number)
    : DEFAULT_POPSIZE;
  const mutationOps = Array.isArray(conf.mutation)
    ? conf.mutation
    : resolveDefaultMutationShelf(allowRecurrent);

  // Step 2: Compute derived integer settings with descriptive names.
  const elitism = Math.max(1, Math.floor(popSize * DEFAULT_ELITISM_FRACTION));
  const provenance = Math.max(
    1,
    Math.floor(popSize * DEFAULT_PROVENANCE_FRACTION),
  );

  // Step 3: Compose other option objects using nullish coalescing for defaults.
  const adaptiveMutation = conf.adaptiveMutation ?? {
    enabled: true,
    strategy: 'twoTier',
  };
  const multiObjective = conf.multiObjective ?? {
    enabled: true,
    complexityMetric: 'nodes',
    autoEntropy: true,
  };
  const telemetry = conf.telemetry ?? {
    enabled: true,
    performance: true,
    complexity: true,
    hypervolume: true,
  };
  const lineageTracking = conf.lineageTracking === true;
  const novelty = conf.novelty ?? { enabled: true, blendFactor: 0.15 };
  const targetSpecies = conf.targetSpecies ?? DEFAULT_TARGET_SPECIES;
  const adaptiveTargetSpecies = conf.adaptiveTargetSpecies ?? {
    enabled: true,
    entropyRange: DEFAULT_ENTROPY_RANGE,
    speciesRange: [6, 14],
    smooth: DEFAULT_ADAPTIVE_SMOOTH,
  };

  // Step 4: Instantiate the Neat driver with the assembled options.
  const neatInstance = new NeatNetworkConstructor(
    inputCount,
    outputCount,
    fitnessCallback,
    {
      popsize: popSize,
      mutation: mutationOps,
      mutationRate: DEFAULT_MUTATION_RATE,
      mutationAmount: DEFAULT_MUTATION_AMOUNT,
      elitism,
      provenance,
      allowRecurrent,
      minHidden: DEFAULT_MIN_HIDDEN,
      adaptiveMutation,
      multiObjective,
      telemetry,
      lineageTracking,
      novelty,
      network: conf.network,
      targetSpecies,
      adaptiveTargetSpecies,
    },
  );

  return neatInstance;
};

/**
 * Resolves the default mutation shelf for the ASCII Maze demo.
 *
 * The maze example already opts into recurrent-capable search at the
 * controller level, so its default mutation shelf should expose the newer
 * temporal and gated operators that make that policy meaningful. When callers
 * explicitly disable recurrent growth, the shelf falls back to the compact
 * feed-forward subset so the controller contract stays aligned with the shelf.
 *
 * @param allowRecurrent - Whether recurrent and gated growth is allowed.
 * @returns Demo-aligned default mutation shelf.
 */
function resolveDefaultMutationShelf(allowRecurrent: boolean): unknown[] {
  const baseMutationShelf = [
    methods.mutation.ADD_NODE,
    methods.mutation.SUB_NODE,
    methods.mutation.ADD_CONN,
    methods.mutation.SUB_CONN,
    methods.mutation.MOD_BIAS,
    methods.mutation.MOD_ACTIVATION,
    methods.mutation.MOD_WEIGHT,
  ];

  if (!allowRecurrent) {
    return baseMutationShelf;
  }

  return [
    ...baseMutationShelf,
    methods.mutation.ADD_GATE,
    methods.mutation.ADD_SELF_CONN,
    methods.mutation.ADD_BACK_CONN,
    methods.mutation.ADD_LSTM_NODE,
    methods.mutation.ADD_GRU_NODE,
  ];
}

/**
 * Seed the NEAT population from an optional initial population and/or an optional
 * initial best network.
 *
 * This function is intentionally best-effort and non-throwing: any cloning or
 * driver mutating errors are swallowed so the evolution loop can continue.
 *
 * Design Rationale:
 *  - Uses pooled buffer to avoid per-call allocations (amortized O(1) cloning)
 *  - Defensive cloning: fallback to original reference on clone failure
 *  - Keeps `neat.options.popsize` synchronized with actual population length
 *  - All errors swallowed to maintain resilient evolution loop
 *
 * Population Seeding Strategy:
 *  1) When `initialPopulation` is provided, clone all networks into pooled buffer
 *  2) Grow pooled buffer only when necessary (reuse across calls)
 *  3) When `initialBestNetwork` is provided, place clone at index 0
 *  4) Synchronize `neat.options.popsize` with actual population length
 *
 * Pooling Pattern:
 *  - The pooled buffer `#SCRATCH_POP_CLONE` is shared across all seeding operations
 *  - Buffer grows monotonically (never shrinks) to amortize allocation costs
 *  - Logical length is set via `.length` property for correct slice semantics
 *
 * Parameters:
 * @param neat - NEAT driver/manager object which may receive the initial population
 * @param initialPopulation - Optional array of networks to use as the starting population
 * @param initialBestNetwork - Optional single network to place at index 0 (best seed)
 * @param targetPopSize - Fallback population size used when `neat.population` is missing
 * @param scratchPopClone - Pooled clone buffer to reuse across calls (will be grown if needed)
 *
 * @returns The scratchPopClone buffer (may be a new array if grown)
 *
 * @example
 * // Seed with a provided population and ensure `neat.options.popsize` is kept in sync
 * const pool = seedInitialPopulation(neat, providedPopulation, providedBest, 150, pool);
 *
 * @example
 * // Seed with only a best network (population will be empty except index 0)
 * const pool = seedInitialPopulation(neat, undefined, bestNetwork, 150, pool);
 */
export const seedInitialPopulation = (
  neat: Neat,
  initialPopulation: Network[] | undefined,
  initialBestNetwork: Network | undefined,
  targetPopSize: number,
  scratchPopClone: Network[],
): Network[] => {
  // Step 1: Defensive guard - nothing to do without a neat manager
  if (!neat) return scratchPopClone;

  try {
    // Step 2: If an explicit initial population was provided, clone into pooled buffer.
    if (Array.isArray(initialPopulation) && initialPopulation.length > 0) {
      const sourceLength = initialPopulation.length;

      // Reuse the pooled clone array. Grow only when necessary to
      // avoid repeated allocations across runs.
      let pooledCloneBuffer = scratchPopClone;
      if (
        !Array.isArray(pooledCloneBuffer) ||
        pooledCloneBuffer.length < sourceLength
      ) {
        // Allocate a new array of the required capacity and replace the pooled reference.
        pooledCloneBuffer = new Array(sourceLength);
      }

      // Fill the pooled buffer with cloned networks (when `.clone()` exists).
      for (let sourceIndex = 0; sourceIndex < sourceLength; sourceIndex++) {
        const candidateNetwork = initialPopulation[sourceIndex];
        try {
          pooledCloneBuffer[sourceIndex] =
            candidateNetwork && typeof candidateNetwork.clone === 'function'
              ? candidateNetwork.clone()
              : candidateNetwork;
        } catch {
          // Best-effort: if cloning fails, fall back to the original reference.
          pooledCloneBuffer[sourceIndex] = candidateNetwork;
        }
      }

      // Mark the logical length of the pooled buffer and adopt it as the starting population.
      pooledCloneBuffer.length = sourceLength;
      neat.population = pooledCloneBuffer;
      scratchPopClone = pooledCloneBuffer;
    }

    // Step 3: If a single best network was provided, ensure it's placed at index 0.
    if (initialBestNetwork) {
      // Ensure we have an actual population array to write into.
      if (!Array.isArray(neat.population)) neat.population = [];
      try {
        neat.population[0] =
          typeof initialBestNetwork.clone === 'function'
            ? initialBestNetwork.clone()
            : initialBestNetwork;
      } catch {
        // Swallow per-network clone errors to preserve best-effort semantics.
      }
    }

    // Step 4: Keep the driver's configured popsize in sync with the actual population.
    try {
      neat.options = neat.options || {};
      neat.options.popsize = Array.isArray(neat.population)
        ? neat.population.length
        : targetPopSize;
    } catch {
      /* best-effort; swallow */
    }
  } catch {
    // Top-level safety net: swallow all errors to avoid breaking the evolution loop.
    try {
      neat.options = neat.options || {};
      neat.options.popsize = Array.isArray(neat.population)
        ? neat.population.length
        : targetPopSize;
    } catch {
      /* ignore */
    }
  }

  return scratchPopClone;
};
