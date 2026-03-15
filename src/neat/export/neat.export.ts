import type { NeatLike } from '../shared/neat.shared.types';

/**
 * Persistence helpers for the NEAT controller's evolutionary state.
 *
 * This export chapter exists so snapshotting and rehydration stay discoverable
 * as their own boundary instead of living in the shared root controller folder.
 * The helpers here intentionally avoid importing the concrete `Neat` class
 * directly, which keeps the serialization flow reusable across the public
 * facade, tests, and static restore entrypoints.
 *
 * Typical usage follows two tracks:
 * - export or import just the population when you only need genome payloads;
 * - export or import the full state when you need innovation history,
 *   generation counters, and population data to resume a run faithfully.
 */

/**
 * JSON representation of an individual genome (network). The concrete shape is
 * produced by `Network#toJSON()` and re-hydrated via `Network.fromJSON()`. We use
 * an open record signature here because the network architecture may evolve with
 * plugins / future features (e.g. CPPNs, substrate metadata, ONNX export tags).
 */
export interface GenomeJSON {
  [key: string]: unknown;
}

/**
 * Connection innovation map entry.
 */
type InnovationMapEntry = [string, number];

/**
 * Serialized meta information describing a NEAT run, excluding the concrete
 * population genomes. This allows you to persist and resume experiment context
 * without committing to a particular population snapshot.
 */
export interface NeatMetaJSON {
  /** Number of input nodes expected by evolved networks. */
  input: number;
  /** Number of output nodes produced by evolved networks. */
  output: number;
  /** Current evolutionary generation index (0-based). */
  generation: number;
  /** Full options object (hyper-parameters) used to configure NEAT. */
  options: Record<string, unknown>;
  /** Innovation records for node split mutations: [compositeKey, innovationId]. */
  nodeSplitInnovations: InnovationMapEntry[];
  /** Innovation records for connection mutations: [compositeKey, innovationId]. */
  connInnovations: InnovationMapEntry[];
  /** Next global innovation number that will be assigned. */
  nextGlobalInnovation: number;
}

/**
 * Genome with toJSON serialization method.
 */
interface GenomeWithSerialization {
  toJSON: () => GenomeJSON;
}

/**
 * NEAT controller interface for export operations.
 */
interface NeatControllerForExport {
  input: number;
  output: number;
  generation: number;
  options: Record<string, unknown> & { popsize?: number };
  population: GenomeWithSerialization[];
  _nodeSplitInnovations: Map<string, number>;
  _connInnovations: Map<string, number>;
  _nextGlobalInnovation: number;
}

/**
 * Network class with static fromJSON method.
 */
interface NetworkClass {
  fromJSON: (json: GenomeJSON) => GenomeWithSerialization;
}

/**
 * NEAT class constructor interface.
 */
interface NeatConstructor {
  new (
    input: number,
    output: number,
    fitness: (network: GenomeWithSerialization) => number | Promise<number>,
    options?: Record<string, unknown>,
  ): NeatControllerForExport;
  fromJSON?: (
    meta: NeatMetaJSON,
    fitness: (network: GenomeWithSerialization) => number | Promise<number>,
  ) => NeatControllerForExport;
}

/**
 * Top-level bundle containing both NEAT meta information and the full array of
 * serialized genomes (population). This is what you get from `exportState()` and
 * feed into `importStateImpl()` to resume exactly where you left off.
 */
export interface NeatStateJSON {
  /** Serialized NEAT meta (innovation history, generation, options, etc.). */
  neat: NeatMetaJSON;
  /** Array of serialized genomes representing the current population. */
  population: GenomeJSON[];
}

/**
 * Export the current population (array of genomes) into plain JSON objects.
 * Each genome is converted via its `toJSON()` method. You can persist this
 * result (e.g. to disk, a database, or localStorage) and later rehydrate it
 * with {@link importPopulation}.
 *
 * Why export population only? Sometimes you want to snapshot just the set of
 * candidate solutions (e.g. for ensemble evaluation) without freezing the
 * innovation counters or hyper-parameters.
 *
 * Example:
 *
 * ```ts
 * // Assuming `neat` is an instance exposing this helper
 * const popSnapshot = neat.exportPopulation();
 * fs.writeFileSync('population.json', JSON.stringify(popSnapshot, null, 2));
 * ```
 *
 * @category Serialization
 * @returns Array of genome JSON objects.
 */
export function exportPopulation(this: NeatLike): GenomeJSON[] {
  const internal = this as unknown as NeatControllerForExport;
  return internal.population.map((genome) => genome.toJSON());
}

/**
 * Import (replace) the current population from an array of serialized genomes.
 * This does not touch NEAT meta state (generation, innovations, etc.) - only the
 * population array and implied `popsize` are updated.
 *
 * Example:
 *
 * ```ts
 * const populationData: GenomeJSON[] = JSON.parse(fs.readFileSync('population.json', 'utf8'));
 * neat.importPopulation(populationData); // population replaced
 * neat.evolve(); // continue evolving with new starting genomes
 * ```
 *
 * Edge cases handled:
 * - Empty array => becomes an empty population (popsize=0).
 * - Malformed entries will throw if `Network.fromJSON` rejects them.
 *
 * @param populationJSON Array of serialized genome objects.
 */
export async function importPopulation(
  this: NeatLike,
  populationJSON: GenomeJSON[],
): Promise<void> {
  const { default: Network } = await import('../../architecture/network');
  const internal = this as unknown as NeatControllerForExport;
  internal.population = populationJSON.map((serializedGenome) =>
    (Network as unknown as NetworkClass).fromJSON(serializedGenome),
  );
  internal.options.popsize = internal.population.length;
}

/**
 * Convenience helper that returns a full evolutionary snapshot: both NEAT meta
 * information and the serialized population array. Use this when you want a
 * truly pause-and-resume capability including innovation bookkeeping.
 *
 * Example:
 *
 * ```ts
 * const state = neat.exportState();
 * fs.writeFileSync('state.json', JSON.stringify(state));
 * // ...later / elsewhere...
 * const raw = JSON.parse(fs.readFileSync('state.json', 'utf8')) as NeatStateJSON;
 * const neat2 = Neat.importState(raw, fitnessFn); // identical evolutionary context
 * ```
 *
 * @returns A {@link NeatStateJSON} bundle containing meta + population.
 */
export function exportState(this: NeatLike): NeatStateJSON {
  return {
    neat: toJSONImpl.call(this),
    population: exportPopulation.call(this),
  };
}

/**
 * Static-style helper that rehydrates a full evolutionary state previously
 * produced by {@link exportState}. Invoke this with the NEAT class (not an
 * instance) bound as `this`, e.g. `Neat.importStateImpl(bundle, fitnessFn)`.
 * It constructs a new NEAT instance using the meta data, then imports the
 * population (if present).
 *
 * Safety and validation:
 * - Throws if the bundle is not an object.
 * - Silently skips population import if `population` is missing or not an array.
 *
 * Example:
 *
 * ```ts
 * const bundle: NeatStateJSON = JSON.parse(fs.readFileSync('state.json', 'utf8'));
 * const neat = Neat.importStateImpl(bundle, fitnessFn);
 * neat.evolve();
 * ```
 *
 * @param stateBundle Full state bundle from {@link exportState}.
 * @param fitnessFunction Fitness evaluation callback used for new instance.
 * @returns Rehydrated NEAT instance ready to continue evolving.
 */
export async function importStateImpl(
  this: NeatConstructor,
  stateBundle: NeatStateJSON,
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
): Promise<NeatControllerForExport> {
  if (!stateBundle || typeof stateBundle !== 'object')
    throw new Error('Invalid state bundle');

  const neatInstance = (
    this as NeatConstructor & {
      fromJSON?: typeof fromJSONImpl;
    }
  ).fromJSON?.(stateBundle.neat, fitnessFunction);

  if (!neatInstance)
    throw new Error('Failed to create NEAT instance from JSON');

  if (Array.isArray(stateBundle.population))
    await importPopulation.call(
      neatInstance as unknown as NeatLike,
      stateBundle.population,
    );

  return neatInstance;
}

/**
 * Serialize NEAT meta (excluding the mutable population) for persistence of
 * innovation history and experiment configuration. This is sufficient to
 * recreate a blank NEAT run at the same evolutionary generation with the same
 * innovation counters, enabling deterministic continuation when combined later
 * with a saved population.
 *
 * Example:
 *
 * ```ts
 * const meta = neat.toJSONImpl();
 * fs.writeFileSync('neat-meta.json', JSON.stringify(meta));
 * // ... later ...
 * const metaLoaded = JSON.parse(fs.readFileSync('neat-meta.json', 'utf8')) as NeatMetaJSON;
 * const neat2 = Neat.fromJSONImpl(metaLoaded, fitnessFn); // empty population
 * ```
 *
 * @returns {@link NeatMetaJSON} object describing current NEAT meta state.
 */
export function toJSONImpl(this: NeatLike): NeatMetaJSON {
  const internal = this as unknown as NeatControllerForExport;
  return {
    input: internal.input,
    output: internal.output,
    generation: internal.generation,
    options: internal.options,
    nodeSplitInnovations: Array.from(internal._nodeSplitInnovations.entries()),
    connInnovations: Array.from(internal._connInnovations.entries()),
    nextGlobalInnovation: internal._nextGlobalInnovation,
  };
}

/**
 * Static-style implementation that rehydrates a NEAT instance from previously
 * exported meta JSON produced by {@link toJSONImpl}. This does not restore a
 * population; callers typically follow up with `importPopulation` or use
 * {@link importStateImpl} for a complete restore.
 *
 * Example:
 *
 * ```ts
 * const meta: NeatMetaJSON = JSON.parse(fs.readFileSync('neat-meta.json', 'utf8'));
 * const neat = Neat.fromJSONImpl(meta, fitnessFn); // empty population, same innovations
 * neat.importPopulation(popSnapshot); // optional
 * ```
 *
 * @param neatJSON Serialized meta (no population).
 * @param fitnessFunction Fitness callback used to construct the new instance.
 * @returns Fresh NEAT instance with restored innovation history.
 */
export function fromJSONImpl(
  this: NeatConstructor,
  neatJSON: NeatMetaJSON,
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
): NeatControllerForExport {
  const neatInstance = new this(
    neatJSON.input,
    neatJSON.output,
    fitnessFunction,
    neatJSON.options || {},
  );

  neatInstance.generation = neatJSON.generation || 0;

  if (Array.isArray(neatJSON.nodeSplitInnovations))
    neatInstance._nodeSplitInnovations = new Map(neatJSON.nodeSplitInnovations);

  if (Array.isArray(neatJSON.connInnovations))
    neatInstance._connInnovations = new Map(neatJSON.connInnovations);

  if (typeof neatJSON.nextGlobalInnovation === 'number')
    neatInstance._nextGlobalInnovation = neatJSON.nextGlobalInnovation;

  return neatInstance;
}
