import { NeatLike } from './neat.types';

// ----------------------------------------------------------------------------------
// Export / Import helpers for NEAT evolutionary state.
// These utilities deliberately avoid importing the concrete Neat class directly so
// they can be mixed into lighter-weight facades or used in static contexts.
// ----------------------------------------------------------------------------------

/**
 * JSON representation of an individual genome (network). The concrete shape is
 * produced by `Network#toJSON()` and re‑hydrated via `Network.fromJSON()`. We use
 * an open record signature here because the network architecture may evolve with
 * plugins / future features (e.g. CPPNs, substrate metadata, ONNX export tags).
 */
export interface GenomeJSON {
  [key: string]: any; // eslint-disable-line @typescript-eslint/no-explicit-any
}

/**
 * Serialized meta information describing a NEAT run, excluding the concrete
 * population genomes. This allows you to persist & resume experiment context
 * (innovation history, current generation, IO sizes, hyper‑parameters) without
 * committing to a particular population snapshot.
 */
export interface NeatMetaJSON {
  /** Number of input nodes expected by evolved networks. */
  input: number;
  /** Number of output nodes produced by evolved networks. */
  output: number;
  /** Current evolutionary generation index (0-based). */
  generation: number;
  /** Full options object (hyper‑parameters) used to configure NEAT. */
  options: any; // retained as any until options interface is extracted
  /** Innovation records for node split mutations: [compositeKey, innovationId]. */
  nodeSplitInnovations: [any, any][]; // key/value pairs serialised from Map
  /** Innovation records for connection mutations: [compositeKey, innovationId]. */
  connInnovations: [any, any][];
  /** Next global innovation number that will be assigned. */
  nextGlobalInnovation: number;
}

/**
 * Top‑level bundle containing both NEAT meta information and the full array of
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
 * Why export population only? Sometimes you want to snapshot *just* the set of
 * candidate solutions (e.g. for ensemble evaluation) without freezing the
 * innovation counters or hyper‑parameters.
 *
 * Example:
 * ```ts
 * // Assuming `neat` is an instance exposing this helper
 * const popSnapshot = neat.exportPopulation();
 * fs.writeFileSync('population.json', JSON.stringify(popSnapshot, null, 2));
 * ```
 * @category Serialization
 * @returns Array of genome JSON objects.
 */
export function exportPopulation(this: NeatLike): GenomeJSON[] {
  // 1. Map each genome in the current population to its serializable form
  return (this as any).population.map((genome: any) => genome.toJSON());
}

/**
 * Import (replace) the current population from an array of serialized genomes.
 * This does not touch NEAT meta state (generation, innovations, etc.)—only the
 * population array and implied `popsize` are updated.
 *
 * Example:
 * ```ts
 * const populationData: GenomeJSON[] = JSON.parse(fs.readFileSync('population.json','utf8'));
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
export function importPopulation(
  this: NeatLike,
  populationJSON: GenomeJSON[]
): void {
  /** const Network class used for genome (network) rehydration */
  const Network = require('../architecture/network').default;
  // 1. Recreate each genome via Network.fromJSON
  (this as any).population = populationJSON.map((serializedGenome: any) =>
    Network.fromJSON(serializedGenome)
  );
  // 2. Keep popsize option in sync with actual population length
  (this as any).options.popsize = (this as any).population.length;
}

/**
 * Convenience helper that returns a full evolutionary snapshot: both NEAT meta
 * information and the serialized population array. Use this when you want a
 * truly *pause‑and‑resume* capability including innovation bookkeeping.
 *
 * Example:
 * ```ts
 * const state = neat.exportState();
 * fs.writeFileSync('state.json', JSON.stringify(state));
 * // ...later / elsewhere...
 * const raw = JSON.parse(fs.readFileSync('state.json','utf8')) as NeatStateJSON;
 * const neat2 = Neat.importState(raw, fitnessFn); // identical evolutionary context
 * ```
 * @returns A {@link NeatStateJSON} bundle containing meta + population.
 */
export function exportState(this: NeatLike): NeatStateJSON {
  /** const lazily loaded export helpers (avoids circular deps) */
  const { toJSONImpl, exportPopulation } = require('./neat.export');
  // 1. Serialize meta
  // 2. Serialize population
  // 3. Package into a bundle for persistence
  return {
    neat: toJSONImpl.call(this as any),
    population: exportPopulation.call(this as any),
  };
}

/**
 * Static-style helper that rehydrates a full evolutionary state previously
 * produced by {@link exportState}. Invoke this with the NEAT *class* (not an
 * instance) bound as `this`, e.g. `Neat.importStateImpl(bundle, fitnessFn)`.
 * It constructs a new NEAT instance using the meta data, then imports the
 * population (if present).
 *
 * Safety & validation:
 * - Throws if the bundle is not an object.
 * - Silently skips population import if `population` is missing or not an array.
 *
 * Example:
 * ```ts
 * const bundle: NeatStateJSON = JSON.parse(fs.readFileSync('state.json','utf8'));
 * const neat = Neat.importStateImpl(bundle, fitnessFn);
 * neat.evolve();
 * ```
 * @param stateBundle Full state bundle from {@link exportState}.
 * @param fitnessFunction Fitness evaluation callback used for new instance.
 * @returns Rehydrated NEAT instance ready to continue evolving.
 */
export function importStateImpl(
  this: any,
  stateBundle: NeatStateJSON,
  fitnessFunction: (network: any) => number
): any {
  // 1. Basic validation of bundle shape
  if (!stateBundle || typeof stateBundle !== 'object')
    throw new Error('Invalid state bundle');
  // 2. Reconstruct Neat meta & instance
  const neatInstance = this.fromJSON(stateBundle.neat, fitnessFunction);
  // 3. Import population if provided
  if (Array.isArray(stateBundle.population))
    neatInstance.import(stateBundle.population);
  // 4. Return fully restored instance
  return neatInstance;
}

/**
 * Serialize NEAT meta (excluding the mutable population) for persistence of
 * innovation history and experiment configuration. This is sufficient to
 * recreate a *blank* NEAT run at the same evolutionary generation with the
 * same innovation counters, enabling deterministic continuation when combined
 * later with a saved population.
 *
 * Example:
 * ```ts
 * const meta = neat.toJSONImpl();
 * fs.writeFileSync('neat-meta.json', JSON.stringify(meta));
 * // ... later ...
 * const metaLoaded = JSON.parse(fs.readFileSync('neat-meta.json','utf8')) as NeatMetaJSON;
 * const neat2 = Neat.fromJSONImpl(metaLoaded, fitnessFn); // empty population
 * ```
 * @returns {@link NeatMetaJSON} object describing current NEAT meta state.
 */
export function toJSONImpl(this: NeatLike): NeatMetaJSON {
  // 1. Return a plain object with primitive / array serializable fields only
  return {
    input: (this as any).input,
    output: (this as any).output,
    generation: (this as any).generation,
    options: (this as any).options,
    nodeSplitInnovations: Array.from(
      (this as any)._nodeSplitInnovations.entries()
    ),
    connInnovations: Array.from((this as any)._connInnovations.entries()),
    nextGlobalInnovation: (this as any)._nextGlobalInnovation,
  };
}

/**
 * Static-style implementation that rehydrates a NEAT instance from previously
 * exported meta JSON produced by {@link toJSONImpl}. This does *not* restore a
 * population; callers typically follow up with `importPopulation` or use
 * {@link importStateImpl} for a complete restore.
 *
 * Example:
 * ```ts
 * const meta: NeatMetaJSON = JSON.parse(fs.readFileSync('neat-meta.json','utf8'));
 * const neat = Neat.fromJSONImpl(meta, fitnessFn); // empty population, same innovations
 * neat.importPopulation(popSnapshot); // optional
 * ```
 * @param neatJSON Serialized meta (no population).
 * @param fitnessFunction Fitness callback used to construct the new instance.
 * @returns Fresh NEAT instance with restored innovation history.
 */
export function fromJSONImpl(
  this: any,
  neatJSON: NeatMetaJSON,
  fitnessFunction: (network: any) => number
): any {
  /** const alias for the constructor (class) this function is bound to */
  const NeatClass = this as any;
  // 1. Instantiate with stored IO sizes & options
  const neatInstance = new NeatClass(
    neatJSON.input,
    neatJSON.output,
    fitnessFunction,
    neatJSON.options || {}
  );
  // 2. Restore generation index
  neatInstance.generation = neatJSON.generation || 0;
  // 3. Restore innovation maps when present
  if (Array.isArray(neatJSON.nodeSplitInnovations))
    neatInstance._nodeSplitInnovations = new Map(neatJSON.nodeSplitInnovations);
  if (Array.isArray(neatJSON.connInnovations))
    neatInstance._connInnovations = new Map(neatJSON.connInnovations);
  // 4. Restore next innovation counter
  if (typeof neatJSON.nextGlobalInnovation === 'number')
    neatInstance._nextGlobalInnovation = neatJSON.nextGlobalInnovation;
  // 5. Return reconstructed instance (empty population)
  return neatInstance;
}
