import type Network from '../../architecture/network/network';
import type { NetworkJSON } from '../../architecture/network/network.types';
import { toJSONImpl as serializeNetworkToJson } from '../../architecture/network/serialize/network.serialize.utils';
import type { NeatLike } from '../shared/neat.shared.types';
import {
  createGenomeFromNetwork,
  createGenomeFromNetworkJson,
  createNetworkFromGenome,
  createNetworkJsonFromGenome,
} from '../genome/genome';
import type { NeatGenomeCaptureOptions } from '../genome/genome';
import {
  restoreInnovationTracker,
  serializeInnovationTracker,
} from '../innovation-tracker/innovation-tracker';
import {
  CURRENT_META_FORMAT_VERSION,
  CURRENT_STATE_FORMAT_VERSION,
  FULL_CHECKPOINT_MODE,
} from './neat.export.types';
import type {
  GenomeJSON,
  GenomeWithSerialization,
  NeatControllerForExport,
  NeatMetaJSON,
  NeatStateJSON,
  NeatConstructor,
  NetworkClass,
} from './neat.export.types';
import {
  assertCheckpointGenomeIsNative,
  assertSerializedGenomeCarriesCheckpointIdentity,
  findNextGenomeIdFloor,
  resolveMetaFormatVersion,
  resolveStateFormatVersion,
} from './neat.export.utils';
import {
  hydrateGenomeControllerMeta,
  serializeGenomeCheckpoint,
  splitSerializedGenomeCheckpoint,
} from './neat.export.population.utils';
import {
  restoreRuntimeMeta,
  serializeRuntimeMeta,
} from './neat.export.runtime.utils';
import {
  restoreSpeciationCheckpoint,
  serializeSpeciationCheckpoint,
} from './neat.export.speciation.utils';
import {
  NeatExportPopulationValidationError,
  NeatExportStateBundleValidationError,
  NeatExportStateControllerRestoreError,
} from './neat.export.errors';

export type {
  GenomeControllerMetaJSON,
  GenomeJSON,
  NeatMetaJSON,
  NeatRuntimeMetaJSON,
  NeatStateJSON,
  SpeciationCheckpointJSON,
  SpeciesCheckpointJSON,
} from './neat.export.types';

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
 *
 * A useful way to read this chapter is as a pause-and-resume ladder:
 * - `exportPopulation()` and `importPopulation()` move only candidate genomes
 * - `toJSONImpl()` and `fromJSONImpl()` move only controller meta state
 * - `exportState()` and `importStateImpl()` combine both layers into one full resume bundle
 *
 * Deterministic replay contract (why this boundary exists):
 *
 * - **Population-only snapshots** are intentionally *not* a full replay.
 *   They preserve the candidate networks plus controller-owned per-genome
 *   metadata (stable genome ids, lineage hints, optional per-genome RNG state),
 *   but they do not promise that a resumed run will make the same future
 *   structural innovation assignments.
 * - **Meta-only checkpoints** preserve controller bookkeeping without forcing a
 *   particular population to travel with it. This is useful for carrying
 *   options, generation counters, and innovation tracking across environments.
 * - **Full checkpoints** are the pause-and-resume surface. When you restore a
 *   full checkpoint into the same codebase, the controller is expected to
 *   continue evolving as if it had never stopped.
 *
 * “Same seed + same checkpoint + same code” is the target replay promise.
 * That promise only holds when controller-owned randomness and architecture
 * counters are treated as explicit state (see `neat.export.runtime.utils.ts`),
 * and when structural identity is treated as explicit history (node gene ids
 * plus connection innovation numbers).
 *
 * Legacy/import bridge: `Network.fromJSON()` supports permissive restore flows
 * for older payloads, but the strict checkpoint path in this chapter validates
 * identity fields before allowing a resume. Treat fallback compatibility as a
 * deliberate opt-in bridge, not as native proper-NEAT semantics.
 *
 * That split matters because not every persistence use case is a full replay.
 * Sometimes you want to archive candidate solutions for later inspection,
 * benchmark the same population under a new fitness function, or ship genomes
 * between environments without also freezing the controller's innovation
 * history. Other times you need a true checkpoint that can continue evolving as
 * if the process had never stopped.
 *
 * Read the chapter in this order:
 * - start with `exportPopulation()` and `importPopulation()` when you only need
 *   candidate genomes,
 * - continue to `toJSONImpl()` and `fromJSONImpl()` when you need controller
 *   metadata without the live population,
 * - finish with `exportState()` and `importStateImpl()` when you need a full
 *   pause-and-resume checkpoint.
 *
 * ```mermaid
 * flowchart TD
 *   Runtime[Live NEAT controller] --> PopOnly[Population-only snapshot]
 *   Runtime --> MetaOnly[Meta-only checkpoint]
 *   Runtime --> FullState[Full state bundle]
 *   PopOnly --> ImportPop[Replace population in an existing controller]
 *   MetaOnly --> ImportMeta[Rebuild controller bookkeeping]
 *   FullState --> Resume[Restore bookkeeping and population together]
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   Snapshot[Saved checkpoint] --> Restore[importState]
 *   Restore --> Determinism{Has runtime meta\n+ explicit identity?}
 *   Determinism -->|yes| Replay[Controller-owned replay\nfuture innovations match]
 *   Determinism -->|no| Bridge[Legacy or import bridge\nno replay guarantee]
 * ```
 *
 * Background reading: Wikipedia contributors,
 * [Serialization](https://en.wikipedia.org/wiki/Serialization).
 */

/**
 * Export the current population (array of genomes) into plain JSON objects.
 * Each genome is converted via its `toJSON()` method. You can persist this
 * result (e.g. to disk, a database, or localStorage) and later rehydrate it
 * with {@link importPopulation}.
 *
 * Why export population only? Sometimes you want to snapshot just the set of
 * candidate solutions (e.g. for ensemble evaluation) without freezing the
 * innovation counters or hyper-parameters. Even in that lighter mode, the
 * export keeps controller-owned genome metadata next to each network payload so
 * imported populations do not silently lose genome ids or lineage evidence.
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
  const genomeCaptureOptions = resolveGenomeCaptureOptions(internal);

  return internal.population.map((genome, genomeIndex) => {
    assertCheckpointGenomeIsNative(genome, genomeIndex, 'export');

    const runtimeGenome = genome as unknown as Network;
    const runtimePayload = serializeNetworkToJson.call(runtimeGenome);
    const strictGenome = createGenomeFromNetwork(
      runtimeGenome,
      genomeCaptureOptions,
    );
    const strictNetworkPayload = createNetworkJsonFromGenome(strictGenome, {
      dropout: runtimePayload.dropout,
      architecture: runtimePayload.architecture,
    });

    return serializeGenomeCheckpoint(
      genome,
      strictNetworkPayload as unknown as GenomeJSON,
    );
  });
}

/**
 * Import (replace) the current population from an array of serialized genomes.
 * This does not touch NEAT meta state (generation, innovations, etc.) - only the
 * population array and implied `popsize` are updated.
 * That makes it the right tool when you want to swap candidate solutions into an
 * existing controller context instead of restoring a full historical checkpoint.
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
 * - Legacy snapshots without controller genome ids are upgraded by assigning
 *   fresh ids inside the destination controller.
 * - Malformed entries or native genomes that fail proper-NEAT validation throw
 *   explicit population-validation errors.
 *
 * @param populationJSON Array of serialized genome objects.
 * @returns Promise that resolves once all genomes have been rehydrated and the
 * controller population has been replaced.
 */
export async function importPopulation(
  this: NeatLike,
  populationJSON: GenomeJSON[],
): Promise<void> {
  if (!Array.isArray(populationJSON)) {
    throw new NeatExportPopulationValidationError(
      'Population snapshots must be arrays of serialized genomes.',
    );
  }

  const internal = this as unknown as NeatControllerForExport;
  const genomeCaptureOptions = resolveGenomeCaptureOptions(internal);
  const seenGenomeIds = new Set<number>();
  let nextAssignedGenomeId = internal._nextGenomeId ?? 1;

  internal.population = populationJSON.map((serializedGenome, genomeIndex) => {
    if (
      !serializedGenome ||
      typeof serializedGenome !== 'object' ||
      Array.isArray(serializedGenome)
    ) {
      throw new NeatExportPopulationValidationError(
        `Population snapshot entry ${genomeIndex} must be a serialized genome object.`,
      );
    }

    const { controllerMeta, networkPayload } =
      splitSerializedGenomeCheckpoint(serializedGenome);
    assertSerializedGenomeCarriesCheckpointIdentity(
      networkPayload,
      genomeIndex,
    );
    const serializedNetworkPayload = networkPayload as unknown as NetworkJSON;
    const strictGenome = createGenomeFromNetworkJson(
      serializedNetworkPayload,
      genomeCaptureOptions,
    );
    const genome = createNetworkFromGenome(strictGenome, {
      dropout:
        typeof serializedNetworkPayload.dropout === 'number'
          ? serializedNetworkPayload.dropout
          : undefined,
      architecture: serializedNetworkPayload.architecture,
    }) as unknown as ReturnType<NetworkClass['fromJSON']>;

    nextAssignedGenomeId = hydrateGenomeControllerMeta(
      genome,
      controllerMeta,
      seenGenomeIds,
      nextAssignedGenomeId,
    );
    assertCheckpointGenomeIsNative(genome, genomeIndex, 'import');
    return genome;
  });

  internal.options.popsize = internal.population.length;
  internal._nextGenomeId = Math.max(
    nextAssignedGenomeId,
    findNextGenomeIdFloor(internal.population),
  );
}

function resolveGenomeCaptureOptions(
  controller: NeatControllerForExport,
): NeatGenomeCaptureOptions {
  const genomeExtensions = controller.options.genomeExtensions;
  if (
    !genomeExtensions ||
    typeof genomeExtensions !== 'object' ||
    Array.isArray(genomeExtensions)
  ) {
    return {};
  }

  return {
    connectionGain: genomeExtensions.connectionGain === true,
    nodeResponse: genomeExtensions.nodeResponse === true,
    disabledConnectionReenableProbability:
      genomeExtensions.disabledConnectionReenableProbability === true,
  };
}

/**
 * Convenience helper that returns a full evolutionary snapshot: both NEAT meta
 * information and the serialized population array. Use this when you want a
 * truly pause-and-resume capability including innovation bookkeeping, stable
 * genome ids, species history, and live speciation bookkeeping.
 *
 * In practice this is the "checkpoint" export. It is the safest default when
 * you care about reproducible continuation rather than only preserving candidate
 * genomes for later inspection.
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
    formatVersion: CURRENT_STATE_FORMAT_VERSION,
    checkpointMode: FULL_CHECKPOINT_MODE,
    neat: toJSONImpl.call(this),
    population: exportPopulation.call(this),
    speciation: serializeSpeciationCheckpoint(
      this as unknown as NeatControllerForExport,
    ),
  };
}

/**
 * Static-style helper that rehydrates a full evolutionary state previously
 * produced by {@link exportState}. Invoke this with the NEAT class (not an
 * instance) bound as `this`, e.g. `Neat.importStateImpl(bundle, fitnessFn)`.
 * It constructs a new NEAT instance using the meta data, then imports the
 * population (if present).
 *
 * This is the most complete restore path in the chapter. If a saved bundle is
 * valid, the caller gets back a fresh controller that knows both where the run
 * was in evolutionary time and which genomes were alive at that moment.
 *
 * Safety and validation:
 * - Throws if the bundle is not an object.
 * - Throws if the bundle omits the full population array.
 * - Throws if the population or species payload cannot satisfy the proper-NEAT
 *   resume contract.
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
    throw new NeatExportStateBundleValidationError('Invalid state bundle');

  const checkpointFormatVersion = resolveStateFormatVersion(stateBundle);
  if (checkpointFormatVersion > CURRENT_STATE_FORMAT_VERSION) {
    throw new NeatExportStateBundleValidationError(
      `Unsupported NEAT checkpoint format version: ${checkpointFormatVersion}.`,
    );
  }
  if (!stateBundle.neat || typeof stateBundle.neat !== 'object') {
    throw new NeatExportStateBundleValidationError(
      'Full checkpoint bundles must include serialized NEAT meta state.',
    );
  }
  if (!Array.isArray(stateBundle.population)) {
    throw new NeatExportStateBundleValidationError(
      'Full checkpoint bundles must include a population array.',
    );
  }
  if (
    checkpointFormatVersion >= CURRENT_STATE_FORMAT_VERSION &&
    stateBundle.checkpointMode !== FULL_CHECKPOINT_MODE
  ) {
    throw new NeatExportStateBundleValidationError(
      'Versioned full checkpoints must declare checkpointMode: "full".',
    );
  }
  if (
    checkpointFormatVersion >= CURRENT_STATE_FORMAT_VERSION &&
    (!stateBundle.speciation || typeof stateBundle.speciation !== 'object')
  ) {
    throw new NeatExportStateBundleValidationError(
      'Versioned full checkpoints must include speciation resume state.',
    );
  }

  const neatInstance = (
    this as NeatConstructor & {
      fromJSON?: typeof fromJSONImpl;
    }
  ).fromJSON?.(stateBundle.neat, fitnessFunction);

  if (!neatInstance)
    throw new NeatExportStateControllerRestoreError(
      'Failed to create NEAT instance from JSON',
    );

  await importPopulation.call(
    neatInstance as unknown as NeatLike,
    stateBundle.population,
  );

  if (stateBundle.speciation) {
    const { default: Network } =
      await import('../../architecture/network/network');

    restoreSpeciationCheckpoint(
      neatInstance,
      stateBundle.speciation,
      Network as unknown as NetworkClass,
    );
  }

  restoreRuntimeMeta(neatInstance, stateBundle.neat.runtime);

  return neatInstance;
}

/**
 * Serialize NEAT meta (excluding the mutable population) for persistence of
 * innovation history and experiment configuration. This is sufficient to
 * recreate a blank NEAT run at the same evolutionary generation with the same
 * innovation counters, genome-id cursor, and archived species history,
 * enabling deterministic continuation when combined later with a saved
 * population.
 *
 * Use this path when the controller context matters but the population payload
 * should be stored, transferred, or versioned separately.
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
    formatVersion: CURRENT_META_FORMAT_VERSION,
    input: internal.input,
    output: internal.output,
    generation: internal.generation,
    options: internal.options,
    innovationTracker: serializeInnovationTracker(internal._innovationTracker),
    runtime: serializeRuntimeMeta(internal),
  };
}

/**
 * Static-style implementation that rehydrates a NEAT instance from previously
 * exported meta JSON produced by {@link toJSONImpl}. This does not restore a
 * population; callers typically follow up with `importPopulation` or use
 * {@link importStateImpl} for a complete restore. Population-dependent state
 * such as the live species registry remains reserved for the full checkpoint
 * bundle.
 *
 * This helper is the mirror image of `toJSONImpl()`: rebuild the controller's
 * evolution bookkeeping first, then decide separately whether the population
 * should be restored from another source.
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
  const metaFormatVersion = resolveMetaFormatVersion(neatJSON);
  if (metaFormatVersion > CURRENT_META_FORMAT_VERSION) {
    throw new NeatExportStateControllerRestoreError(
      `Unsupported NEAT meta format version: ${metaFormatVersion}.`,
    );
  }
  if (!neatJSON.innovationTracker) {
    throw new NeatExportStateControllerRestoreError(
      'Missing innovation tracker state in NEAT meta JSON',
    );
  }

  const neatInstance = new this(
    neatJSON.input,
    neatJSON.output,
    fitnessFunction,
    neatJSON.options || {},
  );

  neatInstance.generation = neatJSON.generation || 0;
  neatInstance._innovationTracker = restoreInnovationTracker(
    neatJSON.innovationTracker,
  );
  restoreRuntimeMeta(neatInstance, neatJSON.runtime);

  return neatInstance;
}
