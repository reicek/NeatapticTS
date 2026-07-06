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
  LEGACY_CHECKPOINT_FORMAT_VERSION,
  CURRENT_META_FORMAT_VERSION,
  CURRENT_STATE_FORMAT_VERSION,
  FULL_CHECKPOINT_MODE,
  LIGHT_CHECKPOINT_MODE,
} from './neat.export.types';
import type {
  GenomeControllerCarrier,
  GenomeJSON,
  GenomeWithSerialization,
  NeatCheckpointRestoreOptions,
  NeatControllerForExport,
  NeatLightCheckpointExportOptions,
  NeatLightStateJSON,
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
  NeatCheckpointRestoreOptions,
  NeatLightCheckpointExportOptions,
  NeatLightMetaJSON,
  NeatLightStateJSON,
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
 * Typical usage follows three tracks:
 * - export or import just the population when you only need genome payloads;
 * - export or import a light checkpoint when you want to restart from retained
 *   elites without claiming exact future replay;
 * - export or import the full state when you need innovation history,
 *   generation counters, and population data to resume a run faithfully.
 *
 * A useful way to read this chapter is as a pause-and-resume ladder:
 * - `exportPopulation()` and `importPopulation()` move only candidate genomes
 * - `toJSONImpl()` and `fromJSONImpl()` move only controller meta state
 * - `exportLightState()` and `importLightStateImpl()` move retained elites plus
 *   bootstrap controller metadata for best-effort restart
 * - `exportState()` and `importStateImpl()` combine both layers into one full
 *   resume bundle
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
 * - **Light checkpoints** are the restart surface. They keep a curated elite
 *   subset plus enough bootstrap metadata to repopulate through the ordinary
 *   evolution path, but they intentionally omit replay-critical runtime state.
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
 * Both full and light checkpoint bundles also reserve a top-level
 * `extensions` bag for downstream metadata. That pocket exists so consumers
 * such as a downstream application can attach namespaced descriptors without
 * redefining the checkpoint semantics that this chapter owns.
 *
 * Read the chapter in this order:
 * - start with `exportPopulation()` and `importPopulation()` when you only need
 *   candidate genomes,
 * - then read `exportLightState()` and `importLightStateImpl()` when you need a
 *   smaller restart artifact built around retained elites,
 * - continue to `toJSONImpl()` and `fromJSONImpl()` when you need controller
 *   metadata without the live population,
 * - finish with `exportState()` and `importStateImpl()` when you need a full
 *   pause-and-resume checkpoint.
 *
 * ```mermaid
 * flowchart TD
 *   Runtime[Live NEAT controller] --> PopOnly[Population-only snapshot]
 *   Runtime --> MetaOnly[Meta-only checkpoint]
 *   Runtime --> LightState[Light checkpoint]
 *   Runtime --> FullState[Full state bundle]
 *   PopOnly --> ImportPop[Replace population in an existing controller]
 *   MetaOnly --> ImportMeta[Rebuild controller bookkeeping]
 *   LightState --> Restart[Import retained elites and refill later]
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

  return exportSelectedPopulation(internal, internal.population);
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

function exportSelectedPopulation(
  controller: NeatControllerForExport,
  selectedPopulation: GenomeControllerCarrier[],
): GenomeJSON[] {
  const genomeCaptureOptions = resolveGenomeCaptureOptions(controller);

  return selectedPopulation.map((genome, genomeIndex) => {
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

function resolveRetainedEliteCount(
  requestedEliteCount: number,
  availableGenomeCount: number,
): number {
  if (!Number.isInteger(requestedEliteCount) || requestedEliteCount < 1) {
    throw new NeatExportPopulationValidationError(
      'Light checkpoint export requires eliteCount to be a positive integer.',
    );
  }

  return Math.min(requestedEliteCount, availableGenomeCount);
}

function selectRetainedElitePopulation(
  population: GenomeControllerCarrier[],
  retainedEliteCount: number,
): GenomeControllerCarrier[] {
  return population
    .map((genome, populationIndex) => ({
      genome,
      populationIndex,
      score:
        typeof genome.score === 'number'
          ? genome.score
          : Number.NEGATIVE_INFINITY,
    }))
    .toSorted(
      (leftEntry, rightEntry) =>
        rightEntry.score - leftEntry.score ||
        leftEntry.populationIndex - rightEntry.populationIndex,
    )
    .slice(0, retainedEliteCount)
    .map((entry) => entry.genome);
}

function resolveRestartPopulationSize(
  controller: NeatControllerForExport,
): number {
  return typeof controller.options.popsize === 'number'
    ? controller.options.popsize
    : controller.population.length;
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
 * Callers may attach a top-level `extensions` bag after export for downstream,
 * non-core metadata. That bag is intentionally outside the strict resume
 * contract: import validation still decides exact versus best-effort behavior
 * from the checkpoint-owned fields in `neat`, `population`, and `speciation`.
 *
 * Example:
 *
 * ```ts
 * const state = neat.exportState();
 * state.extensions = {
 *   myApp: {
 *     memoryBankId: 'memory-bank-1',
 *   },
 * };
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
 * Export a light checkpoint containing only retained elite genomes plus
 * bootstrap controller metadata.
 *
 * This is the best-effort restart sibling of {@link exportState}. It keeps a
 * curated high-quality subset of the current population and the original
 * restart-scale population target, but it intentionally omits replay-critical
 * innovation, speciation, and runtime metadata.
 *
 * Callers may attach a top-level `extensions` bag after export for downstream
 * metadata that should travel with the bundle. That reserved surface is for
 * namespaced add-ons, not for overriding the light checkpoint's bootstrap
 * contract.
 *
 * @example
 * ```ts
 * const checkpoint = neat.exportLightState({ eliteCount: 4 });
 * checkpoint.extensions = {
 *   myApp: {
 *     branchId: 'draft-1',
 *   },
 * };
 * ```
 *
 * @param exportOptions Export policy describing how many elite genomes to keep.
 * @returns Light checkpoint bundle for approximate restart.
 */
export function exportLightState(
  this: NeatLike,
  exportOptions: NeatLightCheckpointExportOptions,
): NeatLightStateJSON {
  const internal = this as unknown as NeatControllerForExport;
  const retainedEliteCount = resolveRetainedEliteCount(
    exportOptions.eliteCount,
    internal.population.length,
  );
  const retainedElitePopulation = selectRetainedElitePopulation(
    internal.population,
    retainedEliteCount,
  );

  return {
    formatVersion: CURRENT_STATE_FORMAT_VERSION,
    checkpointMode: LIGHT_CHECKPOINT_MODE,
    neat: {
      input: internal.input,
      output: internal.output,
      generation: internal.generation,
      options: internal.options,
    },
    population: exportSelectedPopulation(internal, retainedElitePopulation),
    restartPopulationSize: resolveRestartPopulationSize(internal),
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
 * Any top-level `extensions` bag is treated as downstream metadata only. It is
 * allowed to travel with the bundle, but it does not weaken the strict checks
 * around replay-critical speciation and runtime state.
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
 * @param restoreOptions Explicit restore-mode override. Defaults to strict exact resume.
 * @returns Rehydrated NEAT instance ready to continue evolving.
 */
export async function importStateImpl(
  this: NeatConstructor,
  stateBundle: NeatStateJSON,
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
  restoreOptions?: NeatCheckpointRestoreOptions,
): Promise<NeatControllerForExport> {
  assertStateBundleObject(stateBundle);

  const checkpointFormatVersion = validateFullCheckpointBundle(stateBundle);
  const neatMeta = stateBundle.neat as NeatMetaJSON;
  const runtimeMeta = readCheckpointRuntimeMeta(neatMeta);
  const requireStrictReplayResume = shouldRequireStrictReplayResume(
    checkpointFormatVersion,
    restoreOptions,
  );

  assertFullCheckpointResumeState(
    stateBundle,
    runtimeMeta,
    requireStrictReplayResume,
  );

  const neatInstance = createCheckpointNeatInstance(
    this,
    neatMeta,
    fitnessFunction,
  );

  await importPopulation.call(
    neatInstance as unknown as NeatLike,
    stateBundle.population,
  );

  await restoreCheckpointSpeciationIfPresent(
    neatInstance,
    stateBundle.speciation,
  );

  restoreRuntimeMeta(neatInstance, runtimeMeta);

  return neatInstance;
}

/**
 * Static-style helper that rehydrates a controller from a light checkpoint.
 *
 * Light checkpoints preserve only bootstrap controller metadata plus a retained
 * elite subset, so this restore path rebuilds a compatible controller, imports
 * the retained genomes, and then reapplies the original restart-scale
 * population target without claiming exact replay.
 *
 * Any top-level `extensions` bag is preserved as user-owned metadata rather
 * than part of the restart contract. Import therefore ignores that bag while it
 * validates the light checkpoint-owned bootstrap fields.
 *
 * @param stateBundle Light checkpoint bundle produced by {@link exportLightState}.
 * @param fitnessFunction Fitness evaluation callback used for the new instance.
 * @returns Rehydrated NEAT instance ready for approximate restart.
 */
export async function importLightStateImpl(
  this: NeatConstructor,
  stateBundle: NeatLightStateJSON,
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
): Promise<NeatControllerForExport> {
  assertLightStateBundleObject(stateBundle);
  validateLightCheckpointBundle(stateBundle);

  const lightBootstrapMeta = stateBundle.neat as NeatLightStateJSON['neat'];
  const neatInstance = createLightCheckpointInstance(
    this,
    lightBootstrapMeta,
    fitnessFunction,
  );

  // Step 1: Reapply the saved generation marker before importing elites.
  neatInstance.generation =
    typeof lightBootstrapMeta.generation === 'number'
      ? lightBootstrapMeta.generation
      : 0;

  // Step 2: Import only the retained elites from the light bundle.
  await importPopulation.call(
    neatInstance as unknown as NeatLike,
    stateBundle.population,
  );

  // Step 3: Restore the intended future population target after import.
  neatInstance.options.popsize = stateBundle.restartPopulationSize;

  return neatInstance;
}

function assertStateBundleObject(stateBundle: NeatStateJSON): void {
  if (!stateBundle || typeof stateBundle !== 'object') {
    throw new NeatExportStateBundleValidationError('Invalid state bundle');
  }
}

function validateFullCheckpointBundle(stateBundle: NeatStateJSON): number {
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

  return checkpointFormatVersion;
}

function readCheckpointRuntimeMeta(
  neatMeta: NeatMetaJSON,
): NonNullable<NeatMetaJSON['runtime']> | undefined {
  return neatMeta.runtime && typeof neatMeta.runtime === 'object'
    ? neatMeta.runtime
    : undefined;
}

function shouldRequireStrictReplayResume(
  checkpointFormatVersion: number,
  restoreOptions: NeatCheckpointRestoreOptions | undefined,
): boolean {
  const restoreMode = restoreOptions?.restoreMode ?? 'strict';

  return (
    checkpointFormatVersion >= CURRENT_STATE_FORMAT_VERSION &&
    restoreMode === 'strict'
  );
}

function assertFullCheckpointResumeState(
  stateBundle: NeatStateJSON,
  runtimeMeta: NonNullable<NeatMetaJSON['runtime']> | undefined,
  requireStrictReplayResume: boolean,
): void {
  if (!requireStrictReplayResume) {
    return;
  }
  if (!stateBundle.speciation || typeof stateBundle.speciation !== 'object') {
    throw new NeatExportStateBundleValidationError(
      'Versioned full checkpoints must include speciation resume state.',
    );
  }
  if (!runtimeMeta) {
    throw new NeatExportStateControllerRestoreError(
      'Versioned full checkpoints must include runtime resume state for exact restore.',
    );
  }

  const missingExactResumeRuntimeFields =
    getMissingExactResumeRuntimeFields(runtimeMeta);

  if (missingExactResumeRuntimeFields.length > 0) {
    throw new NeatExportStateControllerRestoreError(
      `Versioned full checkpoints must include exact-resume runtime fields: ${missingExactResumeRuntimeFields.join(', ')}.`,
    );
  }
}

function createCheckpointNeatInstance(
  neatConstructor: NeatConstructor,
  neatMeta: NeatMetaJSON,
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
): NeatControllerForExport {
  const neatInstance = (
    neatConstructor as NeatConstructor & {
      fromJSON?: typeof fromJSONImpl;
    }
  ).fromJSON?.(neatMeta, fitnessFunction);

  if (!neatInstance) {
    throw new NeatExportStateControllerRestoreError(
      'Failed to create NEAT instance from JSON',
    );
  }

  return neatInstance;
}

async function restoreCheckpointSpeciationIfPresent(
  neatInstance: NeatControllerForExport,
  speciationCheckpoint: NeatStateJSON['speciation'],
): Promise<void> {
  if (!speciationCheckpoint) {
    return;
  }

  const { default: Network } =
    await import('../../architecture/network/network');

  restoreSpeciationCheckpoint(
    neatInstance,
    speciationCheckpoint,
    Network as unknown as NetworkClass,
  );
}

function assertLightStateBundleObject(stateBundle: NeatLightStateJSON): void {
  if (!stateBundle || typeof stateBundle !== 'object') {
    throw new NeatExportStateBundleValidationError(
      'Invalid light checkpoint bundle',
    );
  }
}

function validateLightCheckpointBundle(stateBundle: NeatLightStateJSON): void {
  const checkpointFormatVersion =
    typeof stateBundle.formatVersion === 'number'
      ? stateBundle.formatVersion
      : LEGACY_CHECKPOINT_FORMAT_VERSION;

  if (checkpointFormatVersion > CURRENT_STATE_FORMAT_VERSION) {
    throw new NeatExportStateBundleValidationError(
      `Unsupported NEAT light checkpoint format version: ${checkpointFormatVersion}.`,
    );
  }
  if (
    checkpointFormatVersion >= CURRENT_STATE_FORMAT_VERSION &&
    stateBundle.checkpointMode !== LIGHT_CHECKPOINT_MODE
  ) {
    throw new NeatExportStateBundleValidationError(
      'Versioned light checkpoints must declare checkpointMode: "light".',
    );
  }
  if (!stateBundle.neat || typeof stateBundle.neat !== 'object') {
    throw new NeatExportStateBundleValidationError(
      'Light checkpoint bundles must include serialized NEAT bootstrap state.',
    );
  }
  if (!Array.isArray(stateBundle.population)) {
    throw new NeatExportStateBundleValidationError(
      'Light checkpoint bundles must include a retained population array.',
    );
  }
  if (
    !Number.isInteger(stateBundle.restartPopulationSize) ||
    stateBundle.restartPopulationSize < 0 ||
    stateBundle.restartPopulationSize < stateBundle.population.length
  ) {
    throw new NeatExportStateBundleValidationError(
      'Light checkpoint bundles must include a restartPopulationSize that is at least the retained elite count.',
    );
  }
}

function createLightCheckpointInstance(
  neatConstructor: NeatConstructor,
  lightBootstrapMeta: NeatLightStateJSON['neat'],
  fitnessFunction: (
    network: GenomeWithSerialization,
  ) => number | Promise<number>,
): NeatControllerForExport {
  return new neatConstructor(
    lightBootstrapMeta.input as number,
    lightBootstrapMeta.output as number,
    fitnessFunction,
    resolveLightCheckpointBootstrapOptions(lightBootstrapMeta),
  );
}

function resolveLightCheckpointBootstrapOptions(
  lightBootstrapMeta: NeatLightStateJSON['neat'],
): Record<string, unknown> {
  return lightBootstrapMeta.options &&
    typeof lightBootstrapMeta.options === 'object' &&
    !Array.isArray(lightBootstrapMeta.options)
    ? lightBootstrapMeta.options
    : {};
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

function getMissingExactResumeRuntimeFields(runtimeMeta: {
  nextGenomeId?: number;
  nextConnectionInnovation?: number;
  nextNodeGeneId?: number;
  nextNodeIndex?: number;
  rngState?: number;
}): string[] {
  const missingFieldNames: string[] = [];

  if (typeof runtimeMeta.nextGenomeId !== 'number') {
    missingFieldNames.push('nextGenomeId');
  }
  if (typeof runtimeMeta.nextConnectionInnovation !== 'number') {
    missingFieldNames.push('nextConnectionInnovation');
  }
  if (typeof runtimeMeta.nextNodeGeneId !== 'number') {
    missingFieldNames.push('nextNodeGeneId');
  }
  if (typeof runtimeMeta.nextNodeIndex !== 'number') {
    missingFieldNames.push('nextNodeIndex');
  }
  if (typeof runtimeMeta.rngState !== 'number') {
    missingFieldNames.push('rngState');
  }

  return missingFieldNames;
}
