# neat/export

Persistence helpers for the NEAT controller's evolutionary state.

This export chapter exists so snapshotting and rehydration stay discoverable
as their own boundary instead of living in the shared root controller folder.
The helpers here intentionally avoid importing the concrete `Neat` class
directly, which keeps the serialization flow reusable across the public
facade, tests, and static restore entrypoints.

Typical usage follows two tracks:
- export or import just the population when you only need genome payloads;
- export or import the full state when you need innovation history,
  generation counters, and population data to resume a run faithfully.

## neat/export/neat.export.ts

### exportPopulation

```ts
exportPopulation(): GenomeJSON[]
```

Export the current population (array of genomes) into plain JSON objects.
Each genome is converted via its `toJSON()` method. You can persist this
result (e.g. to disk, a database, or localStorage) and later rehydrate it
with {@link importPopulation}.

Why export population only? Sometimes you want to snapshot just the set of
candidate solutions (e.g. for ensemble evaluation) without freezing the
innovation counters or hyper-parameters.

Example:

```ts
// Assuming `neat` is an instance exposing this helper
const popSnapshot = neat.exportPopulation();
fs.writeFileSync('population.json', JSON.stringify(popSnapshot, null, 2));
```

Returns: Array of genome JSON objects.

### exportState

```ts
exportState(): NeatStateJSON
```

Convenience helper that returns a full evolutionary snapshot: both NEAT meta
information and the serialized population array. Use this when you want a
truly pause-and-resume capability including innovation bookkeeping.

Example:

```ts
const state = neat.exportState();
fs.writeFileSync('state.json', JSON.stringify(state));
// ...later / elsewhere...
const raw = JSON.parse(fs.readFileSync('state.json', 'utf8')) as NeatStateJSON;
const neat2 = Neat.importState(raw, fitnessFn); // identical evolutionary context
```

Returns: A  {@link NeatStateJSON} bundle containing meta + population.

### fromJSONImpl

```ts
fromJSONImpl(
  neatJSON: NeatMetaJSON,
  fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>,
): NeatControllerForExport
```

Static-style implementation that rehydrates a NEAT instance from previously
exported meta JSON produced by {@link toJSONImpl}. This does not restore a
population; callers typically follow up with `importPopulation` or use
{@link importStateImpl} for a complete restore.

Example:

```ts
const meta: NeatMetaJSON = JSON.parse(fs.readFileSync('neat-meta.json', 'utf8'));
const neat = Neat.fromJSONImpl(meta, fitnessFn); // empty population, same innovations
neat.importPopulation(popSnapshot); // optional
```

Parameters:
- `neatJSON` - Serialized meta (no population).
- `fitnessFunction` - Fitness callback used to construct the new instance.

Returns: Fresh NEAT instance with restored innovation history.

### GenomeJSON

JSON representation of an individual genome (network). The concrete shape is
produced by `Network#toJSON()` and re-hydrated via `Network.fromJSON()`. We use
an open record signature here because the network architecture may evolve with
plugins / future features (e.g. CPPNs, substrate metadata, ONNX export tags).

### GenomeWithSerialization

Genome with toJSON serialization method.

### importPopulation

```ts
importPopulation(
  populationJSON: GenomeJSON[],
): Promise<void>
```

Import (replace) the current population from an array of serialized genomes.
This does not touch NEAT meta state (generation, innovations, etc.) - only the
population array and implied `popsize` are updated.

Example:

```ts
const populationData: GenomeJSON[] = JSON.parse(fs.readFileSync('population.json', 'utf8'));
neat.importPopulation(populationData); // population replaced
neat.evolve(); // continue evolving with new starting genomes
```

Edge cases handled:
- Empty array => becomes an empty population (popsize=0).
- Malformed entries will throw if `Network.fromJSON` rejects them.

Parameters:
- `populationJSON` - Array of serialized genome objects.

### importStateImpl

```ts
importStateImpl(
  stateBundle: NeatStateJSON,
  fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>,
): Promise<NeatControllerForExport>
```

Static-style helper that rehydrates a full evolutionary state previously
produced by {@link exportState}. Invoke this with the NEAT class (not an
instance) bound as `this`, e.g. `Neat.importStateImpl(bundle, fitnessFn)`.
It constructs a new NEAT instance using the meta data, then imports the
population (if present).

Safety and validation:
- Throws if the bundle is not an object.
- Silently skips population import if `population` is missing or not an array.

Example:

```ts
const bundle: NeatStateJSON = JSON.parse(fs.readFileSync('state.json', 'utf8'));
const neat = Neat.importStateImpl(bundle, fitnessFn);
neat.evolve();
```

Parameters:
- `stateBundle` - Full state bundle from  {@link exportState} .
- `fitnessFunction` - Fitness evaluation callback used for new instance.

Returns: Rehydrated NEAT instance ready to continue evolving.

### InnovationMapEntry

Connection innovation map entry.

### NeatConstructor

NEAT class constructor interface.

### NeatControllerForExport

NEAT controller interface for export operations.

### NeatMetaJSON

Serialized meta information describing a NEAT run, excluding the concrete
population genomes. This allows you to persist and resume experiment context
without committing to a particular population snapshot.

### NeatStateJSON

Top-level bundle containing both NEAT meta information and the full array of
serialized genomes (population). This is what you get from `exportState()` and
feed into `importStateImpl()` to resume exactly where you left off.

### NetworkClass

Network class with static fromJSON method.

### toJSONImpl

```ts
toJSONImpl(): NeatMetaJSON
```

Serialize NEAT meta (excluding the mutable population) for persistence of
innovation history and experiment configuration. This is sufficient to
recreate a blank NEAT run at the same evolutionary generation with the same
innovation counters, enabling deterministic continuation when combined later
with a saved population.

Example:

```ts
const meta = neat.toJSONImpl();
fs.writeFileSync('neat-meta.json', JSON.stringify(meta));
// ... later ...
const metaLoaded = JSON.parse(fs.readFileSync('neat-meta.json', 'utf8')) as NeatMetaJSON;
const neat2 = Neat.fromJSONImpl(metaLoaded, fitnessFn); // empty population
```
