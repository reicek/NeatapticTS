# neat/genome

First-class genome boundary for the proper-NEAT Step 7.1 lift.

This chapter owns the strict structural contract that sits between the NEAT
controller and the executable `Network` phenotype. Its job is deliberately
narrow in the first pass:

1. define the pure node-gene and connection-gene contract,
2. provide validated phenotype-to-genome and genome-to-phenotype adapters,
3. expose a pure genome-facing validation seam that later chapters can adopt
   without widening the public runtime `Network` surface.

Step 7.2a adds the first heredity-specific subchapter under
`genome/heredity/`, where innovation-aligned connection-gene selection can
move off the runtime crossover shelf while phenotype scaffolding stays in the
`Network` layer.

Runtime activation state, species membership, controller replay metadata,
and architecture allocation counters remain outside this boundary.

## neat/genome/genome.types.ts

### GenomeMaterializationRuntimeHints

Optional phenotype-only hints applied when materializing a `Network` from one
strict genome contract.

These values do not become genome state. They exist so checkpoint import and
other bridges can preserve non-genetic runtime metadata while the genotype
boundary stays narrow.

### NeatGenome

First-pass genotype contract for the proper-NEAT lift.

This is intentionally structural only. Replay state, species membership,
controller-owned genome ids, caches, runtime node indexes, and activation
traces stay outside this contract.

### NeatGenomeCaptureOptions

Opt-in capture settings used when projecting runtime payloads into the.
strict genome contract.

### NeatGenomeConnectionGene

Pure structural connection-gene contract owned by the NEAT genome subtree, carrying innovation identity, endpoint gene ids, weight, enabled state, and optional gater.

### NeatGenomeExtensions

Versioned extension bag reserved for beyond-paper genome traits.

Step 7.1 keeps this bag optional and empty by default so the structural
identity contract can land without immediately committing to extension
semantics. Later phases can add opt-in traits here without widening the core
node-gene and connection-gene shapes.

### NeatGenomeExtensionValues

Typed extension payload reserved for additive beyond-paper genome traits that augment canonical genes without widening the base connection or node contracts.

### NeatGenomeGatedBlockDescriptor

Descriptor for one explicit gated block stored in the extension bag.

Gated blocks preserve grouped gating intent without changing canonical
connection-gene ownership. The descriptor is therefore about identity and
checkpoint semantics, not about replacing individual connection genes.
Disabled gated connection genes still count as dormant block structure until
the connection or gater identity is removed from the genome.

### NeatGenomeNodeGene

Pure node-gene contract owned by the NEAT subtree.

Array order carries the canonical interface ordering. Runtime node indexes do
not belong here because they are phenotype-only bookkeeping rebuilt during
materialization.

### NeatGenomeNodeType

Canonical node-role literals supported by the first-pass NEAT genome boundary for input, hidden, and output nodes.

### NeatGenomeRecurrentModuleDescriptor

Descriptor for one explicit recurrent module stored in the extension bag.

The executable graph still lives in canonical node and connection genes.
This additive descriptor preserves the higher-level block identity so
checkpoints and later mutation passes can tell an intentional recurrent
module from an arbitrary cyclic subgraph. Disabling one referenced
connection gene does not retire the module by itself; the descriptor stays
valid until the referenced genes or gating ownership disappear structurally.

### NeatGenomeRecurrentModuleKind

Supported recurrent-module family identifiers tracked by the Step 7.4 temporal extension lane for LSTM, GRU, and NARX.

### NeatGenomeValidationIssue

One structured finding produced by the pure genome validator carrying a stable code, path, and human-readable message.

### NeatGenomeValidationIssueCode

Stable machine-readable issue codes produced by the pure genome validator for each detected structural violation.

### NeatGenomeValidationReport

Complete validation report returned for one strict genome contract containing all structural findings and summary counts.

## neat/genome/genome.ts

### assertValidGenomeContract

```ts
assertValidGenomeContract(
  genome: NeatGenome,
): void
```

Assert that one strict genome contract is structurally valid before crossover, mutation, import, replay, or serialization boundaries consume it.
This forwarding seam keeps barrel-level API docs explicit while implementation details remain in the utility chapter.

### createCompatibilityGenomeView

```ts
createCompatibilityGenomeView(
  source: GenomeLike | NeatGenome | RuntimeCompatibilitySource,
): GenomeLike
```

Create the compatibility-layer genome view used by legacy paths that still bridge strict genome contracts and runtime phenotypes safely.
The view preserves adapter behavior while allowing strict-genome-first internals to evolve independently.

### createGenomeFromNetwork

```ts
createGenomeFromNetwork(
  network: default,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenome
```

Capture one runtime phenotype network into the strict genome contract while preserving only explicitly modeled extension families and invariants.
This documented export keeps genome capture semantics discoverable from the chapter entrypoint.

### createGenomeFromNetworkJson

```ts
createGenomeFromNetworkJson(
  networkJson: NetworkJSON,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenome
```

Convert one versioned network JSON payload into a validated strict genome contract for deterministic NEAT-core heredity and evaluation workflows.
The conversion route is intentionally explicit at the barrel surface for docs consumers.

### createNetworkFromGenome

```ts
createNetworkFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints,
): default
```

Materialize one executable runtime phenotype from a validated strict genome contract plus optional runtime-only hints and diagnostics metadata.
Keeping this forwarder documented helps users discover genome-to-network materialization from the public chapter.

### createNetworkJsonFromGenome

```ts
createNetworkJsonFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints,
): NetworkJSON
```

Convert one strict genome contract back into the versioned network JSON payload consumed by runtime serializers and import seams consistently.
This export documents the genome-to-json bridge as a first-class persistence boundary.

### GenomeHereditySelectionContext

Pure selection context for innovation-aligned genome heredity.

This contract keeps the heredity pass structural-first: two strict genomes,
the fitness or equality policy that decides disjoint inheritance, and the
deterministic RNG that resolves matching-gene choices plus disabled-gene
re-enable behavior.

### GenomeHereditySourceParent

Stable string literal labels identifying which parent contributed a given connection gene during the genome-owned heredity selection pass.

### GenomeMaterializationRuntimeHints

Optional phenotype-only hints applied when materializing a `Network` from one
strict genome contract.

These values do not become genome state. They exist so checkpoint import and
other bridges can preserve non-genetic runtime metadata while the genotype
boundary stays narrow.

### NeatGenome

First-pass genotype contract for the proper-NEAT lift.

This is intentionally structural only. Replay state, species membership,
controller-owned genome ids, caches, runtime node indexes, and activation
traces stay outside this contract.

### NeatGenomeCaptureOptions

Opt-in capture settings used when projecting runtime payloads into the.
strict genome contract.

### NeatGenomeConnectionGene

Pure structural connection-gene contract owned by the NEAT genome subtree, carrying innovation identity, endpoint gene ids, weight, enabled state, and optional gater.

### NeatGenomeConversionError

Raised when one boundary tries to project malformed state into the strict.
genome contract.

### NeatGenomeExtensions

Versioned extension bag reserved for beyond-paper genome traits.

Step 7.1 keeps this bag optional and empty by default so the structural
identity contract can land without immediately committing to extension
semantics. Later phases can add opt-in traits here without widening the core
node-gene and connection-gene shapes.

### NeatGenomeExtensionValues

Typed extension payload reserved for additive beyond-paper genome traits that augment canonical genes without widening the base connection or node contracts.

### NeatGenomeGatedBlockDescriptor

Descriptor for one explicit gated block stored in the extension bag.

Gated blocks preserve grouped gating intent without changing canonical
connection-gene ownership. The descriptor is therefore about identity and
checkpoint semantics, not about replacing individual connection genes.
Disabled gated connection genes still count as dormant block structure until
the connection or gater identity is removed from the genome.

### NeatGenomeNodeGene

Pure node-gene contract owned by the NEAT subtree.

Array order carries the canonical interface ordering. Runtime node indexes do
not belong here because they are phenotype-only bookkeeping rebuilt during
materialization.

### NeatGenomeNodeType

Canonical node-role literals supported by the first-pass NEAT genome boundary for input, hidden, and output nodes.

### NeatGenomeRecurrentModuleDescriptor

Descriptor for one explicit recurrent module stored in the extension bag.

The executable graph still lives in canonical node and connection genes.
This additive descriptor preserves the higher-level block identity so
checkpoints and later mutation passes can tell an intentional recurrent
module from an arbitrary cyclic subgraph. Disabling one referenced
connection gene does not retire the module by itself; the descriptor stays
valid until the referenced genes or gating ownership disappear structurally.

### NeatGenomeRecurrentModuleKind

Supported recurrent-module family identifiers tracked by the Step 7.4 temporal extension lane for LSTM, GRU, and NARX.

### NeatGenomeValidationError

Raised when a strict genome contract fails structural validation, carrying a structured issue list for diagnostics-first callers.

#### issues

Structured validator findings attached to the thrown error.

### NeatGenomeValidationIssue

One structured finding produced by the pure genome validator carrying a stable code, path, and human-readable message.

### NeatGenomeValidationIssueCode

Stable machine-readable issue codes produced by the pure genome validator for each detected structural violation.

### NeatGenomeValidationReport

Complete validation report returned for one strict genome contract containing all structural findings and summary counts.

### SelectedGenomeConnectionGene

One inherited connection gene selected by the genome-owned heredity pass.

The source-parent label preserves parent provenance for runtime adapters and
future narrow seams without pushing runtime node indexing into the genome
surface.

### selectGenomeHeredityConnectionGenes

```ts
selectGenomeHeredityConnectionGenes(
  context: GenomeHereditySelectionContext,
): SelectedGenomeConnectionGene[]
```

Select inherited connection genes using only the strict genome contract.

Step 7.2a moves innovation-aligned heredity selection behind the genome
boundary without widening the runtime crossover facade. The runtime shelf
still owns node scaffolding and phenotype materialization, while this helper
owns three structural decisions:

1. collect parent connection genes by preserved innovation number,
2. resolve matching, disjoint, and excess inheritance from scores plus
   equal-mode policy,
3. apply the explicit disabled-gene re-enable rule through the inherited RNG.

Parameters:
- `context` - Pure genome heredity context.

Returns: Ordered inherited connection genes plus their source-parent labels.

Example:

```ts
const selectedGenes = selectGenomeHeredityConnectionGenes({
  parent1Genome,
  parent2Genome,
  parent1Score: 2,
  parent2Score: 1,
  equal: false,
  randomGenerator: () => 0.25,
});
```

### validateGenomeContract

```ts
validateGenomeContract(
  genome: NeatGenome,
): NeatGenomeValidationReport
```

Validate one strict genome contract and return a structured report that callers can inspect before deciding to throw validation errors.
This forwarding seam supports diagnostics-first workflows while sharing one canonical validator implementation.

## neat/genome/genome.errors.ts

### NeatGenomeConversionError

Raised when one boundary tries to project malformed state into the strict.
genome contract.

### NeatGenomeValidationError

Raised when a strict genome contract fails structural validation, carrying a structured issue list for diagnostics-first callers.

#### issues

Structured validator findings attached to the thrown error.

## neat/genome/genome.utils.ts

### assertValidGenomeContract

```ts
assertValidGenomeContract(
  genome: NeatGenome,
): void
```

Assert that one strict genome contract is valid and throw a rich validation error when any invariant fails.
This guard keeps downstream genome operators free from repetitive defensive contract checks.

Parameters:
- `genome` - Strict structural genome contract.

Returns: Nothing.

### createCompatibilityGenomeView

```ts
createCompatibilityGenomeView(
  source: GenomeLike | NeatGenome | RuntimeCompatibilitySource,
): GenomeLike
```

Create the compatibility-layer view for a runtime phenotype or strict genome.

Native explicit-innovation flows are normalized through the strict genome
contract. Deliberate fallback-innovation flows remain on the legacy runtime
edge path so compatibility can keep using endpoint-derived synthetic ids.

Parameters:
- `source` - Runtime genome or strict genome contract.

Returns: Compatibility-layer genome view.

### createGenomeFromNetwork

```ts
createGenomeFromNetwork(
  network: default,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenome
```

Convert one executable phenotype into the strict NEAT genome contract.

This is the phenotype-to-genome adapter introduced in Step 7.1. It strips
runtime-only state and keeps only structural identity plus portable gene
attributes.

Parameters:
- `network` - Executable phenotype.
- `captureOptions` - Optional opt-in runtime-to-genome extension capture settings.

Returns: Strict structural genome contract.

### createGenomeFromNetworkJson

```ts
createGenomeFromNetworkJson(
  networkJson: NetworkJSON,
  captureOptions: NeatGenomeCaptureOptions,
): NeatGenome
```

Convert one versioned network JSON payload into the strict NEAT genome contract with canonical node ordering and validated edge identities.
This conversion isolates phenotype serialization details from genome-native heredity and compatibility workflows.

Parameters:
- `networkJson` - Versioned phenotype JSON payload.
- `captureOptions` - Optional opt-in runtime-to-genome extension capture settings.

Returns: Strict structural genome contract.

### createNetworkFromGenome

```ts
createNetworkFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints,
): default
```

Materialize one executable phenotype from the strict genome contract after validation and JSON reconstruction.
Runtime-only hints and optional extension-derived knobs are applied after structural materialization completes.

Parameters:
- `genome` - Strict structural genome contract.
- `runtimeHints` - Optional phenotype-only metadata to preserve.

Returns: Executable runtime phenotype.

### createNetworkJsonFromGenome

```ts
createNetworkJsonFromGenome(
  genome: NeatGenome,
  runtimeHints: GenomeMaterializationRuntimeHints,
): NetworkJSON
```

Convert one strict genome contract into the versioned network JSON payload understood by the runtime phenotype serializer.
The mapping preserves historical identifiers so roundtrips remain deterministic for replay and checkpoint lanes.

Parameters:
- `genome` - Strict structural genome contract.
- `runtimeHints` - Optional phenotype-only metadata to preserve.

Returns: Versioned network JSON payload.

### validateGenomeContract

```ts
validateGenomeContract(
  genome: NeatGenome,
): NeatGenomeValidationReport
```

Validate one strict genome contract and return a structured report covering size, node, connection, and extension invariants.
Callers can use the report for diagnostics-first flows without throwing on first failure.

Parameters:
- `genome` - Strict structural genome contract.

Returns: Structured validation report.
