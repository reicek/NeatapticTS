# neat/nge-dna

Canonical envelope bridge between the NGE developmental pipeline and the
runtime {@link Network} substrate.

The NGE developmental pipeline produces a fully realized phenotype
descriptor — a JSON-serializable blueprint of modules and edges — through a
deterministic sequence of rule execution, CPPN evaluation, and substrate
placement. The bridge is the final translation step: it turns that blueprint
into a live, mutable `Network` that the rest of the NeatapticTS runtime can
train, evaluate, and serialize exactly like any classic-NEAT network.

The bridge also provides the reverse path: extracting the canonical NGE DNA
envelope back out of a serialized `Network` so that round-trips preserve
identity, substrate metadata, and fingerprint integrity.

## Why the bridge routes through NetworkJSON

Rather than constructing `Node` and `Connection` objects directly, the
bridge assembles a {@link NetworkJSON} intermediate and delegates hydration
to the canonical serializer (`fromJSONImpl`). This mirrors the genome bridge
pattern used in `genome.utils.ts` and ensures that historical identity
restoration, extension-bag attachment, and node/connection lifecycle all flow
through a single, well-tested code path. Two materializations of the same
DNA produce byte-identical `toJSON()` payloads regardless of global counter
drift because `geneId` and `innovation` are injected deterministically.

## Determinism contract

- **Node `geneId`** = module index + 1 (1-based, position-stable).
- **Connection `innovation`** = djb2-style hash of `sourceModuleId|targetModuleId`,
  always positive. Identical adjacencies always restore the same historical
  identity across round-trips.
- **Zone → node type**: `z=0` → `input`, `z=1` → `output`, `z>1` → `hidden`.
- **Squash mapping** (NGE enabled): `DenseFeedForward→relu`,
  `AttentionHead→sigmoid`, `GatedRecurrentCell→tanh`,
  `EpisodicSlot→identity`, `ModulatorBroadcaster→identity`,
  `GatingRouter→sigmoid`.

## Opt-in isolation

When `runtimeHints.ngeEnabled` is not `true`, the bridge omits the NGE
extension bag entirely and collapses every squash function to `identity`.
Classic-NEAT consumers therefore see a plain `Network` with no NGE
properties, preserving full backward compatibility. The NGE surface is
strictly opt-in.

## Extension carrier

The NGE extension carrier is stored inside `NetworkJSONExtensions.values` as:
`{ version: 1, ngeDescriptor, ngeEnvelope }`. The outer
`NetworkJSONExtensions.version` tracks the bag wrapper; the inner
`version` tracks the carrier schema so downstream consumers can migrate
independently.

```mermaid
flowchart LR
  classDef dna fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
  classDef plan fill:#0f2233,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
  classDef pheno fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
  classDef bridge fill:#1a0f1a,stroke:#ff6b9d,color:#ffd6e5,stroke-width:1.5px;
  classDef net fill:#08131f,stroke:#06d6a0,color:#bff7e6,stroke-width:1.5px;

  env["NgeDnaCanonicalEnvelope"]:::dna
  plan["buildVirtualPlan"]:::plan
  real["realizePhenotype"]:::plan
  desc["NgeRealizedPhenotypeDescriptor"]:::pheno
  mat["materializeNetworkFromPhenotype"]:::bridge
  json["NetworkJSON intermediate"]:::bridge
  net["Network"]:::net

  env -->|"seed"| plan
  plan -->|"plan, seed"| real
  real --> desc
  desc --> mat
  env --> mat
  mat --> json
  json -->|"fromJSONImpl"| net

  ext["extractCanonicalEnvelopeFromNetwork"]:::bridge
  net -->|"toJSON"| ext
  ext -->|"structuredClone"| env
```

## Background reading

The NEAT algorithm — speciation through historical markings and
complexification of topologies — is described in Stanley and Miikkulainen,
[Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02).
The CPPN substrate-encoding idea that NGE builds on is explained in
Wikipedia contributors,
[Compositional pattern-producing network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network),
and the broader neuroevolution context is covered in Wikipedia contributors,
[Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution).

## neat/nge-dna/neat.nge-dna.types.ts

### NgeAssignedRegionStrategy

Region-assignment strategies used for polyandric drone patch selection among donors.
Determines how the queen distributes writable DNA regions among secondary drone contributors.

### NgeAxisPartitionConfig

Zone partitioning configuration for one substrate axis in the unit-cube grid.
Defines the equal-sized partition count used to assign deterministic zone cells during module placement.

### NgeCppnActivationKind

CPPN activation families supported by the Phase A deterministic evaluator.

### NgeCppnEdge

One directed weighted connection inside the CPPN topology graph.
During adjacency realization, the weight is forwarded to the corresponding realized edge descriptor.

### NgeCppnEdgeInput

Loose constructor input for one directed weighted CPPN connection in a program descriptor.
Omitted `weight` fields resolve to zero during canonical CPPN edge construction.

### NgeCppnNode

One explicit non-input CPPN node.

Output-node descriptors may appear here to override the default linear,
zero-bias output nodes implied by the canonical output ids.

### NgeCppnNodeInput

Loose constructor input for one explicit non-input CPPN node in a program descriptor.
Omitted activation and bias fields resolve to conservative defaults during CPPN program construction.

### NgeCppnProgram

Full canonical CPPN program descriptor carried by the NGE DNA envelope.
Evaluated during Step 04 materialization to derive sparse adjacency between realized modules.

### NgeCppnProgramInput

Loose constructor input for one canonical CPPN program descriptor for the Phase A evaluator.
Omitted fields resolve to canonical defaults; omitted edge weights resolve to zero.

### NgeDnaCanonicalEnvelope

Full canonical NGE DNA envelope serialized by the owner-local module.

### NgeDnaModuleArchetype

DNA-level module archetype registry entry consumed during the Step 04 realization pass.
Associates a computation motif with optional coordinate injection and weight-shared cohort membership.

### NgeDnaModuleArchetypeInput

Loose constructor input for one DNA-level module archetype registry entry.

### NgeEncodingMode

Encoding modes supported by the canonical NGE DNA envelope during serialization.
`lossless` preserves all structural data; `lossy` is an opt-in flag for extreme-scale compression.

### NgeIdentityFields

Identity fields that make one NGE DNA envelope self-describing and hashable.

### NgeRealizedEdge

One realized directed adjacency edge emitted by the CPPN evaluation pass.
Carries wiring cost, residual-tap, and broadcast flags for downstream budget-aware stages.

### NgeRealizedModule

One realized module emitted by the Step 04 materialization pass.

### NgeRealizedPhenotypeDescriptor

Fully JSON-serializable realized phenotype descriptor produced at the end of Step 04.

### NgeReproductionPolicy

Fully resolved reproduction policy stored in the canonical DNA envelope.

### NgeReproductionPolicyMode

Supported reproduction modes tracked by the NGE DNA policy shelf.
Governs whether offspring arise from a single parent, multiple drone donors, or sexual crossover.

### NgeRulePass

One deterministic rule-pass record carried by the NGE DNA envelope.
Encodes a family of module placements that the rule executor unfolds deterministically during development.

### NgeRulePassKind

Supported rule-pass kinds recognized by the deterministic NGE rule executor.
Each kind implies a distinct geometry strategy applied to the module placement list during development.

### NgeRulePlacement

One requested placement emitted by a DNA rule pass during deterministic development.
Specifies the unit-cube coordinate and computation motif assigned to one future virtual module.

### NgeSchemaVersion

Branded schema version tag carried by the canonical NGE DNA envelope.

### NgeSeedPolicy

Seed governance toggles controlling sibling divergence and identical twin generation.
Determines whether offspring sharing the same DNA diverge by seed or remain exact replicas.

### NgeSubstrateBudgetOverride

Optional substrate budget override reserved for later NGE development passes.
When present, clamps the maximum node and edge counts allowed for one materialized substrate.

### NgeSubstrateConfig

Substrate contract governing the deterministic development boundary for the NGE phenotype.
Fixes the three-axis unit-cube geometry and zone-partition scheme used during module placement.

### NgeSubstrateZone

Resolved zone descriptor for one deterministic unit-cube cell in the substrate grid.
Carries the integer-grid–derived `zoneId` and the inclusive coordinate bounds for the cell.

### NgeVirtualModule

One placed module inside the in-memory deterministic virtual module plan.
Carries the archetype identity, computation motif, unit-cube coordinate, and zone assignment.

### NgeVirtualModulePlan

Deterministic in-memory module-placement plan produced by NGE rule execution.
Consumed by the Step 04 materialization pass and validated via a canonical plan fingerprint.

### NgeZonePartitionConfig

Full zone partition configuration for all three substrate axes of the unit cube.
Determines how the unit-cube space is divided into deterministic cells for module placement.

## neat/nge-dna/neat.nge-dna.ts

### NGE_DNA

Deterministic owner-local DNA boundary for Phase A schema identity and canonical encoding.

#### buildVirtualPlan

```ts
buildVirtualPlan(
  seed: number,
): NgeVirtualModulePlan
```

Build the deterministic virtual module plan for one seed.

Parameters:
- `seed` - Deterministic seed folded into the plan fingerprint.

Returns: Stable in-memory virtual module plan for the current DNA envelope.

#### compatibilityVersion

Compatibility contract version carried by the canonical envelope.

#### cppnPrograms

Canonical CPPN programs carried by the DNA envelope.

#### deserialize

```ts
deserialize(
  json: string,
): NGE_DNA
```

Deserialize one canonical JSON payload into a DNA instance.

Parameters:
- `json` - Canonical JSON payload.

Returns: New deterministic DNA instance.

#### encodingMode

Encoding mode recorded in the canonical envelope.

#### fingerprint

Deterministic SHA-256 fingerprint of the canonical envelope content.

#### fromCanonical

```ts
fromCanonical(
  envelope: NgeDnaCanonicalEnvelope,
): NGE_DNA
```

Construct one DNA instance from canonical envelope data.

Parameters:
- `envelope` - Canonical envelope whose identity and fingerprint must already be valid.

Returns: New deterministic DNA instance.

#### moduleArchetypes

Canonical module archetype registry carried by the DNA envelope.

#### realizePhenotype

```ts
realizePhenotype(
  plan: NgeVirtualModulePlan,
  seed: number,
): NgeRealizedPhenotypeDescriptor
```

Materialize one serializable phenotype descriptor from the current DNA envelope.

Parameters:
- `plan` - Deterministic virtual module plan to realize.
- `seed` - Deterministic seed folded into the phenotype fingerprint.

Returns: Fully serializable realized phenotype descriptor.

#### recomputeFingerprint

```ts
recomputeFingerprint(): string
```

Recompute the fingerprint from the current canonical content.

Returns: Updated SHA-256 fingerprint.

#### schemaVersion

Schema version carried by the canonical envelope.

#### serialize

```ts
serialize(): string
```

Serialize the canonical envelope into key-sorted JSON.

Returns: Stable JSON representation of the DNA envelope.

#### substrate

Fully resolved substrate configuration carried by the canonical envelope.

#### toCanonical

```ts
toCanonical(): NgeDnaCanonicalEnvelope
```

Materialize the full canonical envelope for serialization or inspection.

Returns: Deeply copied canonical DNA envelope.

## neat/nge-dna/neat.nge-dna.cppn.ts

### evaluateCppnProgram

```ts
evaluateCppnProgram(
  program: NgeCppnProgram,
  inputVector: readonly number[],
): readonly [number, number]
```

Evaluate one feedforward CPPN program over the canonical seven-dimensional input vector.

Parameters:
- `program` - Canonical CPPN descriptor carried by the DNA envelope.
- `inputVector` - Ordered `[x1, y1, z1, x2, y2, z2, dist]` input vector.

Returns: Fixed `[weight, enableBias]` output tuple.

## neat/nge-dna/neat.nge-dna.rules.ts

### executeRulePasses

```ts
executeRulePasses(
  passes: NgeRulePass[],
  substrateConfig: NgeSubstrateConfig,
  seed: number,
): NgeVirtualModulePlan
```

Execute one deterministic set of rule passes into an in-memory virtual module plan.

Parameters:
- `passes` - Rule passes carried by one canonical DNA envelope.
- `substrateConfig` - Canonical substrate config used for normalization and zoning.
- `seed` - Deterministic seed folded into the plan fingerprint only.

Returns: Stable virtual module plan with deterministic ordering and fingerprints.

## neat/nge-dna/neat.nge-dna.bridge.ts

### buildConnectionJsonEntry

```ts
buildConnectionJsonEntry(
  edge: NgeRealizedEdge,
  moduleIdToNodeIndex: ModuleIdToNodeIndex,
): NetworkJSONConnection
```

Build one {@link NetworkJSONConnection} from a realized edge.
The `innovation` is a deterministic positive hash of the module-id pair so the
same adjacency always restores the same historical identity.

### buildNgeExtensions

```ts
buildNgeExtensions(
  envelope: NgeDnaCanonicalEnvelope,
  descriptor: NgeRealizedPhenotypeDescriptor,
): NetworkJSONExtensions
```

Build the NGE extension bag carried inside `NetworkJSONExtensions`.

### buildNodeJsonEntry

```ts
buildNodeJsonEntry(
  module: NgeRealizedModule,
  index: number,
  runtimeHints: GenomeMaterializationRuntimeHints,
): NetworkJSONNode
```

Build one {@link NetworkJSONNode} from a realized module.
The `geneId` is the deterministic 1-based module index so round-trips stay stable.

### computeStableInnovation

```ts
computeStableInnovation(
  sourceModuleId: string,
  targetModuleId: string,
): number
```

Compute a deterministic positive integer hash from a source/target module-id pair.
Uses a djb2-style fold so identical adjacencies always yield the same innovation
number across round-trips, independent of global counter state.

### extractCanonicalEnvelopeFromNetwork

```ts
extractCanonicalEnvelopeFromNetwork(
  network: default,
): NgeDnaCanonicalEnvelope
```

Extract the canonical NGE DNA envelope previously attached to a runtime Network.

This is the reverse direction of the canonical envelope bridge: it reads the
NGE extension carrier from the network's serialized extension bag and returns a
structured clone so callers cannot mutate the stored envelope. The envelope's
`schemaVersion` and `fingerprint` survive the round-trip unchanged, which means
a `materialize → extract` cycle is a lossless identity-preserving operation.

The function throws when the Network carries no NGE extension bag — for
example, when the network was materialized without `ngeEnabled: true`, or when
it is a classic-NEAT network that was never touched by the NGE pipeline.

Parameters:
- `network` - Runtime Network produced by  {@link materializeNetworkFromPhenotype} * with `ngeEnabled: true`, or any Network whose serialized extension bag contains
 * an `ngeEnvelope` carrier.

Returns: A structured clone of the stored canonical NGE DNA envelope.

Examples:

```ts
const network = materializeNetworkFromPhenotype(envelope, plan, descriptor, {
  ngeEnabled: true,
});
const extracted = extractCanonicalEnvelopeFromNetwork(network);
console.log(extracted.fingerprint === envelope.fingerprint); // true
```

```ts
// Classic-NEAT network without NGE extension → throws.
const plainNet = materializeNetworkFromPhenotype(envelope, plan, descriptor);
try {
  extractCanonicalEnvelopeFromNetwork(plainNet);
} catch (err) {
  console.log(err instanceof NGE_DNA_BridgeError); // true
}
```

### materializeNetworkFromPhenotype

```ts
materializeNetworkFromPhenotype(
  envelope: NgeDnaCanonicalEnvelope,
  _plan: NgeVirtualModulePlan,
  descriptor: NgeRealizedPhenotypeDescriptor,
  runtimeHints: GenomeMaterializationRuntimeHints,
): default
```

Materialize a runtime {@link Network} from one realized NGE phenotype descriptor.

This is the forward direction of the canonical envelope bridge: it translates a
fully realized phenotype descriptor (modules + edges) into a live, mutable
`Network` that integrates with the rest of the NeatapticTS runtime. The bridge
routes through a {@link NetworkJSON} intermediate so the canonical network
serializer (`fromJSONImpl`) owns node/connection hydration, historical identity
restoration, and extension-bag attachment — the same pattern used by the genome
bridge in `genome.utils.ts`.

Deterministic `geneId` and `innovation` values are injected explicitly so two
materializations of the same DNA produce byte-identical `toJSON()` payloads
regardless of global counter drift. The `geneId` is the 1-based module index;
the `innovation` is a djb2-style hash of `sourceModuleId|targetModuleId`.

When `runtimeHints.ngeEnabled` is `true`, the NGE extension carrier — containing
a structured clone of both the descriptor and the envelope — is attached to the
`NetworkJSON` so downstream consumers can extract it via
{@link extractCanonicalEnvelopeFromNetwork}. When `ngeEnabled` is not `true`,
the extension bag is omitted and every squash function collapses to `identity`,
producing a plain classic-NEAT network with no NGE surface.

Parameters:
- `envelope` - Canonical NGE DNA envelope carried alongside the descriptor.
Stored inside the extension bag when NGE is enabled.
- `_plan` - Virtual module plan that produced the descriptor (reserved for
future governance checks; not required for topology materialization).
- `descriptor` - Realized phenotype descriptor whose modules and edges are
translated into network nodes and connections. Must contain at least one module.
- `runtimeHints` - Optional materialization hints. When `ngeEnabled` is not
`true` the NGE extension bag is omitted so classic NEAT consumers see a plain
Network. Defaults to `{}`.

Returns: A runtime Network whose nodes and connections mirror the descriptor.

Examples:

```ts
const network = materializeNetworkFromPhenotype(envelope, plan, descriptor, {
  ngeEnabled: true,
});
console.log(network.nodes.length); // descriptor.modules.length
```

```ts
// Classic-NEAT mode: no NGE extension bag, all squashes collapse to identity.
const classicNet = materializeNetworkFromPhenotype(envelope, plan, descriptor);
// extractCanonicalEnvelopeFromNetwork(classicNet) would throw NGE_DNA_BridgeError.
```

### ModuleIdToNodeIndex

Lookup table from the unit-cube zone coordinate to the runtime node type.

### NgeBridgeExtensionValues

Shape of the NGE extension carrier stored inside `NetworkJSONExtensions.values`.

### resolveNodeType

```ts
resolveNodeType(
  zoneCoordinate: number,
): string
```

Resolve the runtime node type from the unit-cube zone coordinate.

### resolveSquash

```ts
resolveSquash(
  computationType: "DenseFeedForward" | "AttentionHead" | "GatedRecurrentCell" | "EpisodicSlot" | "ModulatorBroadcaster" | "GatingRouter",
  runtimeHints: GenomeMaterializationRuntimeHints,
): string
```

Resolve the squash function name from the computation motif and runtime hints.

## neat/nge-dna/neat.nge-dna.errors.ts

Error thrown when one DNA payload is missing required identity fields or uses an incompatible schema version.

### NGE_DNA_BridgeError

Error thrown when the phenotype → Network bridge cannot materialize or extract
a canonical envelope, such as when a descriptor has zero modules or a Network
carries no NGE extension bag.

This error is the bridge's only failure surface. It is thrown by
{@link materializeNetworkFromPhenotype} when the descriptor is empty and by
{@link extractCanonicalEnvelopeFromNetwork} when the Network's serialized
extension bag contains no `ngeEnvelope` carrier. Callers that need to
distinguish between the two failure modes should inspect the error message.

Parameters:
- `message` - Human-readable bridge validation failure message.
- `options` - Optional native error options containing the underlying cause.

Example:

```ts
try {
  materializeNetworkFromPhenotype(envelope, plan, emptyDescriptor);
} catch (err) {
  if (err instanceof NGE_DNA_BridgeError) {
    console.log(err.message); // "Cannot materialize a Network from a phenotype descriptor with zero modules."
  }
}
```

### NGE_DNA_BudgetError

Error thrown when one resolved substrate budget exceeds the module's conservative guardrails.

### NGE_DNA_CppnError

Error thrown when one Phase A CPPN descriptor or realization dispatch contract is invalid.

### NGE_DNA_SchemaError

Error thrown when one DNA payload is missing required identity fields or uses an incompatible schema version.

### NGE_DNA_SubstrateError

Error thrown when one substrate coordinate or zone-partition input is invalid.

## neat/nge-dna/neat.nge-dna.realize.ts

### realizePhenotypeFromPlan

```ts
realizePhenotypeFromPlan(
  plan: NgeVirtualModulePlan,
  envelope: NgeDnaCanonicalEnvelope,
  seed: number,
): NgeRealizedPhenotypeDescriptor
```

Materialize one serializable phenotype descriptor from the deterministic virtual module plan.

Parameters:
- `plan` - Canonical virtual module plan emitted by Step 03 rule execution.
- `envelope` - Canonical DNA envelope carrying CPPN programs and archetype metadata.
- `seed` - Deterministic seed folded into the realized phenotype fingerprint.

Returns: Fully JSON-serializable realized phenotype descriptor.

## neat/nge-dna/neat.nge-dna.constants.ts

### NGE_DNA_CPPN_INPUT_COUNT

Fixed Phase A CPPN input count: `x1`, `y1`, `z1`, `x2`, `y2`, `z2`, `dist`.

### NGE_DNA_CPPN_OUTPUT_COUNT

Fixed Phase A CPPN output count covering `weight` and `enableBias` output channels.

### NGE_DNA_DEFAULT_BUDGET_MAX_EDGES

Default edge-budget sentinel used until development passes own stricter budgets.

### NGE_DNA_DEFAULT_BUDGET_MAX_NODES

Default node-budget sentinel used until development passes own stricter budgets.

### NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD

Inclusive absolute CPPN weight floor below which one realized edge is discarded.

### NGE_DNA_DEFAULT_RULE_PRIORITY

Default rule-pass priority applied when constructor inputs omit one explicitly.

### NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT

Default per-axis partition count for the Step 03 unit-cube substrate.

### NGE_DNA_SCHEMA_VERSION

Initial schema version for the Phase A canonical NGE DNA envelope.

## neat/nge-dna/neat.nge-dna.substrate.ts

### assignZone

```ts
assignZone(
  coord: NeatGenomeSubstrateCoordinate,
  partition: NgeZonePartitionConfig,
): string
```

Resolve one deterministic zone id for a normalized substrate coordinate.

Parameters:
- `coord` - Raw or normalized three-axis coordinate.
- `partition` - Per-axis partition configuration for the substrate grid.

Returns: Deterministic zone identifier of the form `z:x:y:z`.

### buildSubstrateFingerprint

```ts
buildSubstrateFingerprint(
  config: NgeSubstrateConfig,
): string
```

Compute the deterministic SHA-256 fingerprint of one canonical substrate configuration.

Parameters:
- `config` - Canonical substrate configuration to fingerprint.

Returns: SHA-256 fingerprint of the canonical substrate JSON.

### buildZoneMap

```ts
buildZoneMap(
  partition: NgeZonePartitionConfig,
): Map<string, NgeSubstrateZone>
```

Build the full deterministic zone map for one unit-cube substrate partition.

Parameters:
- `partition` - Per-axis partition configuration for the substrate grid.

Returns: Map from zone id to resolved zone descriptor.

### normalizeCoordinate

```ts
normalizeCoordinate(
  raw: NeatGenomeSubstrateCoordinate,
): NeatGenomeSubstrateCoordinate
```

Clamp one raw substrate coordinate into the Step 03 unit cube.

Parameters:
- `raw` - Raw three-axis coordinate to normalize.

Returns: Clamped coordinate whose axes stay within `[0, 1]`.

## neat/nge-dna/neat.nge-dna.utils.ts

### canonicalSerialize

```ts
canonicalSerialize(
  value: TValue,
): string
```

Serialize one DNA envelope into a deterministic key-sorted JSON string.

Parameters:
- `value` - Canonical value to serialize.

Returns: Stable JSON whose object keys are sorted recursively.

### computeFingerprint

```ts
computeFingerprint(
  canonical: string,
): string
```

Compute the SHA-256 fingerprint for one canonical DNA JSON string.

Parameters:
- `canonical` - Canonical JSON text produced by  {@link canonicalSerialize} .

Returns: Lowercase hexadecimal SHA-256 digest used as the envelope fingerprint.

### validateIdentity

```ts
validateIdentity(
  identity: NgeIdentityFields,
): void
```

Validate the identity shelf of one NGE DNA envelope for schema version and completeness.

Parameters:
- `identity` - Identity fields to validate.

Returns: Nothing when the identity is valid.
