# neat/nge-dna

Error thrown when one DNA payload is missing required identity fields or uses an incompatible schema version.

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

## neat/nge-dna/neat.nge-dna.errors.ts

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
