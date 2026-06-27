# NGE Core Algorithm Workstream — Phase Logs

**Status:** [WIP]

Durable done-state records for completed phases of the NGE Core Algorithm Workstream.
Detailed step/slice/VALIDATION_EVIDENCE blocks are compressed here from the plan file
to keep the plan lean and focused on active work.

---

## Phase 1 — NGE_DNA Adoption & Canonical Envelope Bridge (P1) [DONE]

**Phase objective:** Bridge the runtime `Network` phenotype to the canonical
`NgeDnaCanonicalEnvelope` so that polyandric reproduction (which requires the envelope)
can operate on racing agents that currently only hold a `Network`. This resolves P1.

**Final state:** Bridge module created with `materializeNetworkFromPhenotype` and
`extractCanonicalEnvelopeFromNetwork`. 20 bridge tests pass, 100% coverage on touched
files, 301 tests across 10 suites pass with zero regressions. JSDoc, Mermaid diagram,
and academic citations documented. Generated README reflects the bridge module.

**Artifacts produced:**

- `src/neat/nge-dna/neat.nge-dna.bridge.ts` — bridge module
- `src/neat/nge-dna/neat.nge-dna.errors.ts` — `NGE_DNA_BridgeError` class added
- `src/neat/nge-dna/neat.nge-dna.bridge.test.ts` — 20 tests, 100% coverage
- `src/neat/nge-dna/README.md` — regenerated with bridge documentation, Mermaid, citations

**Validation summary:**

- 20 bridge tests pass (bridge.ts 100/100/100/100, errors.ts 100/100/100/100)
- 87 nge-dna folder tests pass (2 suites)
- 301 tests across 10 suites pass (broader regression check, zero regressions)
- tsc: 0 diagnostics, lint: 0 errors, JSDoc: 18/18 exported symbols documented
- plan-sync gate: pass

---

### Step 01: Plan Phase 1 — NGE_DNA Adoption & Canonical Envelope Bridge [DONE]

**Step objective:** Author the remaining Step 02–07 packets for Phase 1 and produce a
boundary map of the NGE_DNA ↔ Network seam so implementation can proceed in a fresh
session.

**Delegation:** `boundary-mapper` (Tier 3) + `nge-core-scout` (Tier 3) for boundary
reconnaissance.

**Evidence:**

- Step 02–07 packets authored with red-green TDD sequence for Steps 03–05.
- Step-packet gate: pass.
- Boundary map of `src/neat/nge-dna/` ↔ `src/architecture/network/` produced (see
  "Phase 1 Boundary Map" section below).

#### Step 01 Step Packet

```yaml
phase: 1
step: 1
title: 'Plan Phase 1 — NGE_DNA Adoption & Canonical Envelope Bridge'
status: '[DONE]'
goal: 'planning'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 02 — Research the NGE_DNA ↔ Network boundary [DONE]'
skills:
  - 'plan-alignment'
  - 'nge-core-algorithm'
specialists:
  - 'nge-core-scout'
  - 'boundary-mapper'
  - 'planning-context-coordinator'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Step 02-07 packets authored with red-green slices where behavior change is involved'
  - 'step-packet gate returns pass: true'
  - 'Boundary map of src/neat/nge-dna/ ↔ src/architecture/network/ produced'
```

**Context the agent must know:**

- `NGE_DNA` lives in `src/neat/nge-dna/` and exposes `toCanonical(): NgeDnaCanonicalEnvelope`.
- `NgeDnaCanonicalEnvelope` is defined in `src/neat/nge-dna/neat.nge-dna.types.ts` and
  carries identity fields, substrate config, reproduction policy, CPPN program, and
  module plans.
- `realizePhenotypeFromPlan` in `src/neat/nge-dna/neat.nge-dna.realize.ts` materializes a
  phenotype descriptor from a plan + envelope + seed — but does NOT produce a runtime
  `Network`.
- Racing agents hold a `Network` (from `src/architecture/network/`), not an
  `NgeDnaCanonicalEnvelope`, so polyandric reproduction cannot reach them.
- The bridge must be opt-in: classic NEAT must remain unchanged when NGE is disabled.

---

### Step 02: Research the NGE_DNA ↔ Network boundary [DONE]

**Step objective:** Produce a research brief that maps the exact boundary between
`src/neat/nge-dna/` and `src/architecture/network/`, identifies what exists, what is
missing, and what the implementation must build.

**Delegation:** `boundary-mapper` (Tier 3, SUCCESS) + `nge-core-scout` (Tier 3, SUCCESS).

#### Step 02 Step Packet

```yaml
phase: 1
step: 2
title: 'Research the NGE_DNA ↔ Network boundary'
status: '[DONE]'
goal: 'researching'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 03 — Red tests for phenotype→Network bridge'
skills:
  - 'research-methodology'
  - 'nge-core-algorithm'
specialists:
  - 'nge-core-scout'
  - 'boundary-mapper'
validation:
  - 'manual evidence gate: research brief produced and recorded in plan'
acceptance_criteria:
  - 'Research brief documents the full NGE_DNA development pipeline: envelope → buildVirtualPlan → realizePhenotype → NgeRealizedPhenotypeDescriptor'
  - 'Research brief identifies the gap: NgeRealizedPhenotypeDescriptor does NOT materialize into a runtime Network'
  - 'Research brief identifies the gap: no Network → NgeDnaCanonicalEnvelope extraction path exists'
  - 'Research brief maps the existing genome↔Network bridge pattern (createNetworkFromGenome / createGenomeFromNetwork) as a precedent for the NGE bridge'
  - 'Research brief confirms the NgeDnaCanonicalEnvelope fields that must round-trip: identity (schemaVersion, compatibilityVersion, encodingMode, fingerprint), substrate, reproductionPolicy, rulePasses, cppnPrograms, moduleArchetypes'
  - 'Research brief identifies opt-in isolation boundary: how to ensure classic NEAT is unchanged when NGE is disabled'
```

#### Research Brief — NGE_DNA ↔ Network Boundary

**Evidence sources:** `boundary-mapper` (Tier 3, SUCCESS) + `nge-core-scout` (Tier 3, SUCCESS) +
direct source verification of `src/neat/nge-dna/neat.nge-dna.realize.ts`,
`src/architecture/network/network.types.ts`, cross-boundary import audit, and nondeterministic-primitive
grep. Both scouts returned consistent findings with no conflicts. All key claims verified against
static code (runtime/validation > static code > comments).

##### 1. Full NGE_DNA Development Pipeline

The pipeline is fully implemented, pure, and deterministic. Verified: zero `Math.random` / `Date.now` /
`performance.now` / `crypto.random` / `uuid` in `src/neat/nge-dna/*.ts` (grep returned no matches).

| #   | Stage                                                               | Inputs                                                             | Outputs                                                    | Determinism   | RNG usage                                                                                               |
| --- | ------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------- |
| 1   | `NGE_DNA.toCanonical()`                                             | internal `#canonicalEnvelope`                                      | `NgeDnaCanonicalEnvelope` deep clone via `structuredClone` | Deterministic | None                                                                                                    |
| 2   | `NGE_DNA.buildVirtualPlan(seed)`                                    | `#canonicalEnvelope.rulePasses`, `this.substrate` (cloned), `seed` | `NgeVirtualModulePlan`                                     | Deterministic | **Seed is NOT threaded into any RNG** — folded into `planFingerprint` SHA-256 only                      |
| 3   | `NGE_DNA.realizePhenotype(plan, seed)` → `realizePhenotypeFromPlan` | `plan`, `toCanonical()`, `seed`                                    | `NgeRealizedPhenotypeDescriptor`                           | Deterministic | **Seed is NOT threaded into any RNG** — folded into `phenotypeFingerprint` only; CPPN eval is pure math |

**Critical seed-semantics finding:** the seed NEVER affects topology. Same DNA → identical topology
regardless of seed; the seed only salts `phenotypeFingerprint` (lifecycle-identity). Red tests MUST NOT
assert seed-driven topology variation — they must assert "same DNA + same seed → identical Network" and
"different seed → same topology but different `phenotypeFingerprint`".

**CPPN weight materialization (verified at `neat.nge-dna.realize.ts:152`):** `evaluateCppnProgram` is
called inside `realizeDirectedEdges` for every ordered source→target module pair, receiving the 7-dim
input `[x1,y1,z1,x2,y2,z2,dist]` and returning `[weight, enableBias]`. Only `weight` is consumed; the
`enableBias` output channel is currently discarded. The weight is thresholded at
`NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD` (0.3 absolute) and surviving edges carry the raw CPPN weight on
`NgeRealizedEdge.weight`. **The bridge does NOT need to re-evaluate CPPNs to set connection weights — it
reads `edge.weight` directly.** The CPPN program stays in the envelope for re-derivation/mutation.

**Single-CPPN limit:** `realizeDirectedEdges` evaluates only `envelope.cppnPrograms[0]`; additional CPPN
programs are silently ignored. The bridge must document/honour this or core must extend realization.

##### 2. Gap A — NgeRealizedPhenotypeDescriptor does NOT materialize into a runtime Network

**Confirmed: NO bridge exists in either direction.** Cross-boundary import audit:

- `Get-ChildItem src/neat/nge-dna -Recurse -Filter *.ts` (excluding tests) → zero `network`/`Network` imports.
- `Get-ChildItem src/architecture/network -Recurse` (excluding tests) → zero `nge`/`NGE_DNA`/`NgeDna` references.
  The only cross-references are type-only imports in `neat.nge-dna.test.ts`.

**Descriptor sufficiency gap — `NgeRealizedPhenotypeDescriptor` is NOT sufficient to fully reconstruct a
runtime Network.** Missing fields vs the classic `NeatGenome`/`Network` contract:

| Missing field                          | Why the bridge must supply it                                                            | Default strategy                                                                                                |
| -------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Input/output node designation (`type`) | Descriptor has modules but no `input`/`hidden`/`output` marker and no input/output count | **Core-algorithm decision required before red tests** — see §6 below                                            |
| Node `bias`                            | `NgeRealizedModule` has no bias                                                          | Default 0                                                                                                       |
| Node `squash`/activation               | `computationType` is a motif, not a classic squash string                                | Bridge must define `computationType → squash` map owned by nge-core-algorithm                                   |
| Numeric `innovation` on edges          | Edges carry string `sourceModuleId`/`targetModuleId` only                                | Deterministic pure function of `(sourceModuleId, targetModuleId)` (stable hash or sorted-edge sequential index) |
| `enabled` flag                         | Surviving edges are implicitly enabled                                                   | Map present → enabled=true                                                                                      |
| `gaterGeneId`                          | NGE edges have no gating                                                                 | Default null                                                                                                    |
| Connection `gain`                      | NGE edges have only `weight`                                                             | Default 1                                                                                                       |

##### 3. Gap B — No Network → NgeDnaCanonicalEnvelope extraction path exists

**Confirmed absent.** No extraction function exists. The critical round-trip gap: the descriptor carries
only `dnaFingerprint` (a hash), NOT the envelope constituents (`substrate`, `reproductionPolicy`,
`rulePasses`, `cppnPrograms`, `moduleArchetypes`). Extraction cannot rebuild the envelope from the
descriptor alone.

**Required extraction strategy:** The bridge must retain the source `NgeDnaCanonicalEnvelope` (or its
serializable form) as opt-in NGE metadata attached to the Network, and extraction reads it back. This
preserves the polyandric reproduction path (P1) which needs the full envelope. Re-deriving constituents
from a runtime Network is infeasible — they are NGE-only constructs with no Network analogue.

##### 4. Genome ↔ Network Bridge Precedent (createNetworkFromGenome / createGenomeFromNetwork)

Located in `src/neat/genome/genome.utils.ts`. Verified signatures:

- `createNetworkFromGenome(genome: NeatGenome, runtimeHints: GenomeMaterializationRuntimeHints = {}): Network`
  — path: `genome → createNetworkJsonFromGenome(genome, runtimeHints) → NetworkJSON → fromJSONImpl(json) → Network`.
  After structural materialization, applies `applyNgePrimitiveModuleMaterialization` which attaches
  `_ngePrimitiveModules` when `runtimeHints.ngeEnabled === true`.

- `createGenomeFromNetwork(network: Network, captureOptions: NeatGenomeCaptureOptions = {}): NeatGenome`
  — path: `network → toJSONImpl.call(network) → NetworkJSON → createGenomeFromNetworkJson(json, captureOptions) → NeatGenome`.

**Contract established by the precedent (the contract the NGE bridge must follow):**

- **Intermediate payload:** `NetworkJSON` is the canonical intermediate. Both directions route through it.
- **Extension carrier:** `NetworkJSONExtensions.values` (`{ version: number, values: Record<string, unknown> }`,
  verified at `network.types.ts:1137`) carries non-structural metadata that survives round-trip without
  Network needing to understand it. `NeatGenomeExtensions.values` is the parallel on the genome side.
- **Preserved across round-trip:** node gene identities (`geneId`), connection genes (`from`/`to`/`weight`/
  `enabled`), `topologyIntent`, `input`/`output` dimensions, dropout, extension families.
- **Lost across round-trip:** runtime-only state (activation values, training state, RNG state, pruning
  config, stochastic depth schedules). Runtime hints are phenotype-only.

**Implication for the NGE bridge:**

- `NgeRealizedPhenotypeDescriptor → NetworkJSON (with NGE fields in extensions) → Network`
- `Network → NetworkJSON → NgeRealizedPhenotypeDescriptor (reconstructed from nodes/connections + extensions)`

##### 5. Schema Mapping Tables

**5a. NgeRealizedPhenotypeDescriptor → Network structures**

| Descriptor field                  | Network structure                         | Mapping                                                                                                                        |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `modules[].moduleId`              | `NetworkJSONNode.geneId` (or extension)   | String→numeric deterministic mapping required;moduleId is stable `${rulePassIndex}:${kind}:${archetypeId}:${placementOrdinal}` |
| `modules[].archetypeId`           | `NetworkJSONExtensions.values`            | No native node field                                                                                                           |
| `modules[].computationType`       | `NetworkJSONNode.squash`                  | Bridge must define `computationType → squash` map (nge-core-algorithm owns)                                                    |
| `modules[].coordinate`            | `NetworkJSONExtensions.values`            | NGE-only; no native node coordinate                                                                                            |
| `modules[].zoneId`                | `NetworkJSONExtensions.values`            | NGE-only; no native node zone                                                                                                  |
| `modules[].receivesCoordinates`   | `NetworkJSONExtensions.values`            | NGE-only runtime flag                                                                                                          |
| `modules[].residualStreamId?`     | `NetworkJSONExtensions.values`            | Descriptor-only; classic Network has no residual stream                                                                        |
| `modules[].weightSharedCohortId?` | `NetworkJSONExtensions.values`            | Descriptor-only; classic Network has no shared-weight primitive                                                                |
| `modules[].archetypeParams?`      | `NetworkJSONExtensions.values`            | Opaque; bridge must not interpret unknown keys                                                                                 |
| `edges[].sourceModuleId`          | `NetworkJSONConnection.from` (node index) | Via module→geneId→index mapping                                                                                                |
| `edges[].targetModuleId`          | `NetworkJSONConnection.to` (node index)   | Via module→geneId→index mapping                                                                                                |
| `edges[].weight`                  | `NetworkJSONConnection.weight`            | Direct carry — already CPPN-materialized                                                                                       |
| `edges[].isResidualTap`           | `NetworkJSONExtensions.values`            | NGE-only cost flag                                                                                                             |
| `edges[].isModulatorBroadcast`    | `NetworkJSONExtensions.values`            | NGE-only cost flag                                                                                                             |
| `edges[].wiringCost`              | `NetworkJSONExtensions.values`            | NGE-only budget value                                                                                                          |
| `residualStreamAssignments`       | `NetworkJSONExtensions.values`            | Descriptor-only; mirror existing `NeatGenomeExtensionValues.residualStreams` shelf                                             |
| `weightSharedCohortAssignments`   | `NetworkJSONExtensions.values`            | Descriptor-only; mirror existing `NeatGenomeExtensionValues.weightSharedCohorts` shelf                                         |
| `phenotypeFingerprint`            | `NetworkJSONExtensions.values`            | Topology-level round-trip fidelity check                                                                                       |
| `dnaFingerprint`                  | `NetworkJSONExtensions.values`            | Envelope-level round-trip fidelity check                                                                                       |
| `seed`                            | `NetworkJSONExtensions.values`            | Identity only; does NOT affect topology                                                                                        |

**5b. Network structures → NgeDnaCanonicalEnvelope fields**

| Envelope field                                  | Round-trips through Network?                    | Carrier                                                                       |
| ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `schemaVersion` (`NgeSchemaVersion`, `'A.1.0'`) | No native field                                 | `NetworkJSONExtensions.values`                                                |
| `compatibilityVersion`                          | No native field                                 | `NetworkJSONExtensions.values`                                                |
| `encodingMode` (`'lossless'\|'lossy'`)          | No native field                                 | `NetworkJSONExtensions.values`                                                |
| `fingerprint`                                   | No native field                                 | `NetworkJSONExtensions.values`                                                |
| `substrate` (`NgeSubstrateConfig`)              | No native field                                 | `NetworkJSONExtensions.values` — required to re-derive plan                   |
| `reproductionPolicy` (`NgeReproductionPolicy`)  | No native field                                 | `NetworkJSONExtensions.values` — required for polyandric reproduction (P1/P5) |
| `rulePasses` (`NgeRulePass[]`)                  | No native field                                 | `NetworkJSONExtensions.values` — required to re-derive plan                   |
| `cppnPrograms` (`NgeCppnProgram[]`)             | No native field                                 | `NetworkJSONExtensions.values` — required to re-derive edges and for mutation |
| `moduleArchetypes` (`NgeDnaModuleArchetype[]`)  | Partial — `computationType` maps to node squash | `NetworkJSONExtensions.values` — required to re-derive module decoration      |

**Conclusion:** The canonical envelope is almost entirely non-round-trippable through Network native
fields. The bridge MUST use `NetworkJSONExtensions.values` as the carrier for all envelope fields. The
recommended extension schema: `{ version: 1, ngeDescriptor: NgeRealizedPhenotypeDescriptor, ngeEnvelope:
NgeDnaCanonicalEnvelope }` inside `extensions.values`, following the `NeatGenomeExtensions` precedent.

##### 6. Determinism Requirements

1. **Same DNA → identical topology** (seed is fingerprint-only, verified). The bridge's topology
   determinism contract reduces to "same DNA → same Network topology".
2. **Same DNA + same seed → bitwise-identical Network** (activation output must also be bitwise-identical
   per the workstream determinism contract).
3. **Deterministic innovation assignment** — must be a pure function of `(sourceModuleId, targetModuleId)`
   so round-trip is stable.
4. **Deterministic `computationType → squash` mapping** — owned by nge-core-algorithm, must default to
   classic NEAT squash when NGE is disabled.
5. **Round-trip implicit-state risk (out of scope for Phase 1 descriptor bridge):** if a Network grows via
   morph/lifecycle passes (Phase B/C), the extracted envelope must capture post-growth rulePasses/
   archetypes or the next development cycle will not reproduce the same checkpoints. This is the
   "experience stream" half of the determinism contract and is flagged as a future-phase dependency.
6. **No nondeterministic primitives found** in the pipeline. Map iteration is restricted to `.get()`
   lookups and array reduces; CPPN topological sort uses a deterministic FIFO queue; `toSorted` is stable.

##### 7. Opt-In Isolation Boundary

**Mechanism:** the `ngeEnabled?: boolean` flag on `GenomeMaterializationRuntimeHints`
(`src/neat/genome/genome.types.ts:389`), gated inside `applyNgePrimitiveModuleMaterialization`
(`genome.utils.ts:701-714`).

```ts
function applyNgePrimitiveModuleMaterialization(
  runtimeNetwork,
  extensions,
  runtimeHints,
): void {
  if (runtimeHints.ngeEnabled !== true) {
    Reflect.deleteProperty(runtimeNetwork, '_ngePrimitiveModules');
    return;
  }
  runtimeNetwork._ngePrimitiveModules = materializeNgePrimitiveModules(
    readModuleArchetypes(extensions),
  );
}
```

**Classic NEAT isolation guarantee:** when `ngeEnabled` is not explicitly `true`, no NGE materialization
occurs and NGE properties are deleted from the runtime Network. A standard
`new Neat(1, 1, fitnessFn, { popsize: 3, seed: 17 })` never sets `ngeEnabled`. Verified by the test at
`neat.nge-dna.test.ts:1598-1613` ("classic NEAT opt-in behavior").

**Additional isolation:** the entire `nge-*` family (`nge-dna`, `nge-evolution`, `nge-adult`, `nge-juvenile`,
`nge-collective`, `nge-assimilation`) is exported via `src/neat/nge-experimental.ts` as a separate
namespace, keeping NGE code paths outside the default Neat import graph.

**Bridge isolation requirement:** the new bridge module must follow the same `ngeEnabled` gate. A red test
must assert that the bridge called WITHOUT `ngeEnabled` does not attach any NGE-specific properties to the
runtime Network. A classic NEAT Network (no NGE metadata) must be byte-identical whether or not the bridge
module is loaded.

##### 8. Residual Streams & Weight-Shared Cohorts (descriptor-only for Phase 1)

- `residualStreamAssignments: Record<string, string[]>` — moduleIds grouped by `residualStreamId`.
  Classic Network has **no residual-stream primitive**. Must be carried as descriptor-only extension
  metadata (mirroring `NeatGenomeExtensionValues.residualStreams`).
- `weightSharedCohortAssignments: Record<string, string[]>` — moduleIds grouped by `weightSharedCohortId`.
  Classic Network has **no shared-weight primitive** (each `Connection` owns its own `weight`/`gain`).
  Must be carried as descriptor-only extension metadata (mirroring
  `NeatGenomeExtensionValues.weightSharedCohorts`).
- **Do NOT no-op them silently** — red-test that the shelves survive round-trip so the contract break is
  visible when core later adds runtime enforcement. Record runtime enforcement as an explicit
  nge-core-algorithm future-phase dependency.

##### 9. Fingerprint Round-Trip Fidelity

- `dnaFingerprint` (envelope-level): SHA-256 of `canonicalSerialize` of the full envelope content with
  `fingerprint` blanked and empty `cppnPrograms`/`moduleArchetypes` elided. **Strong check for envelope
  round-trip fidelity** — `NGE_DNA.fromCanonical` re-computes and throws `NGE_DNA_SchemaError` on mismatch.
- `phenotypeFingerprint` (topology-level): SHA-256 of `canonicalSerialize({edges, modules, seed})`.
  **Strong check for topology round-trip fidelity**, but NOT envelope contents.
- The bridge should assert both survive round-trip.

##### 10. Risk Assessment for Bridge Implementation

| Risk                                                                        | Severity | Mitigation                                                                                                                                                                                  |
| --------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input/output designation decision blocks red tests                          | HIGH     | nge-core-algorithm must choose (i) descriptor schema widening, (ii) coordinate convention (z=0→input, z=1→output), or (iii) archetype `computationType` convention BEFORE Step 03 red tests |
| Network→envelope extraction cannot rebuild constituents from descriptor     | HIGH     | Retain source envelope as attached Network metadata via `NetworkJSONExtensions.values`; do not attempt re-derivation                                                                        |
| Weight-sharing runtime gap (classic Network has no shared-weight primitive) | MEDIUM   | Carry cohorts as descriptor-only extension metadata; record runtime enforcement as future-phase dependency                                                                                  |
| Residual-stream runtime gap (same as above)                                 | MEDIUM   | Carry as descriptor-only extension metadata; record runtime enforcement as future-phase dependency                                                                                          |
| Seed-semantics false invariant (asserting seed drives topology)             | MEDIUM   | Red tests must assert actual semantics: same DNA → same topology; seed only affects fingerprint                                                                                             |
| Single-CPPN limit (`cppnPrograms[0]` only)                                  | LOW      | Document/honour in bridge; flag for core extension if multi-CPPN envelopes appear                                                                                                           |
| CPPN enable threshold hardcoded (0.3, not DNA-governed)                     | LOW      | Deterministic today; flag as hidden policy lever                                                                                                                                            |
| Bridge leaks NGE state into classic NEAT                                    | MEDIUM   | Follow `ngeEnabled` gate; red-test classic Network byte-identity                                                                                                                            |
| Module→node mapping (1:1 vs subgraph)                                       | MEDIUM   | Recommend 1:1 for initial bridge (matches genome precedent); defer subgraph materialization                                                                                                 |

##### 11. Bridge Location & Shape Recommendation

- **Location:** new `src/neat/nge-dna/neat.nge-dna.bridge.ts` (or `bridge/` subfolder if the seam grows).
  This is the first production file to cross the nge-dna ↔ network boundary — keep it narrow and
  orchestration-first.
- **Materialization direction:** `materializeNetworkFromPhenotype(envelope, plan, descriptor, runtimeHints)`
  → builds `NetworkJSON` (nodes from modules, connections from edges, NGE fields in `extensions.values`)
  → `Network.fromJSON(json)`.
- **Extraction direction:** `extractCanonicalEnvelopeFromNetwork(network)` → `network.toJSON()` → read
  `extensions.values` for the retained envelope + descriptor → return `NgeDnaCanonicalEnvelope`.
- **Module→node mapping:** 1:1 (one `NgeRealizedModule` → one `NetworkJSONNode`), matching the genome
  precedent. `computationType` determines `squash`. Defer subgraph materialization to a later step.

##### 12. Coverage Gaps (no test exists at the boundary)

- No test exercises `NgeRealizedPhenotypeDescriptor → Network` materialization.
- No test exercises `Network → NgeRealizedPhenotypeDescriptor` capture.
- No test exercises `NgeRealizedModule → Network node` mapping.
- No test exercises `NgeRealizedEdge → Network connection` mapping.
- No test exercises `residualStreamAssignments` / `weightSharedCohortAssignments` → `NetworkJSONExtensions` round-trip.
- No test exercises `NgeDnaCanonicalEnvelope` → `NetworkJSONExtensions` round-trip.
- No test exercises `NGE_DNA.realizePhenotype()` output being consumed by any Network-facing code.

The existing `neat.nge-dna.test.ts` suite covers canonical serialization, fingerprints, default resolution,
constructor overrides, budget guards, substrate coordinates, rule passes, CPPN evaluation, CPPN edge
realization, and classic NEAT opt-in — but **nothing at the network bridge boundary**.

##### 13. Acceptance Criteria Checklist

- [x] Research brief documents the full NGE_DNA development pipeline: envelope → buildVirtualPlan → realizePhenotype → NgeRealizedPhenotypeDescriptor (§1)
- [x] Research brief identifies the gap: NgeRealizedPhenotypeDescriptor does NOT materialize into a runtime Network (§2)
- [x] Research brief identifies the gap: no Network → NgeDnaCanonicalEnvelope extraction path exists (§3)
- [x] Research brief maps the existing genome↔Network bridge pattern as a precedent (§4)
- [x] Research brief confirms the NgeDnaCanonicalEnvelope fields that must round-trip: identity (schemaVersion, compatibilityVersion, encodingMode, fingerprint), substrate, reproductionPolicy, rulePasses, cppnPrograms, moduleArchetypes (§5b)
- [x] Research brief identifies opt-in isolation boundary (§7)

##### 14. Open Decisions for Step 03 (red tests) — nge-core-algorithm ownership

1. **Input/output designation strategy** — must be chosen before red tests: (i) descriptor schema widening
   (add `inputCount`/`outputCount` + role field), (ii) coordinate convention (z=0→input, z=1→output), or
   (iii) archetype `computationType` convention (dedicated Input/Output motif).
2. **Deterministic innovation assignment** — stable hash of `(sourceModuleId, targetModuleId)` vs
   canonical sequential index over the sorted edge list.
3. **`computationType → squash` map** — owned by nge-core-algorithm; must default classic NEAT squash
   when NGE is disabled.
4. **Round-trip fidelity assertions** — red tests must assert both `dnaFingerprint` and
   `phenotypeFingerprint` survive round-trip.
5. **Residual-stream/weight-sharing shelves** — red-test that shelves survive round-trip even when
   classic Network does not enforce them.

**Stop condition status:** Not blocked. The NGE_DNA schema CAN round-trip a Network without losing
topology data PROVIDED the bridge uses `NetworkJSONExtensions.values` as the carrier for non-structural
fields and retains the source envelope as attached metadata. No escalation to `00-helping` needed.

---

### Step 03: Red tests for phenotype→Network bridge [DONE]

**Step objective:** Write failing tests that define the expected behavior of the
phenotype→Network bridge and the Network→canonical envelope extraction, including
determinism and opt-in isolation.

**Delegation:** `unit-test-writer` (Tier 3, SUCCESS).

#### Step 03 Step Packet

```yaml
phase: 1
step: 3
title: 'Red tests for phenotype→Network bridge'
status: '[DONE]'
goal: 'red-testing'
tdd_sequence: 'red-green'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 04 — Implement the canonical envelope bridge'
skills:
  - 'red-test-contracts'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
specialists:
  - 'unit-test-writer'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-dna/.*bridge.*test'
acceptance_criteria:
  - 'Red test: materializeNetworkFromPhenotype(envelope, plan, seed) returns a runtime Network with correct node and connection count'
  - 'Red test: extractCanonicalEnvelopeFromNetwork(network) returns an NgeDnaCanonicalEnvelope with matching identity fields'
  - 'Red test: Network → envelope → Network round-trip preserves node count, connection count, and topology'
  - 'Red test: same DNA + same seed produces identical Network topology (determinism)'
  - 'Red test: classic NEAT Network (no NGE) is unchanged when the bridge is not invoked (opt-in isolation)'
  - 'Red test: envelope → Network materialization throws on invalid/empty phenotype descriptor'
  - 'All red tests fail for the right reason (missing implementation, not syntax error or bad fixture)'
```

#### Red-test evidence:

- **File created:** `src/neat/nge-dna/neat.nge-dna.bridge.test.ts` (18 tests across 5 describe blocks)
- **Bridge API defined by tests:**
  - `materializeNetworkFromPhenotype(envelope, plan, descriptor, runtimeHints?) → Network`
  - `extractCanonicalEnvelopeFromNetwork(network) → NgeDnaCanonicalEnvelope`
- **Validation command:** `npx jest --config=jest.config.mjs --no-cache --testPathPatterns="src/neat/nge-dna/.*bridge.*test"`
- **Red result:** Exit code 1 — `TS2307: Cannot find module './neat.nge-dna.bridge' or its corresponding type declarations.` All 18 tests blocked by the missing module (correct red-phase failure — missing implementation, not syntax error or bad fixture).
- **Test breakdown:**
  - `materializeNetworkFromPhenotype` (6 tests): node count, connection count, zero-module throw, z=0→input designation, z=1→output designation, extension carrier schema
  - `extractCanonicalEnvelopeFromNetwork` (2 tests): schemaVersion matching, fingerprint matching
  - `round-trip fidelity` (6 tests): node count, connection count, phenotypeFingerprint, dnaFingerprint, residualStreamAssignments, weightSharedCohortAssignments
  - `determinism` (2 tests): same DNA+seed → identical topology, same DNA different seeds → same topology
  - `opt-in isolation` (2 tests): no NGE extensions when ngeEnabled not true, classic Network unchanged
- **Open decisions resolved in test header comments:**
  1. Input/output designation: z=0 → input, z=1 → output, 0<z<1 → hidden
  2. Deterministic innovation: stable hash of (sourceModuleId, targetModuleId)
  3. computationType → squash map: DenseFeedForward→relu, AttentionHead→sigmoid, GatedRecurrentCell→tanh, EpisodicSlot→identity, ModulatorBroadcaster→identity, GatingRouter→sigmoid
  4. Extension carrier: { version: 1, ngeDescriptor, ngeEnvelope } inside NetworkJSONExtensions.values
  5. Residual-stream/weight-sharing shelves: descriptor-only extension metadata, tested for round-trip survival
  6. Seed semantics: seed is fingerprint-only, never affects topology
- **Fixture strategy:** Real NGE_DNA pipeline (buildVirtualPlan + realizePhenotype) for topology tests; synthetic descriptor for shelf round-trip tests; empty descriptor for error-path test

---

### Step 04: Implement the canonical envelope bridge [DONE]

**Step objective:** Implement the canonical envelope bridge: phenotype→Network
materialization, Network→envelope extraction, round-trip preservation, and opt-in
isolation. All red tests from Step 03 must pass.

**Delegation:** `implementation-executor` (Tier 3) + `implementation-pattern-scout` (Tier 3).

#### Step 04 Step Packet

```yaml
phase: 1
step: 4
title: 'Implement the canonical envelope bridge'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 05 — Green validation and coverage guard'
skills:
  - 'implementation-standards'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
specialists:
  - 'implementation-executor'
  - 'implementation-pattern-scout'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-dna/.*bridge.*test'
  - 'npm run lint'
acceptance_criteria:
  - 'All red tests from Step 03 pass'
  - 'materializeNetworkFromPhenotype(envelope, plan, seed) produces a runtime Network with correct topology'
  - 'extractCanonicalEnvelopeFromNetwork(network) produces a valid NgeDnaCanonicalEnvelope'
  - 'Network → envelope → Network round-trip preserves identity and substrate metadata'
  - 'Same DNA + same seed produces identical Network topology'
  - 'Classic NEAT is unchanged when the bridge is not invoked (opt-in isolation)'
  - '100% statements, branches, functions, lines on all touched src/neat/ files'
  - 'No backward-compatibility wrappers or dual-path code (old placeholder removed in same step)'
slices:
  - slice_id: '04-red-tests'
    title: 'Red tests for phenotype→Network bridge and Network→envelope extraction'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'src/neat/nge-dna/neat.nge-dna.bridge.test.ts'
    acceptance_criteria:
      - 'Red test: materializeNetworkFromPhenotype(envelope, plan, seed) returns a runtime Network with correct node and connection count'
      - 'Red test: extractCanonicalEnvelopeFromNetwork(network) returns NgeDnaCanonicalEnvelope with matching identity fields'
      - 'Red test: Network → envelope → Network round-trip preserves topology'
      - 'Red test: same DNA + same seed produces identical Network topology (determinism)'
      - 'Red test: classic NEAT unchanged when bridge not invoked (opt-in isolation)'
      - 'Red test: invalid/empty phenotype descriptor throws'
      - 'All tests fail for the right reason (missing implementation)'
    parallelizable: false
    dependencies: []
    next_slice: '04-impl'
    evidence: 'Completed in Step 03 — 18 red tests in src/neat/nge-dna/neat.nge-dna.bridge.test.ts, all failing with TS2307 (module not found)'
  - slice_id: '04-impl'
    title: 'Implement phenotype→Network materialization and Network→envelope extraction'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 8
    files_to_change:
      - 'src/neat/nge-dna/neat.nge-dna.bridge.ts'
      - 'src/neat/nge-dna/neat.nge-dna.bridge.test.ts'
      - 'src/neat/nge-dna/neat.nge-dna.errors.ts'
    acceptance_criteria:
      - 'materializeNetworkFromPhenotype(envelope, plan, descriptor, runtimeHints?) returns a runtime Network with correct topology'
      - 'extractCanonicalEnvelopeFromNetwork(network) returns a valid NgeDnaCanonicalEnvelope'
      - 'Identity fields (schemaVersion, fingerprint) preserved across round-trip via extension carrier'
      - 'Substrate metadata preserved from network'
      - 'Invalid/empty phenotype descriptor throws NGE_DNA_BridgeError'
      - 'No placeholder, no-op, or dual-path code remains in the bridge module'
    parallelizable: false
    dependencies:
      - '04-red-tests'
    next_slice: '04-green'
    evidence: |
      Created src/neat/nge-dna/neat.nge-dna.bridge.ts with materializeNetworkFromPhenotype
      and extractCanonicalEnvelopeFromNetwork. Added NGE_DNA_BridgeError to errors.ts.
      Added 2 coverage tests (hidden-node designation, missing-extension throw) to bridge.test.ts.
      All 20 bridge tests pass. bridge.ts + errors.ts at 100% statements/branches/functions/lines.
      tsc: OK. ESLint: 0 errors. Prettier: clean. JSDoc: 18/18 exported symbols documented.
  - slice_id: '04-green'
    title: 'Green validation: round-trip, determinism, opt-in isolation, coverage guard'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 4
    files_to_change:
      - 'src/neat/nge-dna/neat.nge-dna.bridge.test.ts'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - 'Network → envelope → Network round-trip preserves node count, connection count, and topology'
      - 'Same DNA + same seed produces identical Network topology (determinism)'
      - 'Classic NEAT Network is unchanged when the bridge is not invoked (opt-in isolation)'
      - '100% statements, branches, functions, lines on all touched src/neat/ files'
      - 'Coverage guard passes on src/neat/nge-dna/neat.nge-dna.bridge.ts'
    parallelizable: false
    dependencies:
      - '04-impl'
    next_slice: null
```

#### PlanUpdate — Step 04 slice `04-impl` [DONE]

```yaml
PlanUpdate:
  slice_id: '04-impl'
  changed_files:
    - 'src/neat/nge-dna/neat.nge-dna.bridge.ts'
    - 'src/neat/nge-dna/neat.nge-dna.bridge.test.ts'
    - 'src/neat/nge-dna/neat.nge-dna.errors.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run quality:folder -- --folder=src/neat/nge-dna'
    - 'npx prettier --check src/neat/nge-dna/neat.nge-dna.bridge.ts src/neat/nge-dna/neat.nge-dna.bridge.test.ts src/neat/nge-dna/neat.nge-dna.errors.ts'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=src/neat/nge-dna/.*bridge.*test'
      expected_exit: 0
      result: '20 passed, 20 total'
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=src/neat/nge-dna/'
      expected_exit: 0
      result: '87 passed, 87 total; bridge.ts 100/100/100/100; errors.ts 100/100/100/100'
  coverage_guard:
    files:
      - 'src/neat/nge-dna/neat.nge-dna.bridge.ts'
      - 'src/neat/nge-dna/neat.nge-dna.errors.ts'
    summary: 'statements:100, branches:100, functions:100, lines:100'
  rollback:
    - 'git checkout -- src/neat/nge-dna/neat.nge-dna.bridge.ts src/neat/nge-dna/neat.nge-dna.bridge.test.ts src/neat/nge-dna/neat.nge-dna.errors.ts'
  next: 'Run 05-green-testing slice 04-green: round-trip, determinism, opt-in isolation, coverage guard'
```

VALIDATION_EVIDENCE (04-impl):

- tsc: OK (0 diagnostics)
- ESLint: 0 errors across 12 files (quality:folder)
- JSDoc: 18/18 exported symbols documented
- Prettier: All matched files use Prettier code style
- Coverage: bridge.ts 100/100/100/100, errors.ts 100/100/100/100
- Tests: 20 bridge tests pass, 87 nge-dna folder tests pass
- quality:folder FAIL is pre-existing (6 missing sibling test files for cppn/errors/realize/rules/substrate/utils — none are the new bridge file)

#### PlanUpdate — Step 04 slice `04-green` [DONE]

```yaml
PlanUpdate:
  slice_id: '04-green'
  changed_files:
    - 'plans/NGE_Core_Algorithm_Workstream.plans.md'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=src/neat/nge-dna/.*bridge.*test'
      expected_exit: 0
      result: '20 passed, 20 total; bridge.ts 100/100/100/100'
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=src/neat/nge-dna/'
      expected_exit: 0
      result: '87 passed, 87 total (2 suites); bridge.ts 100/100/100/100; errors.ts 100/100/100/100'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: '0 diagnostics'
    - command: 'npm run lint'
      expected_exit: 0
      result: '0 errors'
    - command: 'npm run quality:folder -- --folder=src/neat/nge-dna'
      expected_exit: 0
      result: 'FAIL — 6 pre-existing missing sibling test files (cppn, errors, realize, rules, substrate, utils); 0 in-folder TS diagnostics, 0 ESLint errors, 18/18 JSDoc, 0 lcov entries below 100%'
  coverage_guard:
    files:
      - 'src/neat/nge-dna/neat.nge-dna.bridge.ts'
      - 'src/neat/nge-dna/neat.nge-dna.errors.ts'
    summary: 'statements:100, branches:100, functions:100, lines:100 (both files)'
  acceptance_criteria_check:
    round_trip: 'PASS — Network→envelope→Network round-trip preserves node count, connection count, topology'
    determinism: 'PASS — same DNA + same seed produces identical Network topology'
    opt_in_isolation: 'PASS — classic NEAT Network unchanged when bridge not invoked'
    coverage_100: 'PASS — bridge.ts 100/100/100/100, errors.ts 100/100/100/100'
  gate_results:
    plan_sync: 'pass: true — all WIP plans registered in README and Roadmap'
  next: 'Step 04 complete — advance to Step 05 (green validation and coverage guard) or phase compression'
```

VALIDATION_EVIDENCE (04-green):

- tsc: OK (0 diagnostics)
- lint: OK (0 errors, exit 0)
- Bridge focused tests: 20 passed, 20 total — bridge.ts 100/100/100/100
- Full nge-dna folder tests: 87 passed, 87 total (2 suites) — bridge.ts 100/100/100/100, errors.ts 100/100/100/100
- quality:folder: 0 TS diagnostics, 0 ESLint errors, 18/18 JSDoc, 0 lcov below 100%; FAIL is pre-existing (6 missing sibling test files predate this slice)
- plan-sync gate: pass: true
- Acceptance criteria: all 4 criteria verified (round-trip, determinism, opt-in isolation, 100% coverage)
- NOTE: Plan acceptance_criteria text mentions signature (descriptor, seed?) but actual API is (envelope, plan, descriptor, runtimeHints?) — red tests were authoritative. Flagged for 01-planning reconciliation.

---

### Step 05: Green validation and coverage guard [DONE]

**Step objective:** Validate the canonical envelope bridge implementation against the
acceptance criteria, verify 100% coverage on all touched src/neat/ files, and confirm
no regressions in existing NGE_DNA tests.

**Delegation:** `coverage-guard` (Tier 3) + `code-quality-auditor` (Tier 3).

#### Step 05 Step Packet

```yaml
phase: 1
step: 5
title: 'Green validation and coverage guard'
status: '[DONE]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 06 — Document the bridge contract'
skills:
  - 'test-coverage-analysis'
  - 'reproducibility-contracts'
specialists:
  - 'coverage-guard'
  - 'code-quality-auditor'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-dna/.*bridge.*test'
  - 'npm run lint'
acceptance_criteria:
  - 'All targeted bridge tests pass with zero failures'
  - '100% statements, branches, functions, lines on all touched src/neat/ files'
  - 'Coverage guard passes on src/neat/nge-dna/neat.nge-dna.bridge.ts'
  - 'Lint exits with code 0'
  - 'No regressions in existing src/neat/nge-dna/ tests'
```

#### PlanUpdate — Step 05 [DONE]

Claim: 05-green-testing @ 2026-06-27T15:16:15Z

**Broader test run (no regressions):**

- `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=src/neat/nge-dna/.*test|src/neat/genome/.*utils.*test|src/architecture/network/.*serialize.*test`
- Result: 10 suites passed, 301 tests passed, 0 failures, 0 regressions.
- nge-dna folder: 2 suites, 87 tests passed (includes 20 bridge tests).
- genome utils + architecture/network serialize: 8 suites, 214 tests passed.

**Coverage guard on touched src/neat/ files:**

- `src/neat/nge-dna/neat.nge-dna.bridge.ts`: 100% statements, 100% branches, 100% functions, 100% lines.
- `src/neat/nge-dna/neat.nge-dna.errors.ts`: 100% statements, 100% branches, 100% functions, 100% lines.

**TypeScript compilation:**

- `npx tsc --noEmit -p tsconfig.json`: 0 diagnostics, exit 0.
- `npx tsc --noEmit -p tsconfig.test.json`: 27 pre-existing duplicate-identifier errors in 3 racing_curriculum test files (simulation-worker.coevolution.test.ts, simulation-worker.evolution.protocol.test.ts, simulation-worker.independent-genomes.test.ts). These predate this workstream and are NOT regressions.

**Lint:**

- `npm run lint`: exit 0, 0 errors.

**Quality:folder — src/neat/nge-dna:**

- TypeScript: 0 in-folder diagnostics across 12 files.
- ESLint: 0 errors across 12 files.
- JSDoc: 18/18 exported symbols documented across 8 files.
- Coverage: 8 lcov entries matched, 0 below 100% line coverage.
- Pre-existing: 6 source modules missing sibling .test.ts files (cppn.ts, errors.ts, realize.ts, rules.ts, substrate.ts, utils.ts) — these predate this workstream. bridge.ts correctly has its sibling test file.

**Quality:folder — src/neat/genome:**

- TypeScript: 0 in-folder diagnostics across 10 files.
- ESLint: 0 errors across 10 files.
- JSDoc: 17/17 exported symbols documented across 4 files.
- Tests: 0 source modules missing sibling .test.ts file.
- Pre-existing: 3 coverage deficits (genome.errors.ts 28.57%, genome.utils.ts 4.17%, genome.heredity.ts 5.08%) — these predate this workstream; genome folder was not touched by Step 04.

**Bridge barrel export verification:**

- `materializeNetworkFromPhenotype` and `extractCanonicalEnvelopeFromNetwork` are exported from `src/neat/nge-dna/neat.nge-dna.bridge.ts` via `export function`.
- The nge-dna folder barrel (`neat.nge-dna.ts`) does not re-export the bridge functions. This is consistent with other auxiliary modules (substrate.ts, cppn.ts) that are also not re-exported from the barrel. The nge-dna folder is not yet part of the public API surface (not exported from `nge-experimental.ts`). No barrel re-export was required by the Step 04 plan. This may be addressed in Step 06 (Document the bridge contract) when the bridge becomes part of the public API.

**Plan-sync gate:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`: PASS, 0 errors, 0 warnings.
- Tier-1 `plan-sync` gate: pass=true.

**Slice-level gate object:**

```json
{
  "pass": true,
  "slice_id": "05-green-broader",
  "evidence": {
    "coverage_summary": {
      "statements": 100,
      "branches": 100,
      "functions": 100,
      "lines": 100
    },
    "test_results": "10 suites / 301 tests passed (nge-dna: 87, genome+arch: 214)"
  },
  "fixHint": "n/a",
  "owner": "05-green-testing"
}
```

---

### Step 06: Document the bridge contract [DONE]

**Step objective:** Document the canonical envelope bridge contract with JSDoc on the
public API and regenerate docs so the generated README reflects the new bridge module.

**Delegation:** `docs-example-writer` (Tier 3).

#### Step 06 Step Packet

```yaml
phase: 1
step: 6
title: 'Document the bridge contract'
status: '[DONE]'
goal: 'documenting'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 07 — Compress Phase 1 into logs'
skills:
  - 'educational-docs'
specialists:
  - 'docs-example-writer'
validation:
  - 'npm run docs'
acceptance_criteria:
  - 'JSDoc on materializeNetworkFromPhenotype documents parameters, return type, and determinism contract'
  - 'JSDoc on extractCanonicalEnvelopeFromNetwork documents parameters, return type, and round-trip guarantee'
  - 'Generated README for src/neat/nge-dna/ reflects the bridge module'
  - 'npm run docs exits with code 0'
```

#### Documentation evidence:

- Added file-level JSDoc to `neat.nge-dna.bridge.ts` with educational chapter
  introduction: why the bridge exists, NetworkJSON routing rationale,
  determinism contract (geneId, innovation, zone→type, squash mapping),
  opt-in isolation boundary, extension carrier schema, and a Mermaid
  flowchart showing forward (envelope→plan→descriptor→materialize→Network)
  and reverse (Network→extract→envelope) paths.
- Enhanced `materializeNetworkFromPhenotype` JSDoc with conceptual context,
  determinism contract details, opt-in isolation explanation, and a
  classic-NEAT-mode example showing the identity-squash collapse.
- Enhanced `extractCanonicalEnvelopeFromNetwork` JSDoc with round-trip
  guarantee documentation and a failure-path example showing
  `NGE_DNA_BridgeError` thrown on a non-NGE network.
- Enhanced `NGE_DNA_BridgeError` JSDoc in `neat.nge-dna.errors.ts` with
  error surface description, cross-references to both bridge functions,
  and a usage example.
- Added academic citations: Stanley & Miikkulainen (NEAT, 2002), Wikipedia
  CPPN, Wikipedia Neuroevolution — following existing repo citation patterns.
- Mermaid diagram uses Astro Bird visual palette (dark canvas, blue-led
  structural lines, pink accent for bridge, green for runtime Network).
- Ran `npm run docs` — exit code 0, README regenerated with bridge module
  section including all functions, the Mermaid diagram, and academic references.
- Ran `npm run quality:folder -- --folder=src/neat/nge-dna` — JSDoc: 18/18
  exported symbols documented, TypeScript: 0 diagnostics, ESLint: 0 errors,
  coverage: 100% for matched files. (6 pre-existing missing sibling test
  files are out of scope for Step 06.)
- Ran `npx tsc --noEmit -p tsconfig.json` — exit code 0, no type errors.

**Decision:** Step 06 documentation complete. All acceptance criteria met.
Bridge module, Mermaid diagram, academic citations, and examples now appear
in the generated `src/neat/nge-dna/README.md`.

---

### Step 07: Compress Phase 1 into logs [DONE]

**Step objective:** Compress the completed Phase 1 history into the logs file, replace
verbose step transcripts with compact [DONE] markers, and verify the phase-compression
and step-packet gates pass before advancing to Phase 2.

**Actions taken:**

- Created `plans/NGE_Core_Algorithm_Workstream.logs.md` with all detailed Phase 1
  step/slice/VALIDATION_EVIDENCE blocks moved from the plan file.
- Replaced verbose Phase 1 step content in the plan file with compact [DONE] markers
  and a reference to the logs file.
- Kept Phase 1 header, goal, and status as [DONE] in the plan file.
- Marked Step 07 as [DONE] and Phase 1 as [DONE].

---

### Phase 1 Boundary Map — NGE_DNA ↔ Network Seam

**What exists (current state):**

| Component                           | Location                                   | Status                                                                                                                                          |
| ----------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `NGE_DNA` class                     | `src/neat/nge-dna/neat.nge-dna.ts`         | Complete — exposes `toCanonical()`, `fromCanonical()`, `serialize()`, `deserialize()`, `buildVirtualPlan(seed)`, `realizePhenotype(plan, seed)` |
| `NgeDnaCanonicalEnvelope` type      | `src/neat/nge-dna/neat.nge-dna.types.ts`   | Complete — `NgeIdentityFields & { substrate, reproductionPolicy, rulePasses, cppnPrograms, moduleArchetypes }`                                  |
| `NgeIdentityFields`                 | `src/neat/nge-dna/neat.nge-dna.types.ts`   | Complete — `schemaVersion, compatibilityVersion, encodingMode, fingerprint`                                                                     |
| `NgeRealizedPhenotypeDescriptor`    | `src/neat/nge-dna/neat.nge-dna.types.ts`   | Complete — `modules, edges, residualStreamAssignments, weightSharedCohortAssignments, phenotypeFingerprint, dnaFingerprint, seed`               |
| `realizePhenotypeFromPlan`          | `src/neat/nge-dna/neat.nge-dna.realize.ts` | Complete — produces `NgeRealizedPhenotypeDescriptor` from plan + envelope + seed                                                                |
| `NeatGenome` strict genome contract | `src/neat/genome/genome.ts`                | Complete — precedent pattern with `createNetworkFromGenome` / `createGenomeFromNetwork`                                                         |
| `Network` class                     | `src/architecture/network/network.ts`      | Complete — runtime network with `toJSON()`, `serialize()`, `clone()`                                                                            |
| `NetworkJSON` serialization         | `src/architecture/network/serialize/`      | Complete — verbose JSON contract with format version 4                                                                                          |

**What was missing (the gap — P1, now resolved):**

| Gap                                                          | Impact                                                                                                                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `NgeRealizedPhenotypeDescriptor` → `Network` materialization | The NGE development pipeline produces a serializable phenotype descriptor but never materializes it into a runtime Network. Racing agents cannot use NGE-developed networks.         |
| `Network` → `NgeDnaCanonicalEnvelope` extraction             | No path exists to construct a canonical DNA envelope from a live Network. Polyandric reproduction (which requires the envelope) cannot operate on racing agents that hold a Network. |
| Round-trip verification                                      | No guarantee that Network → envelope → Network preserves identity and substrate metadata.                                                                                            |
| Determinism proof                                            | No proof that same DNA + same seed produces identical Network topology through the bridge.                                                                                           |
| Opt-in isolation proof                                       | No test verifying classic NEAT is unchanged when the NGE bridge is not invoked.                                                                                                      |

**What was built:**

1. **`materializeNetworkFromPhenotype(envelope, plan, descriptor, runtimeHints?) → Network`** —
   a function in `src/neat/nge-dna/neat.nge-dna.bridge.ts` that takes the realized
   phenotype descriptor (produced by `NGE_DNA.realizePhenotype`) and materializes it
   into a runtime `Network`. This follows the precedent of `createNetworkFromGenome`
   which converts a genome contract into a Network via JSON reconstruction.

2. **`extractCanonicalEnvelopeFromNetwork(network) → NgeDnaCanonicalEnvelope`** — a
   function that extracts identity fields, substrate metadata, and reproduction
   policy from a live Network and constructs a valid canonical envelope. This follows
   the precedent of `createGenomeFromNetwork` which captures a runtime Network into a
   strict genome contract.

3. **Round-trip guarantee** — the bridge preserves identity (schemaVersion,
   compatibilityVersion, encodingMode, fingerprint) and substrate metadata across
   Network → envelope → Network.

4. **Determinism** — same DNA + same seed produces bitwise-identical Network
   topology through the bridge.

5. **Opt-in isolation** — the bridge module is only imported and invoked when NGE is
   enabled. Classic NEAT code paths do not import or call the bridge.

**Key architectural constraint:** The bridge does NOT introduce dual-path code or
backward-compatibility wrappers. No placeholder or no-op remains in the NGE_DNA
development pipeline (No Deferred Cleanup Policy satisfied).
