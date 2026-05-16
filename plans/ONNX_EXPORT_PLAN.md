# ONNX Export / Import Plan for NeatapticTS

**Status:** [WIP]

## 0. Purpose

Provide a clear, incremental roadmap from the current minimal MLP ONNX export/import toward standards-compliant ONNX serialization, checker-backed validation, runtime parity, and named external import subsets across feed-forward, recurrent, convolutional, attention, quantized, and extensible custom-op surfaces, while preserving NeatapticTS evolutionary features where feasible.

Execution status summary:

- Completed: Phases 0, 1, 2, 3, 4, 5, 6
- Planned: Phases 7–9

Roadmap alignment note:

- The current ONNX priority is to preserve the closed recurrent and spatial stop lines honestly while preparing the next advanced-graph tranche as an explicit planned contract rather than an implicit support expansion.
- Phase 4 is now closed for the current conservative spatial subset; any future spatial expansion should strengthen metadata and import shape discipline without diluting the Phase 3 supported-subset acceptance work.
- `plans/NEATchat.plans.md` remains gated on this file naming, testing, and documenting an honest recurrent seed-import subset rather than a heuristic best-effort surface.
- In the agreed serial pre-NGE sequence, this file is the immediate handoff after the archived Track 1 stop line in `plans/completed/Memory_Optimization.md` and before `plans/Evolution_Training_Interoperability_Contracts.md` becomes active.

## Scope

- Preserve the closed recurrent and spatial stop lines honestly before widening external import claims.
- Keep the supported subset explicit: same-family export/import, single-step self-recurrence, the current heuristic LSTM/GRU family, the closed conservative spatial subset only where tests prove parity and clean fallback behavior, and the closed Phase 5 advanced-graph subset only where explicit mappings and tests prove parity or honest fallback behavior.
- Treat any future spatial extension as secondary work that must strengthen metadata and import shape checks without broadening support claims prematurely.
- Leave Phases 7-9 planned until roadmap priority explicitly selects the next ONNX lane after the closed recurrent, spatial, advanced-graph, and optimization stop lines.

## Current state

### [DONE] Completed baseline

- Phases 0, 1, 2, 3, 4, 5, and 6 are complete.
- Phase 3 Step 6 is now closed with public and owner-level parity/fallback coverage for heuristic LSTM/GRU import, including honest fallback when fused recurrent reconstruction is incomplete.
- The ONNX source-owned docs now name the supported recurrent subset explicitly and document the fallback boundary instead of describing the recurrent importer as vague best-effort behavior.
- Phase 4 is now closed as a tested, source-owned spatial contract for the current ONNX subset. The closed contract includes Conv/Pool metadata, flatten audits, spatial import hardening, and safety-gated heuristic Conv auto-promotion for shared-kernel dense layouts, including the conservative multi-channel subset, unpooled stacked chains, deeper single-channel post-pool chains when each pooled tensor shape can be derived sequentially, deeper pooled multi-channel chains such as 72-50-18-2 when the tensor stays spatial and the pooled source stays compact per channel, the narrow final-stage Conv -> Pool -> Flatten reshape-bridge subset for both single-channel and compact multi-channel pooled inputs, owner-local explicit Conv mapping fallback proofs plus public roundtrip fallback coverage showing the same flatten-after-pool guards keep non-final flattened pooled consumers and representative repeated flatten-bridge chains on the honest fallback path, public explicit Conv mapping roundtrip coverage for the supported declared post-pool and final-stage compact multi-channel reshape-bridge subsets, and owner-local export coverage plus public roundtrip fallback coverage proving pooled multi-channel downstream stages that still depend on non-compact source nodes stay outside the promoted subset while preserving dense-fallback inference.
- Phase 5 is now closed as a tested, source-owned advanced-graph contract for the current ONNX subset. The closed contract includes audit-only cross-layer seams, exact same-family dense/per-neuron alias reuse, one-hop dense-family residual adds, explicit same-family concat mappings with deterministic `previous_then_source` input order, and fixed-width self-attention shadow export with honest dense fallback on import.
- Phase 6 is now closed as a tested, source-owned optimization and fidelity contract for the current ONNX subset. The closed contract includes exact unary activation emission for `Softplus`, `Softsign`, `Selu`, opset-gated `Mish`, and opset-gated `Gelu`, exporter-owned pruning of redundant Identity activation scaffolding, and a conservative internal tensor-shape ledger that rejects unsupported rank, axis, broadcast, and derived-width mismatches before model finalization.

### [PLANNED] Next frontier

- Phase 6 no longer has an active implementation stop line. The current optimization and fidelity contract is now closed, and future work should preserve its explicit support and fallback boundaries unless a narrower extension is named before code changes begin.
- Phase 7 Quantization & Precision is now the active ONNX frontier. Phase 7A is closed, the first 7B storage-FP16 lane is landed for the same-family dense and spatial subset, the 7C calibration contract is landed, and the current 7D same-family dense qlinear slice is landed with explicit float-domain bias bridges plus preserved unary activations while spatial lowering and narrow dynamic quantization guidance remain open work inside the phase.
- NEATchat remains dependency-gated even though Phase 3 is closed; the later conversational lane still needs the explicit external-seed decision described in the unlock boundary below.

### [PLANNED] Later stop lines

- Keep Phase 7-9 planned until roadmap priority explicitly selects the next ONNX lane after the closed Phase 6 contract.
- Keep NEATchat planning-only unless this file names either the approved compact recurrent seed family or the explicit non-ONNX bridge.
- Treat protobuf serialization, checker validation, ONNX Runtime parity, quantization, custom-domain policy, and external tooling as later implementation lanes that must close before this plan can claim compliance or move to `[DONE]`.

## Coverage backlog

- Add negative coverage for any future guarded warning surface so flatten-consistency mismatches stay honest if they graduate beyond metadata-only audits.
- Add negative coverage for future post-pool shape propagation so unsupported downstream spatial stages stay on the dense fallback path honestly.
- Record whether the first NEATchat-grade external seed target fits the honest recurrent subset or still requires a documented non-ONNX bridge.
- Add negative coverage for future residual and concat merge guards so width mismatches, ambiguous branch ancestry, and unstable input ordering stay on the honest fallback path.
- Add negative coverage for future shared-initializer alias detection so numerically similar but semantically distinct tensors never collapse into reused initializer names.
- Add negative coverage for future attention mapping so masked, cross-attention, malformed head partition, and reshape-order variants stay outside the supported subset honestly.
- Add negative coverage for future fusion controls so opset-incompatible activations, approximate-only mappings, and ambiguous dense decompositions stay on the baseline emission path.
- Add negative coverage for future constant folding and shape inference so unknown symbols, dynamic reshape targets, and invalid broadcast paths fail honestly instead of being folded or inferred speculatively.
- Add negative coverage for future precision lanes so unsupported FP16 or int8 requests preserve float32 emission instead of silently mixing numeric types or broadening compatibility claims.
- Add negative coverage for future quantization calibration and parameter emission so invalid scale or zero-point shapes, unsupported per-axis requests, and quantized bias-rule violations fail honestly before graph finalization.

## Immediate next steps

1. Preserve the closed flatten-after-pool contract: final-stage reshape bridges remain the only supported flattened promotion subset for single-channel and compact multi-channel pooled inputs unless a future pass names a narrower extension first.
2. Keep repeated flatten-bridge chains, explicit non-final flattened consumers, and extra-input-dependent downstream stages on the honest fallback path unless a future pass proves a narrower contract explicitly before code changes begin.
3. Preserve the closed Phase 5 advanced-graph contract: only the explicit same-family residual-add, concat, alias-reuse, and fixed-width attention subset is supported unless a future pass names a narrower extension first.
4. Keep Phase 7 Quantization & Precision as the active ONNX stop line; the next implementation target after the landed 7A, storage-FP16 7B, 7C calibration-contract, and current 7D dense qlinear slice with explicit bias bridges plus preserved unary activations is to continue Phase 7D with dense-only parity and supported-subset hardening before any dynamic-guidance or broader spatial lowering widens further.

## 1. Current Status (Implemented)

File: `src/architecture/network/network.onnx.ts`

Completed (Phase 0):

- Export (`exportToONNX`) of strictly layered, fully-connected feed‑forward MLPs (inputs → 0+ hidden layers → outputs).
- Import (`importFromONNX`) for models generated by our exporter (round‑trip reproducibility).
- Layer inference & validation: detects non-layered or missing full connectivity; enforces homogeneous activation per non-input layer.
- Supported activations mapped to ONNX: ReLU, Tanh, Sigmoid (Logistic), Identity (fallback with warning for unknown custom activations).
- Parameter export: weights + biases as ONNX initializers (float32 scalars in `float_data`).
- Graph structure assembly with a pair of nodes per layer transition (Gemm + Activation). The initial implementation emitted nodes in a non-standard `Activation -> Gemm` order, which was corrected in Phase 1 to the standard `Gemm -> Activation` sequence.
- Bias handling integrated in Gemm (alpha=1, beta=1, transB=1 attributes included).
- Single-layer perceptron edge case handled (inputs → outputs only).

## 2. Gaps vs. Initial Intent

Original plan suggested a per-node mapping (Add / MatMul explicit). Implementation instead uses Gemm (fused MatMul + Add) + Activation pairs — more compact but diverges from the bullet description. Update accepted as design choice; doc corrected accordingly.

## 3. Known Limitations (Current)

- Export remains a lightweight ONNX-like JSON model, not a full protobuf `ModelProto` binary.
- Numeric payload is still JSON-first rather than protobuf-native, and the reduced-precision subset remains narrow: storage-fp16 initializer packing and static-8bit same-family dense lowering are implemented, but they are not a blanket external-runtime compatibility claim.
- Dynamic sequence/batch semantics remain limited despite optional batch-dimension metadata.
- Mixed activations and partial connectivity are supported, but can expand graph size due to decomposition (`Gemm+Activation` per neuron + `Concat`).
- No support for: sparse tensor encoding, dropout, batch norm, arbitrary residual/skip branching graphs beyond the explicit same-family residual/concat subset, or shared-weight tying semantics beyond exact same-family alias reuse.
- Recurrent support is partial: single-step self-recurrence and heuristic LSTM/GRU emission/import exist, but dense/arbitrary recurrent topologies and true `Scan`-style temporal graphs are incomplete.
- Convolution/pooling support is no longer groundwork-only: the conservative explicit-mapping and narrow auto-promotion spatial subset is closed, but arbitrary spatial graphs still remain outside the supported contract.
- Quantized import, dynamic quantization, float8, int4, float4, int2, weight-only quantization, quantization-aware training, and arbitrary external reduced-precision graphs remain outside the current supported subset.
- Gating & NEAT-specific structural innovations ignored (cannot serialize gating semantics into ONNX yet).
- Custom activations degrade to Identity with warning-only fallback; there is still no FunctionProto or custom-domain registration path.
- Import path still relies on several naming conventions and best-effort heuristics; robustness against externally generated arbitrary ONNX graphs is limited.
- No protobuf-level checker or ONNX Runtime compatibility suite exists yet; current graph validation is limited to the exporter-owned internal shape and subset checks.

### Compliance-tier matrix

Use this matrix before describing any ONNX slice as compliant, compatible, or runtime-ready. Each later tier assumes the earlier tier is already true.

| Tier | Claim allowed | Minimum evidence | Current repo status |
| --- | --- | --- | --- |
| Tier 1 | Internal ONNX-like roundtrip | JSON-first export and import, same-family parity coverage, explicit fallback behavior, and no universal protobuf or runtime claim | This is the strongest repo-wide claim today. The current exporter and importer support the tested same-family subset and preserve honest fallback boundaries. |
| Tier 2 | Spec-shaped ONNX subset | Standard ONNX operator semantics, explicit opset floors, typed attributes and tensors, and supported-subset docs that do not imply broader runtime compatibility | Partially true for closed same-family slices, but not strong enough for a blanket repo-wide claim because the wire format is still JSON-first, the repo defaults to opset 18 rather than the current ONNX 1.22.0 `ai.onnx` opset 27 baseline, and several import paths still depend on repo-owned metadata or naming conventions. |
| Tier 3 | Binary ONNX serialization compatibility | Protobuf `ModelProto` emission, correct tensor storage fields, correct domain and opset declarations, and checker-style validation | Not implemented. There is still no protobuf serializer or checker-backed validation path. |
| Tier 4 | External runtime compatibility | Successful loading and execution in one or more real ONNX runtimes with parity coverage and clean rejection for unsupported graphs | Not implemented. There is still no ONNX Runtime compatibility suite or backend smoke gate. |
| Tier 5 | Broad external-model import compatibility | Clearly named external import subsets that do not rely on repo-only metadata for core semantics, plus robust negative coverage for malformed or unsupported graphs | Not implemented. The current import contract remains a narrow same-family subset plus explicit heuristic lanes. |

Current approved wording for this plan:

- "ONNX-like JSON export or import" for the current same-family subset.
- "Narrow reduced-precision support" for the landed storage-fp16 and static-8bit same-family dense subset.
- "Spec-shaped operator coverage" only when the exact operator family, opset floor, and fallback boundary are named in the same sentence.

Wording that remains prohibited until later tiers close:

- "100% ONNX compliant"
- "fully ONNX compatible"
- "generally runs on ONNX runtimes"
- "supports ONNX 1.22.0 low-bit types" without explicit float8, int4, float4, or int2 storage and execution support

### Completion gate for `[DONE]`

This plan is not done when the JSON-first same-family subset is strong. It moves to `[DONE]` only when the repo can make a compliance-grade claim for the declared supported subset.

Minimum closure target:

1. Tier 3 is closed for the declared supported subset: real protobuf `ModelProto` emission exists, tensor storage fields follow ONNX rules, domains and opsets are explicit, and checker-backed validation is green.
2. Tier 4 is closed for the declared supported subset: at least one real ONNX runtime loads and executes the compliant export subset, parity coverage is green against the native runtime within declared tolerances, and unsupported graphs reject cleanly.
3. At least one named external import subset is closed beyond same-family roundtrip: supported external graphs import without repo-only metadata carrying core execution semantics, and unsupported external graphs reject explicitly.
4. The source-owned docs and this plan name the targeted ONNX version and opset baseline explicitly, and any unsupported low-bit or custom-domain surfaces remain excluded rather than implied.

## 4. Design Principles Going Forward

1. Determinism: deterministic ordering of initializers and nodes to allow hashing/comparisons.
2. Spec compliance: progress from a lightweight JSON to either (a) fully spec-compliant JSON matching ONNX ModelProto schema or (b) direct protobuf encoding via an optional dependency.
3. Extensibility: pluggable activation & composite op registry mapping internal functions to ONNX standard ops or custom domains.
4. Graceful degradation: always export a valid subgraph even when certain advanced features need approximation (e.g., replace unsupported activation with Identity + metadata note).
5. Evolution fidelity: preserve NEAT-relevant metadata (innovation numbers, topology tags) in model metadata without breaking external consumers.

## 4.5 NEATchat Follow-up Unlock Boundary

`plans/NEATchat.plans.md` is a downstream consumer of this plan, but it should
not treat the current recurrent import work as ready just because recurrent
groundwork exists.

For NEATchat, this plan is considered ready only when all of the following are
true:

1. The exact external-seed subset for NEATchat is named explicitly and kept
   narrow: compact recurrent next-token models only, not arbitrary ONNX chat
   graphs.
2. The chosen recurrent import path is documented as supported rather than
   heuristic best-effort, including what is accepted, what is rejected, and
   what falls back to conversion or distillation.
3. Phase 3 Step 6 parity and robustness tests are complete for the chosen seed
   family, so NEATchat is not built on a "tests still finalizing" boundary.
4. If the chosen seed family still does not fit the supported subset cleanly,
   this plan documents the approved non-ONNX bridge or distillation path so
   NEATchat can remain honest about how imported teachers become native
   artifacts.

Until those conditions are true, NEATchat should treat this plan as a gating
dependency, not as an implementation-ready seed-import surface.

## Recommended agent + skill combo by active phase

- Phase 3 — `Plan Scout` + `onnx-work`
- Phase 4 — `Plan Scout` + `onnx-work`
- Phase 5 — `Plan Scout` + `onnx-work`
- Phase 6 — `Plan Scout` + `onnx-work`
- Phase 7 — `Plan Scout` + `onnx-work`
- Phase 8 — `Plan Scout` + `onnx-work`
- Phase 9 — `Plan Scout` + `onnx-work`

## 5. Phased Roadmap

### Phase 1 (Immediate Hardening) - COMPLETED

Implemented:

- Standard node ordering Gemm → Activation (default) with legacy ordering flag.
- `OnnxExportOptions` including metadata, batch dimension, legacy ordering.
- Metadata fields: `ir_version`, `opset_import`, `producer_name`, `producer_version`, `doc_string` (optional via flag).
- Optional batch dimension with symbolic `N`.
- Round-trip numerical equivalence tests (`onnx.roundtrip.test.ts`).
- Layer-level heterogeneous output activation difference (output layer activation can differ from hidden layers) already supported by per-layer homogeneity rule.

### Phase 2 (Feature Breadth – Feedforward Enhancements) - COMPLETED

Deliverables achieved:

1. Partial connectivity export (validation relaxation + zero-weight insertion).
2. Mixed per-neuron activations (decomposition to Gemm+Activation per neuron + Concat) with reversible import.
3. Metadata enrichment (`metadata_props.layer_sizes`).
4. CLI surface for new capabilities (`--partial`, `--mixed`).
5. Schema documentation updated (Concat node, decomposed naming, implicit sparse via zeros).

Items originally listed under Phase 2 that were intentionally deferred (moved to later phases for clearer scope slicing):

- Sparse encoding (`sparseFormat: 'csr'`).
- Post-export fusion optimization (collapse decomposed homogeneous layers).
- BatchNormalization / Dropout primitives.
- Residual & skip connections.
- Multiple inputs / outputs.

### Phase 3 (Recurrent / Temporal) - COMPLETED

**Status: [DONE] Recurrent hardening closed for the current ONNX subset**

Current execution target:

- Closed: the recurrent import surface is now backed by explicit parity and fallback tests instead of unsupported best-effort wording.
- Closed: the supported recurrent subset is documented as single-step self recurrence plus the same-family heuristic LSTM/GRU roundtrip boundary.
- Ongoing dependency note: NEATchat still remains gated on the later external-seed decision even though the recurrent hardening tranche is complete.

Immediate next implementation slice:

1. Completed: deep parity and robustness tests now cover the active fused LSTM/GRU import reconstruction boundary.
2. Completed: fallback behavior now preserves base recurrent self-connections from generic or same-family fused recurrent tensors when fused reconstruction cannot continue.
3. Completed: the nearest ONNX source-owned docs surface now explains the supported recurrent subset and fallback contract explicitly.
4. Next lane: continue Phase 4 spatial work.

Validation expectations for the current Phase 3 pass:

- Focused recurrent ONNX Jest slice: green.
- `npm run docs`: green.
- `npm run test:silent`: ONNX changes stayed green, but an unrelated existing ASCII Maze browser-env test still fails outside the ONNX boundary.

Baseline IMPLEMENTED (extended):

1. Self-recurrence support for ANY hidden layer (one or multiple) in single-step form when `allowRecurrent && recurrentSingleStep`.

- For each recurrent hidden layer `k` a previous-state input is added with a consistent naming convention: `hidden_prev_l{k}`.
- Forward path per recurrent layer: Gemm (`gemm_in_l{k}`) + recurrent Gemm (`gemm_rec_l{k}` using `R{k-1}`) -> Add (`add_recurrent_l{k}`) -> Activation (`act_l{k}`).
- Recurrent weight matrices `Rk` currently diagonal (self-connections only) but sized for future dense intra-layer recurrence.
- Metadata `recurrent_single_step` now stores JSON array of recurrent layer indices (1-based) instead of boolean.
- Importer reconstructs self-connections layer-by-layer using each `Rk` diagonal.

2. Tests added covering:

- Single hidden-layer recurrence (presence of `hidden_prev`, `R0`).
- Multi-hidden-layer scenario with recurrence only in later layer (presence of `hidden_prev_l2`, `R1` only).
- Error path for mixed activations in recurrent layer.

Deferred (Phase 3 extended):

- Dense intra-layer recurrence (non-diagonal `Rk`).
- Multi-step unrolling or ONNX `Scan` operator for dynamic sequence length.
- Jordan recurrent (output-to-hidden) connections.
- General backward / arbitrary recurrent edges beyond self; gating interactions.
- Time dimension `[seq, batch, feature]` formalization and unified state carry semantics.

LSTM / GRU Heuristic Sub-plan Progress:

1. Structural Pattern Recognition – COMPLETED (heuristic equal partition + self-connection check; metadata `lstm_groups_stub`).
2. Canonical Parameter Extraction – COMPLETED (concatenated W/R/B initializers with simplified biases).
3. ONNX Node Emission – COMPLETED (emits experimental single-step `LSTM` / `GRU` nodes alongside unfused Gemm path; no pruning yet).
4. Metadata & Fallback – COMPLETED (`lstm_emitted_layers`, `gru_emitted_layers`, `rnn_pattern_fallback`).
5. Import Path Extension – COMPLETED (reconstructs Layer.lstm / Layer.gru using emitted tensors; best-effort, silent skip on mismatch).
6. Testing – COMPLETED:

- Unit tests for LSTM/GRU emission presence (initializers + node types) under controlled synthetic layer partitions.
- Public LSTM round-trip regression now verifies that native `Layer.lstm` reconstruction keeps the exported gate biases and recurrent self-weights instead of silently falling back to default native-layer wiring when fused import rewires the hidden slice.
- Round-trip reconstruction tests verifying gate weight and bias mapping fidelity within tolerance (1e-9) and self-connection restoration.
- Negative and fallback tests (near-miss sizes recorded as `rnn_pattern_fallback`, incomplete self-connections skip emission).
- Import robustness tests proving that missing fused recurrent tensors fall back honestly to the layered baseline rather than dropping recurrence silently.

7. Deferred (post-step 6):

- Peephole & projection support.
- Proper ONNX-compliant bias splitting (Wb/Rb) and gate ordering normalization.
- Graph pruning/fusion of redundant unfused paths when fused nodes emitted (feature-flagged optimization).
- Multi-layer stacked fused LSTM/GRU with consistent sequence/time abstraction.

Phase 4 groundwork is now the active roadmap-level ONNX stop line.

### Phase 4 (Convolutional / Spatial) [DONE]

**Status: [DONE] Closed spatial contract for the current ONNX subset**

Status (cumulative so far):

1.  Conv2D Groundwork IMPLEMENTED + TESTS:

- `conv2dMappings` option + `Conv2DMapping` interface.
- Exporter emits `Conv` node with `ConvW*` / `ConvB*` initializers; attributes: `kernel_shape`, `strides`, `pads`.
- Metadata: `conv2d_layers`, `conv2d_specs`.
- Import reconstructs dense weights (approximate inverse) to preserve existing forward parity.
- Tests: emission + dimension mismatch fallback (`onnx.conv.groundwork.test.ts`).

2.  Pool2D Mapping IMPLEMENTED + TESTS:

- `pool2dMappings` option + `Pool2DMapping` interface.
- Emits `MaxPool` / `AveragePool` nodes directly after target layer output.
- Metadata: `pool2d_layers`, `pool2d_specs`.
- Import attaches metadata stub to network (`_onnxPooling`).
- Tests: pooling metadata presence (`onnx.conv.pool.validation.test.ts`).

3.  Conv Weight Sharing Validation IMPLEMENTED:

- `validateConvSharing` flag performs best-effort spatial kernel equality check.
- Metadata: `conv2d_sharing_verified`, `conv2d_sharing_mismatch` (tolerant 1e-9).
- Tests: sharing success + induced mismatch (`onnx.conv.pool.validation.test.ts`).

4.  Heuristic Conv Inference (non-intrusive) IMPLEMENTED:

- Detects simple single-channel square input with 2x2 or 3x3 kernel stride 1 producing exact dense size.
- Metadata only: `conv2d_inferred_layers`, `conv2d_inferred_specs` (does NOT emit Conv node yet).
- Test: metadata presence (`onnx.conv.infer.test.ts`).

5.  Pooling Import Attachment IMPLEMENTED:

- Import attaches `_onnxPooling` with layers & specs (no shape simulation yet).

6.  Flatten-after-Pool OPTION IMPLEMENTED:

- New export option `flattenAfterPooling` inserts `Flatten` node (axis=1) immediately after each emitted Pool.
- Metadata: `flatten_layers` (export-layer indices where flatten applied).
- Test: flatten metadata (`onnx.conv.pool.flatten.test.ts`).

7.  Pooling Shape Simulation Stub IMPLEMENTED + TESTS:

- Import now attaches `flattenLayers` and derived `virtualShapes` onto `_onnxPooling` so later phases can reason about post-pool spatial dimensions without changing numeric inference.
- Shape derivation uses explicit `conv2d_specs` first and falls back to `conv2d_inferred_specs` when only heuristic Conv metadata exists.
- Owner and roundtrip tests cover malformed optional metadata, inferred-spec fallback, multi-spec ordering, and unusable pool geometry.

8.  Flatten-after-Pool Consistency Audit IMPLEMENTED + TESTS:

- Import now attaches optional `flattenConsistency` audit records when `flatten_layers`, `layer_sizes`, and a derived flattened pooled width are all available.
- The audit compares the metadata-only flattened pooled width to the next dense consumer width without changing weights or numeric inference.
- Owner tests cover matching output-width cases, mismatched output-width cases, and hidden-layer consumers; the public roundtrip test now proves the audit survives export/import for the supported Conv + Pool + Flatten subset.

9.  Spatial Import Hardening IMPLEMENTED + TESTS:

- Conv reconstruction now zero-fills non-receptive inbound edges instead of leaving stale or invalid values on the rebuilt dense runtime layer.
- Dense tensor restoration now keeps downstream layers aligned even when an earlier hidden layer was exported as `Conv` and therefore has no `W*` tensor in the dense initializer sequence.
- Activation import now recognizes `act_conv_l*` nodes, so Conv-backed hidden layers restore their exported activation instead of silently defaulting to Identity.
- Public roundtrip coverage now proves a flatten-consistency mismatch remains audit-only: `_onnxPooling.flattenConsistency` can report `matches: false` without changing the imported network’s inference.

10. Heuristic Conv Auto-Promotion IMPLEMENTED + TESTS:

- New export option `autoPromoteInferredConv` keeps heuristic Conv inference metadata-only by default, but can upgrade inferred single-stage Conv-like layouts into real `Conv` emission when explicitly enabled.
- Promotion reuses the existing explicit Conv export path and now runs behind a shared-kernel safety gate, so only inferred layers whose dense weights already behave like Conv weight sharing are promoted.
- The current proven subset now includes the original single-channel 25-9-2 path, the conservative multi-channel 18-8-2 path, unpooled stacked 25-9-4-2 Conv-like chains, deeper single-channel pooled chains such as 25-16-4-2 and 36-25-9-1 when the exporter derives each post-pool tensor shape sequentially, deeper pooled multi-channel chains such as 32-18-2-2 and 72-50-18-2 when the pooled source stays compact per channel, and the narrow 25-16-4-2-style and 32-18-2-2-style flatten-after-pool subsets when the later Conv is the final hidden stage and export emits an explicit reshape before the later Conv.
- Broader flatten-after-pool bridges remain conservative: repeated flatten-bridge chains and earlier flattened pooled consumers stay on the honest fallback path, with the current guard stopping later Conv inference before later reshape bridges or `conv2d_inferred_layers` / `conv2d_inferred_specs` survive, while downstream dense layers that still depend on extra non-pooled inputs can still remain inference-only metadata when a spatial candidate exists but fails the safety gate.
- Owner-local tests cover safe promotion, unsafe fallback, and the new exporter helper branches; the public roundtrip suite now proves the current promoted subsets preserve inference numerically.

New Options / Metadata Added This Increment:

- `flattenAfterPooling` (export option).
- `autoPromoteInferredConv` (export option).
- Metadata key `flatten_layers` (array of layer indices with flatten bridge inserted).

Future spatial backlog beyond the closed Phase 4 contract:

- Keep flatten-consistency mismatches metadata-only for now; revisit a guarded warning surface only if heuristic Conv auto-promotion needs one.
- Multi-channel & multi-stage Conv chains (stacked conv/pool pipelines), dilation, groups, depthwise separable conv.
- Any future extension of the safety-gated `autoPromoteInferredConv` path beyond the now-proven compact per-channel pooled single-channel and multi-channel subsets and the narrow final-stage flatten-after-pool reshape-bridge subset.
- Proper spatial shape tracking & validation across Conv → Pool → Flatten boundaries.
- Activation fusion / redundant Gemm pruning after Conv introduction.
- Residual / skip spatial connections.
- Hybrid recurrent + spatial interleaving semantics.

Phase 4 closure log:

1.  Completed: import-time pooling shape simulation now records virtual (H,W,C) after each pool without changing numeric inference.
2.  Completed: flatten-after-pool sites now attach a metadata-only dense-width consistency audit without changing weights.
3.  Completed: flatten-consistency mismatches remain metadata-only for now; no guarded warning surface or weight rewrite is active in this tranche.
4.  Completed: heuristic Conv auto-promotion now exists behind `autoPromoteInferredConv` and only promotes inferred layers that pass the shared-kernel safety gate.
5.  Completed: the guarded auto-promotion path now infers the conservative multi-channel subset and keeps unsafe multi-channel shapes metadata-only.
6.  Completed: negative tests now prove flatten + pooling mismatch audits stay metadata-only and leave imported inference unchanged.
7.  Completed: negative tests now prove unsafe inferred Conv layers stay metadata-only even when `autoPromoteInferredConv` is enabled.
8.  Completed: stacked unpooled spatial-chain promotion is now proven, and pooled predecessors no longer stop later inferred Conv promotion when the exporter can derive the post-pool tensor shape explicitly.
9.  Completed: the current single-channel 25-16-4-2 pooled chain now promotes safely when the graph stays spatial, while flatten bridges and stray extra dense inputs keep the later stage metadata-only.
10. Completed: the current conservative multi-channel 32-18-2-2 pooled chain now promotes safely when the exporter can derive the pooled tensor shape and preserve the compact per-channel pooled source slice, while pooled-and-flattened bridges keep later stages metadata-only.
11. Completed: deeper single-channel pooled chains such as 36-25-9-1 now promote safely across repeated Conv → Pool stages when the exporter can derive each pooled tensor shape sequentially and preserve the compact per-channel pooled source slice.
12. Completed: deeper pooled multi-channel chains such as 72-50-18-2 now promote safely across repeated Conv → Pool stages when the exporter can derive each pooled tensor shape sequentially and preserve the compact per-channel pooled source slice.
13. Completed: the narrow final-stage single-channel Conv -> Pool -> Flatten subset now promotes safely when the exporter can derive the pooled 3x3 input shape, insert a reshape bridge before the later Conv, and keep broader flattened chains on the fallback path.
14. Completed: the narrow final-stage compact multi-channel Conv -> Pool -> Flatten subset now promotes safely for 32-18-2-2-style pooled 2x2x2 inputs when export, weight collection, and heuristic sharing validation all preserve the compact per-channel pooled source slice and insert a reshape bridge before the later Conv.
15. Completed: earlier flattened pooled consumers such as 25-16-4-3-2 are now covered by owner/public fallback proofs and stay blocked before later inferred Conv metadata survives, so the current honest flatten-after-pool contract remains final-stage only.
16. Completed: representative repeated flatten-bridge chains such as 36-25-9-1-2 and 72-50-18-2 are now covered by owner/public fallback proofs and stay blocked before later reshape bridges or inferred Conv metadata survive after the first flattened pool.
17. Completed: the repeated flatten-bridge lane is closed as a fallback-only stop line under the current architecture, because the active guards require a final hidden stage and no earlier pooling boundary before any flatten-after-pool promotion can survive.
18. Completed: owner-local explicit Conv mapping proofs now cover both a non-final flattened pooled consumer and a representative repeated flatten-bridge chain, confirming declared downstream stages obey the same final-stage/no-earlier-pooling reshape guard instead of widening the supported subset.
19. Completed: owner-local negative coverage now proves the pooled multi-channel 32-18-2-2-style downstream stage stays metadata-only when non-zero weights still depend on source nodes outside the compact per-channel pooled slice.
20. Completed: public roundtrip coverage now proves the pooled multi-channel 32-18-2-2-style downstream stage with non-compact source dependencies stays on the dense fallback path and preserves inference parity.
21. Completed: public explicit Conv mapping roundtrip coverage now mirrors the current declared spatial contract for supported post-pool and final-stage compact multi-channel reshape-bridge subsets while keeping non-final flattened consumers and repeated flatten-bridge chains on the honest fallback path.
22. Completed: Phase 4 is now closed as a tested, source-owned spatial contract for the current conservative ONNX subset.
23. Preserve the closed flatten-after-pool, repeated flatten-bridge, explicit non-final flattened, and extra-input-dependent fallback boundaries unless a future pass names a narrower spatial extension explicitly before code changes begin.
24. Next stop line: Phase 5 Advanced Graph Constructs remains planned only.

### Phase 5 (Advanced Graph Constructs) [DONE]

**Status: [DONE] Closed advanced-graph contract for the current same-family ONNX subset**

Execution goal:

1. Extend the current ONNX surface from linear or spatially annotated layered graphs into a narrow, explicitly modeled advanced-graph subset without relaxing the supported-subset honesty established in Phases 3 and 4.
2. Land reusable graph seams first: deterministic branch tensor naming, merge metadata, and shared-initializer bookkeeping that later residual, concat, and attention work can reuse.
3. Implement residual and DenseNet-like merge patterns before attention so Q/K/V branches, head split or merge operations, and output projection reuse are built on the same tested substrate.
4. Keep Phase 5 same-family first: prioritize export/import parity for source-owned graphs and documented fallback for everything outside the declared contract.

Current execution status:

- Completed: Phase 5A now preserves non-adjacent feed-forward edges as `advanced_graph_cross_layer_connections` audit metadata and re-attaches them on import without widening the layered fallback scaffold.
- Completed: Phase 5B now reuses exact dense and per-neuron initializer aliases when `includeMetadata` is enabled, records `shared_initializer_aliases`, hydrates those aliases on import, and keeps near-equal or unsupported-family tensors duplicated.
- Completed: Phase 5C/5D now support the explicit same-family merge subset: dense-family one-hop residual adds emit `Add` nodes plus `advanced_graph_residual_adds` metadata, explicit concat mappings emit deterministic `Concat -> Gemm` paths plus `advanced_graph_concat_merges` metadata, and import rebuilds those skipped feed-forward edges from the residual branch tensor or widened dense tensor tail while preserving audit metadata.
- Completed: Phase 5E/5F now add an explicit `attentionMappings` export contract for fixed-width self-attention shadow emission, record `advanced_graph_attention_blocks`, validate that same-family shadow structure on import, and preserve it as `_onnxAdvancedGraph.attentionBlocks` audit metadata while runtime inference stays on the dense fallback scaffold.
- Completed: Phase 5G now updates the ONNX chapter and this tracker to name the closed residual, concat, shared-initializer, and attention subset after focused tests, coverage guard, docs, and repo-wide validation stayed green.
- Next stop line: Phase 6 Optimization & Fidelity remains planned only.

Planned supported subset:

- Explicit residual-add patterns where both branches preserve the same exported feature width, no implicit broadcasting is required, and the importer can rebuild the additive merge deterministically.
- DenseNet-like concat patterns where branch ancestry stays acyclic, contributing tensors have deterministic input order, and the downstream consumer width matches the concatenated feature widths exactly.
- Shared-parameter reuse when multiple consumers truly reference the same tensor semantics and the exporter can reuse one initializer name without changing roundtrip fidelity.
- Narrow multi-head self-attention built from dense Q/K/V projections, score `MatMul`, optional scaling, `Softmax`, weighted-sum `MatMul`, head merge, and output projection for fixed head counts and fixed feature widths.

Explicit non-goals for the first Phase 5 closure:

- Arbitrary external ONNX branching, transformer stacks, masking families, KV-cache semantics, dynamic sequence-control graphs, and speculative attention import for graphs the exporter did not produce.
- Residual merges that depend on hidden reshape, width repair, implicit broadcast, or ambiguous branch ordering.
- Concat merges whose branch ancestry, feature ordering, or downstream consumer alignment cannot be reconstructed deterministically.
- Shared-weight aliasing for transposed, sliced, approximately equal, or decomposition-derived tensors whose semantic owner is not exact.

Recommended execution order:

1. Phase 5A - Graph seam preparation.
   - Completed: export/import now preserve cross-layer feed-forward audit metadata through `advanced_graph_cross_layer_connections` and `_onnxAdvancedGraph.crossLayerConnections`.
   - Introduce stable tensor naming and consumer bookkeeping for fan-out greater than one so later merge and attention nodes have deterministic inputs.
   - Define metadata for residual adds, concat merges, attention blocks, and shared-initializer aliases before widening export support.
   - Add owner-local recognition helpers for branch and merge candidates without changing public acceptance yet.
   - Exit condition: export can describe branch topology and alias intent deterministically even when import still falls back.
2. Phase 5B - Parameter sharing detection.
   - Completed subset: exact dense/per-neuron initializer aliases now reuse one canonical tensor name under metadata, preserve `shared_initializer_aliases` audit metadata, round-trip through import, and keep near-equal or unsupported families duplicated.
   - Detect true shared weights and biases before per-consumer decomposition clones them into distinct tensors.
   - Reuse initializer names only when tensor shape, values, orientation, and semantic owner boundary match exactly.
   - Preserve alias metadata so import and tests can distinguish deliberate sharing from accidental duplication.
   - Add guards for near-equal tensors, transposed reuse, sliced reuse, and cross-family reuse that should remain duplicated.
   - Exit condition: the approved subset preserves shared initializer reuse across roundtrip without changing numeric parity.
3. Phase 5C - Residual add and DenseNet-like concat export.
   - Completed subset: one-hop dense-family residual adds now emit explicit `Add` nodes, residual branch tensors, and `advanced_graph_residual_adds` metadata when exactly one skipped source layer feeds the target layer, and explicit `concatMappings` now emit deterministic `Concat -> Gemm` paths with `advanced_graph_concat_merges` metadata for the declared same-family subset.
   - Emit explicit branch nodes and merge nodes using `Add`, `Concat`, `Gemm`, and `Conv` only when source widths and merge ordering are provably compatible.
   - Keep merge intent metadata explicit so importer reconstruction does not depend on brittle node-name heuristics alone.
   - Preserve honest fallback when merge reconstruction cannot be proven, including cases that must remain decomposed dense graphs.
   - Exit condition: one-hop residual adds and deterministic concat merges roundtrip for the declared subset while unsupported merges stay on the fallback path.
4. Phase 5D - Residual and concat import hardening.
   - Completed subset: import now rehydrates the one-hop residual-add branch from `advanced_graph_residual_adds`, the residual branch tensor, and the existing cross-layer audit edge list, and it rehydrates explicit concat mappings from `advanced_graph_concat_merges` by splitting the widened dense tensor across the adjacent and skipped source layers while longer or ambiguous merge families stay on fallback.
   - Reconstruct supported additive and concatenative merge intent back into the native runtime only when branch ancestry, widths, and ordering survive exactly.
   - Keep fallback behavior numerically stable when explicit merge reconstruction cannot continue.
   - Add owner and public negative tests for ambiguous branch ancestry, width mismatch, merge-order drift, and unsupported merge fan-in.
   - Exit condition: merge reconstruction is faithful for the approved subset and explicitly conservative everywhere else.
5. Phase 5E - Attention export groundwork.
   - Completed subset: exporter now accepts explicit `attentionMappings`, emits the fixed-width self-attention shadow block with standard ops, and records `advanced_graph_attention_blocks` metadata for the same-family subset.
   - Define an explicit attention mapping contract before any heuristic promotion so supported attention starts as source-owned behavior rather than speculative graph mining.
   - Emit Q, K, and V projections plus head split, transpose, score `MatMul`, optional scaling, `Softmax`, value aggregation, head merge, and output projection using standard ONNX ops only.
   - Constrain the first tranche to fixed-width self-attention with fixed head counts, no masking, and no cross-attention semantics.
   - Reuse the branch, merge, and alias infrastructure from Phases 5A and 5B instead of introducing a separate attention-only graph contract.
   - Exit condition: exporter emits a deterministic, source-owned attention subgraph for the declared subset.
6. Phase 5F - Attention import, parity, and fallback hardening.
   - Completed subset: import now validates the deterministic fixed-width self-attention shadow structure recorded by `advanced_graph_attention_blocks`, attaches that audit payload as `_onnxAdvancedGraph.attentionBlocks`, and keeps runtime inference on the dense fallback scaffold rather than inventing native attention semantics.
   - Reconstruct the approved attention block only when all required tensors, head metadata, merge order, and projection widths survive exactly.
   - Otherwise preserve an honest decomposed-graph fallback or explicit rejection path rather than silently inventing native attention semantics.
   - Add public roundtrip tests for supported self-attention and negative tests for masked attention, cross-attention, malformed head partitions, and unsupported transpose or reshape order.
   - Exit condition: same-family attention roundtrip either reconstructs faithfully or falls back without semantic drift.
7. Phase 5G - Documentation and closure.
   - Completed: the ONNX chapter now documents the closed residual, concat, shared-initializer, and fixed-width attention subset together with the remaining fallback boundary.
   - Update the source-owned ONNX chapter to name the exact residual, concat, shared-initializer, and attention subset accepted by the importer and exporter.
   - Refresh this tracker's completed-baseline text, closure log, and handoff query only after the focused tests, docs pass, and full-suite validation are green.
   - Keep Phases 6-9 planned; do not let attention support become a back door for arbitrary advanced-graph compatibility claims.
   - Exit condition: Completed. Phase 5 now closes as a tested, documented advanced-graph contract rather than an aspirational feature list.

Detailed sub-steps:

1. Step 1 - Branch tensor naming and merge metadata.
2. Step 2 - Shared initializer alias detection and negative guards.
3. Step 3 - Residual-add export/import support.
4. Step 4 - Dense concat export/import support.
5. Step 5 - Explicit self-attention export mapping.
6. Step 6 - Attention import reconstruction and fallback.
7. Step 7 - Docs, parity audit, and closure update.

Validation contract for every Phase 5 slice:

- Start with the smallest failing owner-local or public test for the active slice before changing production behavior.
- Validate in this order after each slice lands:
  1. focused ONNX Jest slice for the touched boundary,
  2. coverage-guard for every changed production `src/` file,
  3. `npm run docs` if JSDoc or generated-README inputs changed,
  4. `npm run test:silent` before marking the slice closed.
- Required negative coverage for closure:
  - unsupported merge width mismatches stay on the honest fallback path,
  - concat consumer ordering drift never silently reorders inputs,
  - shared initializer reuse never aliases tensors with different semantic owners,
  - near-match attention graphs do not promote into supported blocks,
  - unsupported external branch or attention graphs are rejected or preserved as documented fallback without silent reconstruction.

Design constraints to preserve throughout Phase 5:

- Preserve the closed Phase 4 spatial contract exactly; advanced-graph work may consume Conv or Pool outputs only when the Phase 4-approved shape and fallback rules already hold.
- Prefer explicit mappings and source-owned metadata before heuristic detection for every new advanced-graph family.
- Keep exporter order deterministic: stable node order, stable initializer order, stable branch input ordering, and stable alias naming.
- Keep importer behavior honest: if a merge or attention block cannot be reconstructed exactly, preserve a numerically equivalent decomposed graph or reject it explicitly instead of widening the supported subset implicitly.
- Do not let Phase 5 residual or attention work become a shortcut to arbitrary external ONNX transformer support.

### Phase 6 (Optimization & Fidelity) [DONE]

**Status: [DONE] Closed first-wave optimization and fidelity contract for the current same-family subset**

Closed implementation summary:

- Exact unary activation emission now uses canonical ONNX operators for `Softplus`, `Softsign`, and `Selu`, plus opset-gated `Mish` and `Gelu` payloads when the selected opset makes those mappings honest.
- Export-owned postprocessing now prunes redundant Identity activation scaffolding before final validation so unsupported or opset-incompatible activations can stay on the honest baseline without leaving needless graph noise behind.
- Export-time tensor shape validation now maintains a conservative internal ledger for ranks, broadcastability, axes, and derived widths across the closed dense, recurrent, spatial, residual, concat, and fixed-width attention subsets.
- Closure validation is complete for the landed first wave: focused ONNX Jest slices, coverage-guard on every changed ONNX production file, `npm run docs` after the source-owned JSDoc update, and `npm run test:silent` before closing the tranche.

Execution goal:

1. Reduce exported graph verbosity only when a canonical ONNX form is semantically exact for an already-supported same-family subset.
2. Add exporter-owned constant folding that removes trivial scaffolding without mutating runtime-owned semantics or widening import claims.
3. Introduce an internal tensor shape ledger that validates ranks, broadcastability, and attribute-derived output shapes before model finalization.
4. Keep this phase honesty-first: optimized graphs are allowed only when roundtrip parity, deterministic node ordering, and importer or audit behavior remain explicit.

Non-goals preserved by the closed Phase 6 contract:

- No universal ONNX Runtime or protobuf-compatibility claim.
- No activation or operator support claim until the roundtrip tests, opset floor, and supported-subset docs all exist and pass.
- No speculative folding across runtime inputs, recurrent state carriers, or exporter-external constants.
- No widening of external import boundaries just because exporter output becomes more canonical.

Completed execution order:

1. Phase 6A - Fusion inventory and equivalence matrix.
   - Catalogue each current emitted scaffold eligible for optimization: dense homogeneous layers, mixed or per-neuron layers, recurrent fused-node leftovers, residual or concat auxiliaries, spatial reshape bridges, and fixed-width attention shadow helpers.
   - For each candidate, record opset minimum, canonical ONNX target op or ops, exact semantic preconditions, importer story, and fallback behavior.
   - The first fusion wave should prefer one-to-one unary activation operators with stable ONNX schemas before broader multi-op rewrites.
   - Exit condition: approved matrix of exact same-family fusion candidates and explicit non-goals.
2. Phase 6B - Operator fusion control for exact affine plus activation paths.
   - Keep `Gemm` as the canonical dense affine anchor because its shape rules, optional bias input, and broadcast semantics are explicit in ONNX.
   - Add explicit fusion controls so export can choose between the current baseline graph and approved canonical activation ops such as `Gelu` only when the internal activation semantics, attribute choices such as `approximate`, and opset floor match exactly.
   - Fold decomposed per-neuron or legacy-ordering paths back into one shared dense branch only when layer homogeneity, connectivity, and activation identity can be proven from the graph.
   - Preserve baseline emission when any activation, bias broadcast, transposition, or opset requirement is ambiguous.
   - Exit condition: exact fusion candidates emit deterministically, remain same-family roundtrip safe, and unsupported cases stay on the baseline path.
3. Phase 6C - Export-owned constant folding.
   - Restrict folding to exporter-owned literals, initializers, and stateless helper nodes whose outputs are fully determined at export time.
   - The first folding wave should target trivial identity chains, constant shape-helper paths, scalar scale nodes in approved attention-shadow subsets, and dead initializers or nodes introduced by earlier scaffolding.
   - Never fold across runtime inputs, recurrent state carriers, or values whose semantics depend on symbolic or unknown dimensions.
   - Keep fold order deterministic and observable so diff stability and regression audits remain easy.
   - Exit condition: folded graphs preserve parity and remove only provably redundant exporter scaffolding.
4. Phase 6D - Internal shape inference and validation ledger.
   - Introduce an export-time tensor ledger that records rank, dimension values or symbols, element type, producer op, and broadcast constraints for each emitted tensor name.
   - Seed the ledger from known layer widths, explicit spatial mappings, fixed-width attention metadata, and the current symbolic batch contract instead of inventing broader dynamic support.
   - Implement inference rules in ONNX style for the approved operator subset: `Gemm`, unary activations, `Add`, `Concat`, `MatMul`, `Softmax`, `Conv`, pool ops, `Flatten`, `Reshape`, `Transpose`, and the closed Phase 5 shadow or merge paths.
   - Follow ONNX's own limitation boundary: unknown symbolic dimensions may propagate, but the exporter should not pretend it can solve arbitrary arithmetic over them.
   - Exit condition: export fails early with explicit shape errors when ranks, axes, broadcast rules, or derived widths do not line up.
5. Phase 6E - Documentation, determinism, and closure.
   - Update source-owned ONNX docs to distinguish baseline support from optimized emission forms, including opset floors and exact fold or fusion preconditions.
   - Add determinism tests proving that the same network and options emit the same optimized node and initializer ordering.
   - Add public roundtrip and owner-local negative tests proving optimization never widens the supported subset silently.
   - Keep Phase 7 planned until the optimized export path is documented and green under repo-wide validation.
   - Exit condition: Phase 6 closes as a tested optimization contract rather than an informal cleanup pass.

Closed first-wave targets:

- Canonical dense `Gemm` plus direct unary activation ops where NeatapticTS semantics match an ONNX operator exactly and opset support is stable.
- Safe collapse of decomposed homogeneous paths back to shared dense emission when mixed or per-neuron scaffolding is no longer semantically needed.
- Export-owned constant folding for trivial shape, identity, and dead-scaffold nodes.
- Internal shape validation for the already-closed dense, recurrent, spatial, residual, concat, and fixed-width attention subsets.

Explicit non-goals for the first Phase 6 closure:

- Aggressive algebraic graph rewrites that change numeric behavior or floating-point order.
- Folding or fusing across unknown dynamic reshape targets, symbolic dimension arithmetic that the exporter cannot prove, or runtime-state-carrying recurrent paths.
- Treating optimized export forms as proof of broader import support for arbitrary external ONNX graphs.
- Using `Gelu` or any other higher-level op as a shortcut around missing activation-equivalence proofs.

Validation contract for every Phase 6 slice:

- Start with the smallest failing owner-local or public test for the active optimization or validation rule before production edits.
- Validate in this order after each slice lands:
  1. focused ONNX Jest slice for the touched exporter or importer boundary,
  2. coverage-guard for every changed production `src/` file,
  3. `npm run docs` when source-owned ONNX docs or generated README inputs change,
  4. `npm run test:silent` before marking the slice closed.
- Required negative coverage for closure:
  - opset-incompatible or semantically mismatched activations stay on baseline emission,
  - constant folding never crosses runtime inputs or symbolic unknowns the exporter cannot resolve,
  - shape inference rejects bad broadcast, bad rank, and bad axis cases before graph finalization,
  - optimized graphs preserve deterministic node and initializer order,
  - importer behavior stays honest when optimized forms are exporter-owned but external graphs are not supported.

Design constraints to preserve throughout Phase 6:

- Keep the JSON-first trust boundary explicit; optimization does not imply universal protobuf or ONNX Runtime compatibility.
- Keep `Gemm` attribute derivation canonical (`alpha`, `beta`, `transA`, `transB`) and never hide broadcast assumptions inside ad hoc helpers.
- Treat ONNX operator selection as opset-gated and subset-gated; if an op such as `Gelu` requires a higher opset or approximation flag, that must be explicit in options, tests, and docs.
- Keep shape inference conservative in the same spirit as ONNX: unknown symbols may propagate, but unresolved arithmetic or missing ranks must not be invented into false certainty.
- Do not let Phase 6 become a dumping ground for Phase 7 quantization, Phase 8 custom-op policy, or Phase 9 runtime compatibility work.

### Phase 7 (Quantization & Precision) [WIP]

**Status: [WIP] Phase 7A, the first 7B storage-FP16 lane, the 7C calibration contract, and the current 7D same-family dense qlinear slice with explicit bias bridges plus preserved unary activations are landed; broader quantized lowering remains open**

Current execution status:

- Completed: exporter-owned `precision` and `quantization` packets now exist at the ONNX option boundary, invalid packet combinations reject early, and unsupported packet shapes no longer pass through silently.
- Completed: model metadata now records requested and effective Phase 7 precision or quantization lanes plus explicit float32 fallback reasons, so unsupported graph families stay honest before quantized rewrites exist.
- Completed: `precision.mode = 'storage-fp16'` now packs eligible same-family dense and Conv weight or bias initializers into float16 storage, prepends deterministic `Cast -> float32` bridges for `Gemm` and `Conv`, and round-trips through import by decoding the packed payload back into native float32 runtime weights.
- Completed: static 8-bit requests now require explicit calibration layer targets, validate deterministic calibration-policy fields, emit schema-valid scale and zero-point initializers for the supported same-family dense and explicit Conv subset, and preserve explicit float32 fallback metadata outside the landed qlinear subset.
- Completed: the current 7D dense slice now lowers explicitly targeted same-family dense layers into `QuantizeLinear -> QLinearMatMul -> DequantizeLinear`, emits deterministic transposed quantized weight tensors, reattaches nonzero bias through explicit float-domain `Add` bridges, preserves exporter-owned unary activation nodes, reports `effective_quantization_mode = static-8bit`, and keeps unsupported graph families on the honest float32 path.
- Completed: focused public parity coverage now proves the explicit dense qlinear bias-bridge plus preserved unary-activation path stays within a small tolerance of the baseline dense runtime for the current one-output same-family slice.
- Completed: focused public parity coverage now also proves consecutive targeted same-family dense qlinear layers stay within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves a preserved-unary `Tanh -> Sigmoid` qlinear dense chain stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves a preserved-unary `Softplus` qlinear dense layer stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves a preserved-unary `Softsign` qlinear dense layer stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves an opset-20 preserved-unary `Gelu` qlinear dense layer stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves a preserved-unary `Selu` qlinear dense layer stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: focused public parity coverage now also proves an opset-18 preserved-unary `Mish` qlinear dense layer stays within a small tolerance of the baseline dense runtime for the current one-output calibration-backed slice.
- Completed: supported-subset hardening now also proves qlinear `Mish` requests below opset 18 keep the static-8bit dense lowering path with empty quantization fallback metadata while the unsupported activation itself stays on the implicit identity export path and emits the existing fallback warning.
- Next stop line: continue Phase 7D with dense-only parity and supported-subset hardening before the narrower Phase 7E spatial lowering lane opens.

Execution goal:

1. Add opt-in reduced-precision export lanes that shrink storage or target quantized inference without altering default float32 semantics or widening external compatibility claims.
2. Land a conservative FP16 subset for already-supported same-family dense and spatial graphs, with explicit type-consistency rules and model-level precision metadata.
3. Land post-training static 8-bit quantization only for graph regions whose scales, zero points, bias semantics, activation or weight encodings, and calibration data can be derived and validated honestly under ONNX's quantized operator rules.
4. Treat dynamic quantization as an exporter-owned guidance or rewrite surface for supported dense inference paths only, not as a blanket runtime acceleration promise.

Current non-goals while Phase 7 is planned:

- No blanket FP16 or int8 compatibility claim for arbitrary external runtimes or protobuf flows.
- No quantized import or quantized roundtrip reconstruction claim until the importer accepts a named same-family subset or explicitly dequantizes back to the native runtime with tests.
- No recurrent, residual, concat, attention, mixed-activation, or partial-connectivity quantization in the first closure.
- No float8, int4, blocked quantization, weight-only quantization, quantization-aware training, or backend-specific kernel claims in the first closure.

Recommended execution order:

1. Phase 7A - Precision taxonomy and capability matrix.
   - Catalogue each emitted ONNX family and record whether it can honestly support storage-oriented FP16, end-to-end FP16, static int8 `QLinear*` lowering, or dynamic quantization guidance.
   - Pin the first-wave operator set and opset floors from the ONNX spec: `Cast` for FP16 type conversion, `QLinearMatMul` and `QLinearConv` for static quantized compute, `QuantizeLinear` / `DequantizeLinear` for explicit QDQ boundaries, and `DynamicQuantizeLinear` only for its documented float-to-uint8 scalar-parameter path.
   - Split the planned option surface before implementation starts: keep reduced-precision storage under a dedicated precision packet and keep quantized compute under a separate quantization packet so the API does not overload one flag with two different semantic families.
   - Name the granularity boundaries up front: per-tensor activations first, scalar output scales first, and per-output-channel weight quantization only where the target operator schema and current tensor layout make it explicit.
   - Exit condition: approved matrix of supported first-wave precision targets, required metadata, and explicit exclusions.
2. Phase 7B - FP16 export contract.
   - Add an explicit precision mode that distinguishes storage-oriented FP16 snapshots from any later end-to-end FP16 compute lane.
   - For storage-oriented FP16, cast eligible initializers to `float16` and record model-level precision metadata without silently mixing `float16` weights into `float32` operators.
   - If an end-to-end `float16` graph is later enabled, keep type consistency explicit with boundary `Cast` nodes or full operator-family conversion; never leave `Gemm` or `Conv` inputs and initializers in mismatched numeric types implicitly.
   - Preserve `float32` as the canonical reference path and keep fallback to `float32` emission when an operator family, activation payload, or inferred tensor type cannot stay type-consistent.
   - Exit condition: FP16 emission is opt-in, deterministic, metadata-backed, and numerically audited against `float32` within a declared tolerance.
3. Phase 7C - Calibration and quantization-parameter contract. [DONE]
   - Define the calibration inputs required before static quantization can claim support: representative activation ranges, weight-range policy, zero-inclusion rules, symmetric vs asymmetric policy, and deterministic rounding expectations.
   - Start with exporter-owned scale and zero-point initializers and keep the first closure honest about how those parameters were produced.
   - Validate shape compatibility for every scale and zero-point tensor according to the ONNX operator schema: scalar per-tensor first, 1-D per-axis only where the operator explicitly allows it and the current tensor orientation is proven.
   - Reject or keep `float32` emission when calibration data is missing, unstable, or incompatible with the target operator's scale-shape contract.
   - Exit condition: calibration metadata and emitted quantization parameters are deterministic, schema-valid, and reproducible from the same network plus calibration packet.
4. Phase 7D - Static quantized dense lane.
   - Landed current slice: explicitly targeted same-family dense layers now lower into `QuantizeLinear -> QLinearMatMul -> DequantizeLinear`, reuse the 7C calibration packet, emit deterministic transposed quantized weight tensors, reattach nonzero bias through explicit float-domain `Add` bridges, preserve exporter-owned unary activation nodes, and report `effective_quantization_mode = static-8bit` when that narrow subset is active.
   - Lower approved dense affine regions from float `Gemm` into a conservative quantized affine path built around `QLinearMatMul` plus explicit bias and activation handling.
   - Keep the first dense subset narrow: same-family fully connected layers with homogeneous activations and no unsupported broadcast, merge, recurrent, or mixed-family behavior.
   - Treat bias honestly: `QLinearMatMul` has no fused bias input, so any affine-bias path must use an explicit, schema-valid bridge rather than pretending `Gemm` quantizes one-to-one. If a bias path cannot be represented cleanly and tested, keep that dense case on `float32` emission.
   - Keep the activation subset gated after quantized affine output; the first closure may need a QDQ boundary back into the existing unary activation contract before re-quantizing later layers.
   - Exit condition: supported dense layers emit a deterministic quantized path, unsupported bias or activation cases stay on `float32`, and parity tests quantify the accepted error budget.
5. Phase 7E - Static quantized spatial lane.
   - Lower approved explicit or promoted Conv regions into `QLinearConv` with emitted input, weight, and output scale or zero-point initializers.
   - Respect ONNX's explicit bias rule: optional bias must be `int32`, one value per output channel, with scale = `input_scale * weight_scale` and zero point = `0`.
   - Keep the first Conv subset limited to the already-closed Phase 4 spatial contract, starting with same-family explicit Conv mappings and only the proven auto-promoted Conv families whose kernel orientation and channel layout are already stable.
   - Keep pooling, flatten, reshape bridges, and downstream dense consumers quantization-aware only where the existing virtual-shape contract and precision transitions are explicit; otherwise dequantize or fall back before those boundaries.
   - Exit condition: supported Conv paths emit schema-valid `QLinearConv` graphs with audited bias handling and explicit fallback at unsupported spatial boundaries.
6. Phase 7F - Dynamic quantization guidance.
   - Separate true ONNX `DynamicQuantizeLinear` support from generic runtime-quantization marketing. The ONNX operator is a narrow float-input to uint8-output path with scalar scale and zero-point outputs, and the first closure should respect that documented contract.
   - Start with exporter-owned guidance or explicit `DynamicQuantizeLinear` insertion ahead of supported dense inference paths only.
   - Keep dynamic quantization out of Conv, recurrent, attention, residual, and concat paths in the first closure unless a narrower operator family is explicitly proven later.
   - Preserve importer honesty: dynamic-quantized graphs are either accepted as a named same-family subset or treated as external or runtime-only artifacts rather than silently reconstructed into an approximate native network.
   - Exit condition: the plan names exactly which dense paths can emit dynamic quantization guidance and what importer behavior remains unsupported.
7. Phase 7G - Documentation, determinism, and closure.
   - Update source-owned ONNX docs to separate `float32` baseline support, FP16 support, static int8 support, and dynamic-guidance support.
   - Extend supported-subset tables with opset floors, granularity limits, calibration assumptions, and explicit unsupported families.
   - Add determinism tests for emitted precision metadata and quantization-parameter ordering, plus public parity or accuracy envelopes for each supported first-wave lane.
   - Keep Phase 8 planned until precision modes, calibration inputs, importer boundaries, and runtime honesty are all documented and green.
   - Exit condition: Phase 7 closes as a tested precision contract, not a storage-optimization slogan.

Concrete activation-ready implementation slices for Phase 7A and Phase 7B:

1. Slice 7A.1 - Export option packet split and resolution scaffolding.
   - Target files: `src/architecture/network/onnx/export/network.onnx.export.types.ts`, `src/architecture/network/onnx/export/network.onnx.export-build.utils.ts`, and `src/architecture/network/onnx/export/network.onnx.export-flow.utils.ts`.
   - Red tests first: owner-local option-resolution tests proving default float32 export remains unchanged, unsupported precision or quantization combinations reject clearly, and dynamic quantization cannot claim a generic int8 path.
   - Required API outcome: introduce a dedicated `precision` packet and a dedicated `quantization` packet in `OnnxExportOptions`, plus resolved-option counterparts in `OnnxBuildResolvedOptions`, without changing any current caller behavior when both packets are absent.
   - Exit condition: exporter-owned option normalization is deterministic, backward compatible by default, and explicit about unsupported combinations.
2. Slice 7A.2 - Capability matrix metadata contract.
   - Target files: `src/architecture/network/onnx/export/network.onnx.export-build.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-setup.utils.ts`, and existing metadata append helpers under `src/architecture/network/onnx/export/`.
   - Red tests first: metadata-focused tests proving the model records the chosen precision lane, quantization lane, calibration source marker, and fallback reason only when those packets are present.
   - First metadata keys should stay narrow and source-owned, for example: effective precision mode, quantization mode, activation granularity, weight granularity, calibration source, and explicit fallback reasons.
   - Exit condition: metadata emission is deterministic and does not imply support for operators or import paths that have not landed yet.
3. Slice 7A.3 - Unsupported-request guard rail pass.
   - Target files: `src/architecture/network/onnx/export/network.onnx.export-build.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export.test.ts`, and the nearest owner-local tests around resolved options.
   - Red tests first: storage-FP16 plus quantization simultaneously requested without an explicit composition contract rejects; dynamic-quantization requests targeting Conv reject; per-output-channel requests outside supported Conv weights reject; unsupported merge or recurrent families remain on float32.
   - Exit condition: Phase 7 can start from a narrow, honest request filter before any casting or quantized node rewrite lands.
4. Slice 7B.1 - Storage-FP16 failing test tranche.
   - Target files: `src/architecture/network/onnx/export/network.onnx.export-build.utils.test.ts`, `src/architecture/network/onnx/export/network.onnx.export.test.ts`, and, if needed, one owner-local utility test around initializer rewriting.
   - Red tests first: eligible weight and bias initializers switch to `float16`, model metadata records storage FP16 explicitly, unsupported operator families stay on float32, and default exports remain byte-for-byte stable when `precision` is absent.
   - Parity expectation: this slice audits storage shape and initializer typing first; numerical tolerance tests come after the implementation lands.
   - Exit condition: a failing red slice exists for storage-FP16 only, not for end-to-end fp16 compute.
5. Slice 7B.2 - Storage-FP16 implementation.
   - Target files: `src/architecture/network/onnx/export/network.onnx.export-build.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-orchestrators.utils.ts`, and any chapter-local initializer helpers that own tensor materialization.
   - Implementation rule: cast only exporter-owned eligible initializers and keep operator-boundary typing honest. If an initializer family cannot switch to `float16` without leaving the graph in a mixed-type state the exporter does not yet normalize, preserve float32 for that family and record the fallback.
   - Validation after the first edit: rerun the narrow storage-FP16 slice before touching any additional files.
   - Exit condition: storage-FP16 is deterministic, metadata-backed, and limited to the families that can preserve type consistency under the current float32 operator boundary.
6. Slice 7B.3 - Storage-FP16 docs and closure gate.
   - Target files: `src/architecture/network/onnx/network.onnx.ts`, generated `src/architecture/network/onnx/README.md` via `npm run docs`, and `plans/ONNX_EXPORT_PLAN.md` once the slice is actually landed.
   - Required docs update: separate storage-FP16 from any later end-to-end fp16 compute claim, list the exact eligible families, and document float32 fallback when mixed-type normalization is unavailable.
   - Exit condition: the first Phase 7 implementation slice closes with focused Jest, coverage guard, docs regeneration, and repo-wide silent tests, without activating static quantization early.

Planned first-wave targets:

- Storage-oriented FP16 initializer export with model-level precision metadata and explicit `float32` fallback.
- Static 8-bit lowering for same-family dense affine regions that can represent quantized bias and activation transitions honestly.
- Static 8-bit lowering for the closed explicit or proven Conv subset via `QLinearConv` with per-tensor activations and scalar output quantization, plus per-output-channel weights only when the schema and layout audit already hold.
- Exporter-owned dynamic quantization guidance for supported dense inference inputs only, with no broadened import claims.

Explicit non-goals for the first Phase 7 closure:

- Universal external-runtime compatibility claims for FP16 or int8 graphs.
- Quantized support for recurrent, attention, residual, concat, mixed-activation, or arbitrary branched graphs.
- Float8, int4, blocked quantization, weight-only quantization, or quantization-aware training.
- Silent accuracy drift, silent bias folding, or automatic calibration from arbitrary user datasets without a declared protocol.
- Using dynamic-quantization labeling as a proxy for actual `DynamicQuantizeLinear` schema support.

Validation contract for every Phase 7 slice:

- Start with the smallest failing owner-local or public test for the active precision lane before changing production behavior.
- Validate in this order after each slice lands:
  1. focused ONNX Jest slice for the touched exporter, importer, or calibration boundary,
  2. coverage-guard for every changed production `src/` file,
  3. `npm run docs` when source-owned ONNX docs or generated README inputs change,
  4. `npm run test:silent` before marking the slice closed.
- Required negative coverage for closure:
  - unsupported precisions or opset floors stay on `float32` emission,
  - invalid scale or zero-point shapes, axes, or granularity requests reject before model finalization,
  - dense bias paths that cannot be represented honestly do not silently quantize,
  - `QLinearConv` bias scale and zero-point rules are enforced exactly,
  - dynamic quantization never widens into unsupported Conv, recurrent, residual, concat, or attention paths,
  - importer behavior stays explicit when precision-optimized graphs are exporter-owned but external quantized graphs are not supported.

Design constraints to preserve throughout Phase 7:

- Keep the JSON-first trust boundary explicit; reduced precision does not imply universal protobuf or ONNX Runtime compatibility.
- Keep `float32` export as the canonical semantic reference and default fallback.
- Keep precision modes opt-in and metadata-backed; never infer quantization from a bare opset or from existing integer initializers.
- Keep the Phase 7 API additive at the exporter-owned option boundary: new precision or quantization packets must layer onto `OnnxExportOptions` instead of overloading existing spatial, recurrent, or advanced-graph flags.
- Respect the ONNX operator schemas exactly: `Cast` type consistency, `QuantizeLinear` / `DequantizeLinear` scale-shape matching, `QLinearMatMul` scale and zero-point tensor-shape rules, `QLinearConv` bias quantization rules, and `DynamicQuantizeLinear`'s float-to-uint8 scalar contract.
- Keep calibration deterministic, reproducible, and separable from the network topology; a changed calibration packet is a changed export artifact.
- Do not let Phase 7 become a dumping ground for Phase 8 custom-op policy or Phase 9 runtime or tooling claims.

### Phase 8 (Binary Serialization & Checker Compliance)

- Emit real protobuf `ModelProto` output for the declared supported subset while keeping the current JSON-first artifact as an internal or debug surface rather than the primary compliance path.
- Replace repo-owned tensor shortcuts with ONNX-compliant storage handling for the supported subset, including canonical tensor fields, explicit domain or opset declarations, and honest exclusion of `external_data` or packed low-bit tensors until they are implemented.
- Raise the plan to a named compatibility baseline: target the current ONNX 1.22.0 `ai.onnx` opset 27 directly, or ship an explicit lower-opset compatibility contract plus version-conversion policy when a lower floor remains intentional.
- Add checker-backed validation in CI for the compliant export subset so malformed attributes, tensor fields, domains, opsets, and graph topology fail before release.
- Keep custom domains and `FunctionProto` behind a standard-op-first audit; extensibility is allowed, but it cannot be used as a shortcut around missing standard ONNX compliance.

### Phase 9 (Runtime Compatibility & External Import Closure)

- Add a real ONNX runtime parity harness for the declared compliant subset, with golden models plus randomized parity checks against the native `Network.activate()` path.
- Promote binary `.onnx` export to the primary compliant artifact for the supported subset, while retaining `--json` as an explicit debug or internal surface instead of the main compatibility story.
- Close at least one named external import subset that reconstructs core execution semantics without repo-only metadata carrying the meaning, and add negative coverage proving malformed or out-of-subset external graphs reject cleanly.
- Only after the standard subset is runtime-validated, add optional custom-domain and `FunctionProto` extensibility paths with explicit non-portable wording when a runtime cannot execute them generically.
- Add the final closure gate here: do not mark this plan `[DONE]` until Phase 8 is green, runtime parity is green, and the named external import subset is green with public docs.

## 6. Testing Strategy

Layers:

1. Unit tests: layer inference, homogeneity validation, weight/bias serialization integrity, activation mapping.
2. Property-based tests: random MLPs (sizes 1–6 hidden layers) round-trip equality (MSE < 1e-9 for deterministic forward pass).
3. Negative tests: heterogeneous layer activations, missing connections, unsupported recurrent edges → expect descriptive errors.
4. Future: golden ONNX Runtime inference parity set (serialize known networks; validate outputs within tolerance on random batches).
5. Fuzzing (Phase 3+): generate random topologies, attempt export; assert either success or categorized error (no silent corruption).

## 7. Data Structures & API Evolution

Planned interface additions:

```ts
interface OnnxExportOptions {
  opset?: number; // default 18
  producerName?: string; // default 'neataptic-ts'
  includeMetadata?: boolean; // wrap in ModelProto-like object
  batchDimension?: boolean; // add symbolic batch dim
  allowMixedActivations?: boolean; // relax homogeneity constraint
  legacyNodeOrdering?: boolean; // keep Activation-before-Gemm ordering
   concatMappings?: Array<{ sourceLayerIndex: number; targetLayerIndex: number; inputOrder?: 'previous_then_source' }>;
  sparseFormat?: 'none' | 'csr';
   precision?: {
      mode?: 'float32' | 'storage-fp16';
      metadata?: boolean;
   };
   quantization?:
      | {
            mode: 'static-8bit';
            targets: Array<'dense' | 'conv'>;
            calibration: {
               source: 'external';
               packetId?: string;
               sampleCount?: number;
            };
            activationEncoding?: 'uint8' | 'int8';
            weightEncoding?: 'uint8' | 'int8';
            activationGranularity?: 'per-tensor';
            weightGranularity?: 'per-tensor' | 'per-output-channel';
            representation?: 'qlinear' | 'qdq';
         }
      | {
            mode: 'dynamic-uint8';
            target?: 'dense';
            representation?: 'DynamicQuantizeLinear' | 'metadata-only';
         };
}
```

Backward compatibility: default options reproduce current behavior except corrected node order (opt-in legacy), and new precision or quantization packets remain fully absent by default.

Planned API refinement notes:

- Do not overload one `quantize` flag to mean both reduced-precision storage and quantized compute. FP16 storage and QLinear-style quantization are different contracts and need separate packets.
- Keep `precision.mode = 'storage-fp16'` explicit so the first closure does not accidentally imply end-to-end fp16 compute.
- Keep `quantization.mode = 'dynamic-uint8'` explicit because ONNX `DynamicQuantizeLinear` is a uint8-only contract in the first supported schema lane.
- Keep `targets` explicit for static quantization so unsupported families can reject before graph rewriting begins.
- Keep calibration source explicit and external for the first closure; automatic internal calibration can be a later extension only after the deterministic packet contract exists.

## 8. Evolutionary Metadata Preservation

- Store innovation IDs & genealogy inside model metadata (`model.graph.doc_string` JSON blob) for traceability.
- Provide a `stripEvolutionMetadata(model)` helper for clean distribution.

## 9. Open Questions / Design TBD

- How to map arbitrary evolved recurrent connections to structured RNN cells without losing semantics? Candidate: limited unrolling + annotation.
- Whether to introduce internal canonicalization pass (topological sort + layering) that becomes the source of truth for both evaluation and export, reducing divergence.
- Policy for unsupported activations: warning vs error vs automatic approximation (e.g., approximate Gaussian with exp-based composite subgraph).

## 10. Risks & Mitigations

- Risk: Export divergence from evaluator leads to incorrect weights (Mitigation: round-trip + ONNX Runtime tests in CI).
- Risk: Explosion of custom ops reduces interoperability (Mitigation: prefer composition of standard ops; gate custom domain usage behind explicit flag).
- Risk: Performance regression from per-neuron decomposition (Mitigation: fusion pass before final emission).

## 11. Cross-phase backlog beyond the current frontier

1. Property-based randomized topology tests (1–6 hidden layers; varying sparsity & mixed activations) ensure import/export fidelity.
2. Fusion optimization pass for decomposed layers (homogeneity collapse).
3. Sparse representation design (`sparseFormat` CSR draft spec).
4. ONNX Runtime smoke test harness for unified and decomposed models.
5. Design notes for multi-input or output and residual connection representation (branching graph semantics).

## 12. Contribution Guidelines (ONNX Area)

- Keep exporter pure (no side effects except optional console warnings); return new object.
- Add exhaustive test for any new operator mapping (export → import or export → ORT inference).
- Include spec citation (URL + opset version) in code comments when adding new ops.
- Avoid premature optimization; add baseline first then fusion/optimization passes under feature flags.

## 13. Appendix: Future Ideas (Deferred)

- Automatic pruning & weight compression prior to export (magnitude, structured, or evolutionary salience based).
- Multi-objective export scoring (file size, latency) to guide evolution toward deployable architectures.
- Export of training graph (loss node, optimizer state) using ONNX Training extensions (far future).
- Hybrid symbolic + numeric differentiation metadata for advanced downstream tooling.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Work on plans/ONNX_EXPORT_PLAN.md as the active ONNX tracker.
Treat Phases 0-6 as closed for the current conservative subset and preserve the closed recurrent, spatial, advanced-graph, and optimization contracts exactly. Do not widen those boundaries implicitly.
Phase 7 is active. Treat 7A, the storage-FP16 7B lane, the 7C calibration contract, and the current 7D same-family dense qlinear slice as landed.
The current 7D baseline is exporter-owned only: explicitly targeted same-family dense layers can lower into `QuantizeLinear -> QLinearMatMul -> DequantizeLinear`, emit deterministic transposed quantized weight tensors, reattach nonzero bias through explicit float-domain `Add` bridges, preserve exporter-owned unary activation nodes, and report `effective_quantization_mode = static-8bit`. Quantized import and 7E spatial lowering are still open.
Recent validation baseline is green: focused ONNX export/build/conv/network Jest slices are green, coverage guard is 100% statements/branches/functions/lines for `src/architecture/network/onnx/export/network.onnx.export-build.utils.ts`, `src/architecture/network/onnx/network.onnx.ts`, and `src/architecture/network/onnx/export/network.onnx.export.types.ts`, `npm run docs` is green, and `npm run test:silent` is green.
If implementation continues, stay in Phase 7D. The next narrow task is dense-only parity and supported-subset hardening for the current qlinear dense slice without widening into recurrent, residual, concat, attention, mixed-activation, arbitrary external quantized graphs, or the still-open spatial lane.
Keep `DynamicQuantizeLinear` language honest as the float-to-uint8 scalar-parameter lane only, keep quantized import outside the supported subset unless it is explicitly implemented and tested, and do not treat optimized emission as proof of broader ONNX Runtime compatibility.
After the current Phase 7 work, later compliance closure is still required: Phase 8 owns binary `ModelProto` serialization plus checker validation, and Phase 9 owns runtime parity plus at least one named external import subset.
Do not mark this plan `[DONE]` merely because the current exporter-owned Phase 7 subset is strong; later closure still requires binary serialization, checker validation, runtime parity, and a named external import subset.
Validate future changes with focused ONNX Jest slices first, then coverage guard for each changed `src/` file, then `npm run docs` if JSDoc changes, then `npm run test:silent`.
The worktree may contain unrelated user changes outside this ONNX lane; do not revert or normalize unrelated files.
```

---

Historical reference: See `network.onnx.ts` for the current implementation baseline. Update this document whenever phases or APIs land.

Repository-wide testing and style conventions are defined in the main project guidance and should be applied to ONNX-related changes.
