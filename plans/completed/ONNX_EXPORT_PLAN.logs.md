# ONNX Export / Import Log

**Status:** [DONE]

## Audit scope

- Objective: close the ONNX export or import lane through the declared Phase 9 compliance target for the lower-opset same-family subset and archive the resulting reopen baseline.
- Coverage included same-family JSON-first export or import, recurrent hardening, the conservative spatial subset, same-family advanced-graph support, optimization and fidelity controls, exporter-owned precision lanes, binary `ModelProto` emission, runtime parity, and the first named external binary import subset.

## Durable milestones

### [DONE] Baseline export or import foundation

- Closed the dense same-family export or import surface, canonical `Gemm -> Activation` ordering, partial-connectivity handling, and mixed-activation decomposition or reconstruction.
- Kept the JSON-first boundary explicit so early ONNX support stayed honest about same-family scope and fallback behavior.

### [DONE] Recurrent, spatial, and advanced-graph hardening

- Closed the recurrent hardening boundary for the current single-step self-recurrence plus heuristic same-family LSTM or GRU subset, including parity-backed fallback behavior.
- Closed the conservative spatial subset with explicit Conv or Pool metadata, guarded auto-promotion, flatten-after-pool audits, and documented fallback boundaries.
- Closed the same-family advanced-graph subset for residual adds, concat mappings, exact initializer alias reuse, and fixed-width attention shadow export or import.

### [DONE] Optimization, precision, binary, and external-runtime closure

- Landed canonical unary activation emission, exporter-owned cleanup, conservative tensor-shape validation, storage-fp16, static-8bit dense and explicit-Conv lowering, and dense-only dynamic guidance with deterministic metadata and focused parity evidence.
- Closed deterministic binary `ModelProto` emission, Node-owned decode or verify plus runtime-load validation, binary-first runtime parity for the approved five-lane subset, and the first named external binary import subset through `importFromONNXBinary()` with explicit rejection taxonomy.

## Controls and evidence

- Focused ONNX Jest slices, owner-local acceptance or rejection coverage, and 100% coverage-guard validation on every touched ONNX `src/` file.
- `npm run docs` after source-owned ONNX JSDoc changes.
- Green repo-wide `npm run test:silent` validation after the Phase 9 closure tranche.

## Reopen triggers

- Any regression in the archived same-family, binary, runtime-parity, or first external-import boundary.
- A future narrower ONNX amendment for custom domains, broader external import, wider runtime portability, or a NEATchat-grade external seed path.