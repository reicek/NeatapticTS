# ONNX Export / Import Plan for NeatapticTS

**Status:** [DONE]

## Scope

- Close the ONNX lane through the declared Phase 9 compliance target for the lower-opset same-family subset.
- Preserve the archived support boundary for same-family export or import, the closed recurrent, spatial, advanced-graph, optimization, precision, binary, runtime-parity, and first external-import stop lines, and the honest exclusions that still remain outside that boundary.
- Keep future ONNX widening reopen-only so later work starts from a narrower active amendment instead of silently extending the archived contract.

## Final state

- The ONNX lane is closed through Phases 0-9 for the current declared subset.
- The archived baseline covers JSON-first same-family export and import, the closed recurrent hardening boundary, the conservative spatial contract, the same-family advanced-graph contract, the first-wave optimization and fidelity contract, the exporter-owned precision contract, deterministic binary `ModelProto` emission, runtime-parity evidence for the approved five-lane subset, and the first named external binary import subset through `importFromONNXBinary()`.
- The first external binary import claim remains narrow by design: binary-first, standard-domain, float32-only, single-input or single-output dense `Gemm -> unary activation` chains with explicit rejection outside that documented boundary.
- Broader arbitrary external ONNX import, custom-domain policy, wider runtime-portability claims, broader reduced-precision import, and the NEATchat external-seed decision remain outside this archived contract.

## Audit summary

- Phases 0-2 closed the baseline dense export or import surface, corrected canonical `Gemm -> Activation` ordering, and widened the same-family feed-forward boundary through partial connectivity and mixed-activation support.
- Phases 3-5 closed the recurrent hardening, conservative spatial, and same-family advanced-graph stop lines with explicit parity or fallback coverage and source-owned documentation.
- Phases 6-7 closed canonical unary emission, exporter-owned cleanup, conservative shape validation, storage-fp16, static-8bit dense and explicit-Conv lowering, and dense-only dynamic guidance with deterministic metadata and parity evidence.
- Phases 8-9 closed deterministic binary `ModelProto` emission, checker or runtime-load validation, runtime parity for the approved five-lane subset, the first named external binary import subset, focused 100% coverage guard on the touched ONNX `src/` files, regenerated docs, and a green repo-wide `npm run test:silent` pass.

## Reopen conditions

- A regression breaks the archived same-family, binary, runtime-parity, or first external-import contract.
- A future ONNX pass needs a narrower new claim such as a wider external import subset, custom-domain policy, broader runtime portability, or a NEATchat-grade external seed path.
- Later work needs to revise the archived supported-subset wording, rejection taxonomy, or validation expectations instead of consuming the archived boundary as-is.

## Audit log

- See [ONNX_EXPORT_PLAN.logs.md](ONNX_EXPORT_PLAN.logs.md).