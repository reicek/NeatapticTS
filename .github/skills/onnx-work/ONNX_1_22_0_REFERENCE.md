# ONNX 1.22.0 Reference Pack

Use this file as the first stop before fetching ONNX docs again. It is a repo-local
summary of the ONNX 1.22.0 pages most relevant to NeatapticTS export, import,
quantization, and compatibility audits.

## Source pages

- Concepts: https://onnx.ai/onnx/intro/concepts.html
- Converters: https://onnx.ai/onnx/intro/converters.html
- Float8: https://onnx.ai/onnx/technical/float8.html
- Int4: https://onnx.ai/onnx/technical/int4.html
- Float4: https://onnx.ai/onnx/technical/float4.html
- Int2: https://onnx.ai/onnx/technical/int2.html
- Operator conventions: https://onnx.ai/onnx/repo-docs/OpConventions.html

## Compliance facts that matter most

### 1. Real ONNX serialization is protobuf-based

- ONNX graphs are normally serialized as protobuf `ModelProto` payloads.
- A JSON object that resembles `ModelProto` is useful for internal tooling, but
  it is not the same thing as full ONNX serialization compliance.
- A stronger compatibility claim needs more than shape similarity. It usually
  needs protobuf emission, checker compatibility, and runtime validation.

### 2. ONNX is strongly typed

- ONNX does not rely on implicit casts.
- Mixed numeric types require explicit `Cast` nodes.
- Any reduced-precision lane must keep operator input or output types explicit.
  Silent type mixing is not spec-honest.

### 3. Opsets are part of the contract

- ONNX 1.22.0 reports `ai.onnx` opset 27.
- A graph can declare different opsets per domain.
- Changing the declared opset without changing emitted operators or attributes
  can make the graph invalid.
- Exporters should expose an explicit target opset and keep tests aligned to the
  exact opset floor for each emitted operator.

### 4. Domains matter

- Standard ONNX operators live in `ai.onnx`.
- Classical ML operators live in `ai.onnx.ml`.
- Custom domains are allowed, but they are a portability boundary, not a free
  compatibility upgrade.
- If a feature needs a custom domain or `FunctionProto`, the supported-subset
  docs should say so explicitly.

### 5. Converters are parity obligations, not just graph writers

- A converter is expected to rewrite a source model into ONNX operators while
  preserving predictions exactly or within a known tolerance.
- Converter maintenance is ongoing because both the source framework and ONNX
  opsets keep changing.
- A serious ONNX claim needs target-opset control, source-to-ONNX parity tests,
  and explicit handling for unsupported source features.

## Operator and naming conventions

These ONNX repo conventions are useful when auditing generated graph names or
when deciding whether a repo-specific naming scheme is merely cosmetic or
actually conflicts with operator expectations.

- Attribute names: lower case with underscores when helpful.
- Single-letter inputs or outputs: upper case, for example `X`, `W`, `B`.
- Full-word inputs or outputs: lower case with underscores.
- Bias tensors: `B`.
- Weight tensors: `W`.
- Use `axis` for one axis and `axes` for multiple axes.

These are conventions, not a full validity checker. They help with readability
and consistency, especially when adding new operator mappings.

## Model concepts to preserve honestly

A valid ONNX graph is more than nodes and tensors.

- Inputs, outputs, nodes, initializers, and attributes all have distinct roles.
- Initializers are constants stored inside the graph.
- Metadata such as `producer_name`, `producer_version`, `doc_string`, and
  `metadata_props` is supported by the format, but metadata must not carry core
  semantics that a normal ONNX runtime would need in order to execute the graph.
- Shape inference is optional for execution, but it is important for memory
  planning, validation, and interoperability.
- Subgraphs such as `If`, `Loop`, and `Scan` are part of standard ONNX. If a
  project replaces temporal behavior with custom metadata instead of those
  operators, that is a supported-subset choice, not full ONNX temporal parity.

## Reduced-precision data types in ONNX 1.22.0

### Float8 family

Relevant points from the ONNX technical page:

- ONNX 1.22.0 documents float8 support in the tensor type system.
- Core float8 families include:
  - `FLOAT8E4M3FN`
  - `FLOAT8E4M3FNUZ`
  - `FLOAT8E5M2`
  - `FLOAT8E5M2FNUZ`
  - `FLOAT8E8M0` for microscaling-style scale factors
- Float8 casting rules are format-specific.
- `Cast` behavior may depend on round mode and saturation semantics.
- Any float8 export claim needs explicit operator support, cast semantics,
  and tests. Merely storing bytes is not enough.

### Int4 and Uint4

- ONNX documents `INT4` and `UINT4` support.
- Two 4-bit values are packed into one byte.
- Cast to 4-bit uses nearest-even integer rounding then truncation.
- A real int4 claim needs packed storage semantics and operator support that is
  honest about whether the graph is weight-only, activation-aware, or general
  low-bit execution.

### Float4

- ONNX documents `FLOAT4E2M1`.
- Two float4 values are packed into one byte.
- Downcast behavior is saturating and format-specific.
- A float4 claim requires more than a metadata label. It needs correct packing,
  unpacking, cast behavior, and operator coverage.

### Int2 and Uint2

- ONNX documents `INT2` and `UINT2`.
- Four 2-bit values are packed into one byte.
- Cast to 2-bit uses nearest-even integer rounding then truncation.
- Any int2 claim should stay extremely narrow unless there is explicit packing,
  storage, and execution support.

## What a stronger ONNX compatibility claim would require

Use these tiers instead of saying “fully ONNX compliant” unless every row is
actually true.

### Tier 1. Internal ONNX-like roundtrip

- JSON-first model format is accepted.
- Export and import are same-family and same-version scoped.
- Internal parity tests exist.
- No universal runtime or protobuf claim.

### Tier 2. Spec-shaped JSON or text-level parity

- Graph structure matches ONNX concepts more closely.
- Operator attributes and tensor typing are aligned to the spec.
- Still not enough for broad runtime compatibility by itself.

### Tier 3. Binary ONNX serialization compatibility

- Protobuf `ModelProto` emission exists.
- Tensor fields match ONNX tensor storage rules.
- Domain and opset metadata are correct.
- Checker-style validation can run.

### Tier 4. Runtime compatibility

- Exported models are accepted by one or more real ONNX runtimes.
- Parity tests compare runtime outputs to native outputs within tolerance.
- Unsupported operators or data types reject cleanly instead of silently
  degrading in ways a runtime would not reproduce.

### Tier 5. Broad external-model import compatibility

- Import accepts clearly named external subsets.
- Unsupported graphs reject clearly.
- Import does not rely on repo-only metadata for core graph semantics unless the
  compatibility claim is explicitly same-family only.

## Audit checklist for NeatapticTS ONNX work

Use this checklist before saying a new feature improves compliance.

### Export audit

- Does export produce a real protobuf `ModelProto`, or only a JSON-like shape?
- Are emitted ops standard ONNX ops in the declared domain?
- Are opset floors explicit and tested?
- Are tensor data types legal for the chosen operator set?
- Are reduced-precision casts explicit where ONNX requires them?
- Does the graph avoid relying on repo-only metadata for execution semantics?
- Is there parity coverage against native `Network.activate()`?

### Import audit

- Is the accepted external subset named explicitly?
- Does import rely on repo-specific naming conventions or metadata payloads?
- Does unsupported input reject clearly instead of silently degrading?
- Are quantized or reduced-precision graphs imported honestly, or explicitly
  excluded from the contract?

### Quantization audit

- Is quantization only metadata, or real operator lowering?
- If real lowering exists, which operators are supported exactly?
- Are scale and zero-point shapes validated?
- Are bias rules explicit?
- Are per-tensor versus per-axis boundaries documented?
- Is dynamic quantization actually implemented, or only planned?

### Low-bit audit

- Are float8, int4, float4, or int2 tensor types represented with correct
  storage and packing semantics?
- Are cast and saturation rules implemented?
- Are the consuming operators proven for those types?
- If not, do not claim support.

### Runtime-compatibility audit

- Does the repo run ONNX checker or runtime smoke tests?
- Are there golden models validated outside the repo’s own importer?
- If not, stay honest and call the format ONNX-like or same-family only.

## Known implications for this repo

This section is intentionally general enough to survive code movement, but it is
meant to steer future audits in NeatapticTS.

- A JSON-first payload is not a full ONNX compliance claim.
- Metadata-driven reconstruction is valuable for same-family roundtrip, but it is
  a weaker external compatibility boundary than pure graph reconstruction.
- Storage-fp16 or qlinear slices can still be valid, useful work even when the
  repo is not broadly runtime-compatible.
- Low-bit tensor types from ONNX 1.22.0 should remain explicitly out of scope
  unless the repo implements their storage rules, operator coverage, and tests.

## When to go back to the web

Use the linked ONNX docs again only when one of these is true:

- an operator-specific attribute or type constraint is missing here,
- a new opset changes the relevant operator schema,
- a pass needs exact low-bit cast math beyond this summary,
- or a runtime-compatibility claim depends on checker or backend behavior not
  covered in this local pack.
