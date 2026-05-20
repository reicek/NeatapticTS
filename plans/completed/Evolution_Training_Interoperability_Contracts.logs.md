# Evolution-Training Interoperability Contracts Log

**Status:** [DONE]

## Audit scope

- Objective: close the hybrid-interoperability lane after delivering deterministic parameter vectors, isolated fine-tuning, explicit hybrid evaluation policy, root-facade exports, and source-first docs for downstream consumers such as NEATchat.
- Coverage included the serialize-owned layout and roundtrip contract, the training-owned isolation helper, the hybrid-owned policy helper, the generated docs closeout, and focused coverage guards for the touched runtime surfaces.

## Durable milestones

### [DONE] Deterministic parameter layout and roundtrip contract

- Froze `ParameterLayoutV1` ordering around stable node ids plus innovation-first edge identity, with versioned export and import plus compatibility rejection for mismatched payloads.
- Landed focused owner-local tests and full runtime coverage for the serialize-owned implementation surface that ships the parameter-vector contract.

### [DONE] Training isolation helper

- Added `fineTuneVector(...)`, `FineTuneOptions`, and `FineTuneResult` at the training boundary.
- Verified unchanged input network or vector snapshots, same-seed same-dataset-order stability on the same runtime, source-first docs refresh, and `100/100/100/100` focused coverage for `src/architecture/network/training/network.training.isolate.utils.ts`.

### [DONE] Hybrid policy surface and public docs closure

- Added `evaluateCandidate(...)` plus explicit policy and result types under `src/neat/hybrid/`, with `never` and `always` behavior covered and `conditional` left as an explicit deterministic-ranking blocker.
- Re-exported the vector and hybrid helpers from `src/neataptic.ts`, refreshed generated docs via `npm run docs`, and kept `src/neataptic.ts` plus `src/neat/hybrid/neat.hybrid.ts` at `100/100/100/100` focused coverage.

## Controls and evidence

- Focused Jest slices for the serialize, training-isolation, hybrid-policy, and root-facade owner-local boundaries.
- Documentation regeneration via `npm run docs`.
- Repo-wide TypeScript validation via `npx tsc --noEmit -p tsconfig.json`, with only the unchanged external ONNX `TS2345` baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`.
- Final focused coverage evidence confirmed `src/architecture/network/training/network.training.isolate.utils.ts`, `src/neat/hybrid/neat.hybrid.ts`, and `src/neataptic.ts` at `100/100/100/100`, with `src/neat/hybrid/neat.hybrid.types.ts` treated as a type-only non-runtime surface.

## Reopen triggers

- A future consumer needs parameter-vector versioning or parameter-family coverage beyond the archived v1 contract.
- The hybrid policy surface needs a real deterministic ranking seam for `fineTune: 'conditional'` or worker-returned Lamarckian persistence semantics.
- Cross-runtime determinism or checkpoint integration needs stronger guarantees than the archived hybrid baseline currently makes.
