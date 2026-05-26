# Evolution-Training Interoperability Contracts Plan

**Status:** [DONE]

## Scope

- Define a deterministic, versioned parameter-vector contract owned by the network boundary.
- Add an isolated fine-tuning helper that never mutates shared candidate state accidentally.
- Add an explicit hybrid evaluation policy surface for fitness-only scoring versus Lamarckian persistence.
- Close the public docs and root-facade teaching surface needed for downstream consumers such as NEATchat.

## Final state

- The serialize-owned network boundary now ships `ParameterLayoutV1`, `ParameterVector`, `toParameterVector(...)`, and `fromParameterVector(...)` with deterministic ordering, compatibility checks, and mismatch rejection.
- The training-owned boundary now ships `fineTuneVector(...)`, `FineTuneOptions`, and `FineTuneResult`, using a cloned working copy so the supplied network and input vector stay unchanged.
- The hybrid-owned `src/neat/hybrid/` boundary now ships `evaluateCandidate(...)`, `HybridFineTuneMode`, `HybridEvaluationPolicy`, and result types, with `fineTune: 'never'` and `fineTune: 'always'` implemented and `fineTune: 'conditional'` left as an explicit deterministic-ranking blocker.
- The root facade now re-exports the vector and hybrid workflow surfaces, generated docs are refreshed through `npm run docs`, and the lane now serves as the archived pre-NEATchat hybrid baseline.

## Audit summary

- Phase 1 closed deterministic `ParameterLayoutV1` ordering and owner-local coverage for the layout-owned runtime surface.
- Phase 2 closed parameter-vector export/import, compatibility validation, roundtrip behavior, and focused coverage for the serialize-owned runtime surface.
- Phase 3 closed `fineTuneVector(...)` isolation and same-runtime deterministic training claims for the training-owned boundary, including source-first docs and `100/100/100/100` focused coverage for `src/architecture/network/training/network.training.isolate.utils.ts`.
- Phase 4 closed the explicit hybrid policy surface and Lamarckian opt-in behavior. `src/neat/hybrid/neat.hybrid.ts` remains at `100/100/100/100`, while `src/neat/hybrid/neat.hybrid.types.ts` stays an acknowledged type-only coverage surface.
- Phase 5 closed the public docs and facade pass: `npm run docs` passed, the focused hybrid and root-facade test slices passed, `src/neataptic.ts` and `src/neat/hybrid/neat.hybrid.ts` both validated at `100/100/100/100`, and repo-wide TypeScript still reproduced only the unchanged external ONNX `TS2345` baseline at `src/architecture/network/onnx/validate/network.onnx.validate.ts:135`.
- The hybrid-contract dependency for the NEATchat follow-up lane is now satisfied, but NEATchat remains planning-only until explicitly opened.

## Reopen conditions

- Layout version 1 needs to widen beyond the archived weights-and-biases contract, including disabled-connection or deferred-parameter semantics that would change import or export meaningfully.
- The repo gains a deterministic ranking surface that can honestly unblock `fineTune: 'conditional'`.
- Cross-runtime exact replay, worker-returned Lamarckian persistence, or checkpoint-owned parameter-vector guarantees need stronger contracts than this archived lane provides.

## Audit log

- See [Evolution_Training_Interoperability_Contracts.logs.md](Evolution_Training_Interoperability_Contracts.logs.md).
