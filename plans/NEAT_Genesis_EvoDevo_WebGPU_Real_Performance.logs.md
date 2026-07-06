# NEAT Genesis EvoDevo: WebGPU Real Performance — Log

**Status:** [DONE]

## Phase 1 — Red Testing

[DONE] Step 01: Write failing GPU parity test.

- Test file: `src/architecture/network/gpu/network.gpu.parity-large.red.test.ts`.
- Fixture: 10-64-4 MLP with worker-keyed logistic activation; mock device
  `generateOutput` simulates the current buggy kernel (applies logistic per-node
  without weighted fan-in sums).
- Focused command:
  `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.gpu.parity-large.red`
- Result: 1 failed suite, 2 failed tests (expected red).
- Sample failure:
- maxAbsDiff received `0.03698859398476356` (expected `< 0.001`).
- meanAbsDiff received `0.05352582391539734` (expected `< 0.0001`).
- Handoff: Step 02 implementer should fix `src/architecture/network/gpu/network.gpu.kernel.ts`
  so the shader accumulates weighted incoming contributions before applying the
  activation function.
