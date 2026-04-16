# Flappy Recurrent Architecture Debug Pass

**Status:** [WIP]

## Scope

- Reproduce the live Flappy browser failures still reported for the recurrent
  architecture profiles after the earlier Step 7 integration fixes.
- Investigate why NARX training plateaus around zero pipes despite valid startup
  and whether that is expected profile behavior or a runtime regression.
- Reproduce and repair the GRU and LSTM duplicate-innovation failures reported
  after selecting those profiles in the browser UI.
- Determine whether warm-start or other initial-training behavior contributes to
  the live failures.
- Add temporary or narrowly-scoped debug instrumentation only where needed to
  identify the exact failing phase before broadening fixes.
- Follow the repo test-fix workflow: red-phase targeted reproduction tests
  first, then implementation, then focused green validation.

## Current state

- The main roadmap context remains
  `plans/Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md`, where Flappy Step 7
  is already complete and Step 8 is still the broader rollout frontier.
- Cluster A remains active: browser-default NARX still converges into a
  repeatable zero-pipe local optimum under the tiny interactive budget, even
  though the population stays structurally valid before playback, after
  playback, and after the winner-clone handoff.
- GRU and LSTM are now repaired. The actual root cause was not malformed seed
  serialization or a browser-only clone bug. The browser probe showed that both
  recurrent seeds begin strict-valid but carry many hidden nodes with no direct
  outgoing edge in the live graph: 48 for GRU and 72 for LSTM.
- `ensureNoDeadEnds()` was wrongly treating those module-internal hidden nodes
  as ordinary dead ends during generation-zero repair. That repair path added
  fresh low-id connections, which then collided with the seeds' existing low-id
  innovations and made `createGenomeFromNetwork()` fail at `runtime-init`.
- The fix protects hidden nodes owned by validated temporal recurrent-module
  descriptors. `src/architecture/network/network.temporal.extensions.utils.ts`
  now exposes `resolveTemporalRecurrentModuleNodeGeneIds(...)`, and
  `src/neat/mutation/repair/mutation.dead-ends.ts` skips hidden-node repair for
  those module-owned gene ids while the descriptor remains valid.
- Focused validation is green across the source and browser paths:
  `src/neat/mutation/repair/mutation.repair.test.ts`,
  `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.runtime.service.test.ts`,
  `npm run build:flappy-worker`, and the headless browser probe against
  `http://127.0.0.1:8080/examples/flappy_bird/index.html`.
- The current 8080 browser probe now shows GRU and LSTM are strict-valid at
  `seed-network-after-repair-probe` and `runtime-init`, with no duplicate
  innovations and `repairProbeNextInnovationId: 0`. The recurrent seeds still
  report the same hidden nodes without direct outgoing edges, but those nodes
  are now correctly left untouched.

## Coverage backlog

- [DONE] Cluster A diagnosis: NARX plateau reproduced and explained to >90%
  confidence as a browser-budget and objective local-optimum problem, not a
  structural runtime bug.
- [DONE] Cluster B repair: GRU duplicate innovations removed by protecting
  recurrent-module hidden nodes from generic dead-end repair during
  generation-zero bootstrap.
- [DONE] Cluster C repair: LSTM duplicate innovations removed by the same
  recurrent-module repair guard.
- [DONE] Cluster D owner confirmation: the failing owner was the hidden-node
  branch of `ensureNoDeadEnds()` operating on valid recurrent module internals,
  not warm-start, browser profile switching, or the generation-zero JSON clone
  shelf itself.
- [WIP] Remaining architecture question: decide how browser NARX should escape
  the tiny-budget zero-pipe local optimum without regressing the interactive
  demo experience.

## Immediate next steps

1. Decide whether to keep the temporary recurrent debug instrumentation while
   the NARX budget/objective pass is still open, or remove the GRU/LSTM-only
   probes now that the runtime-init corruption is fixed.
2. Design one narrow NARX browser intervention and validate it with the same
   live probe surface: larger population budget, different reward pressure,
   or both.
3. Refresh docs only if the public explanation of recurrent bootstrap or the
   browser architecture-profile behavior needs to mention the module-aware
   repair guard.

## Deferred questions

- Browser NARX still looks structurally healthy but behaviorally trapped. The
  remaining decision is whether to solve that by adjusting browser-default
  population budget, by biasing reward more strongly toward actual pipe
  progress, or by combining both changes.
- The current recurrent debug probes are still useful for live acceptance, but
  they should not become permanent noise once the NARX pass is complete.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Follow plans/flappy-recurrent-architecture-debug.plans.md. Treat GRU and LSTM
as fixed: the duplicate-innovation bug came from generic hidden dead-end repair
rewiring valid recurrent-module internals during generation-zero bootstrap, and
the browser probe on port 8080 now shows both profiles are strict-valid at
runtime-init. Keep the remaining work focused on browser NARX, which still
converges to a zero-pipe local optimum under the tiny interactive budget.
Choose one narrow NARX intervention, validate it with focused tests plus the
live browser probe, and only keep the recurrent debug instrumentation that is
still needed for that pass.
```
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"population-snapshot","phase":"generation-after-warm-start","architectureProfileId":"gru","generation":0,"populationSize":5,"population":[{"populationIndex":0,"nodeCount":184,"forwardConnectionCount":3920,"selfConnectionCount":0,"gateCount":960,"maxInnovation":3872,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[3873].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["118->38@null","55->61@null"]},{"innovation":2,"count":2,"endpoints":["118->39@null","56->125@null"]},{"innovation":3,"count":2,"endpoints":["118->40@null","57->120@null"]}]},{"populationIndex":1,"nodeCount":184,"forwardConnectionCount":3920,"selfConnectionCount":0,"gateCount":960,"maxInnovation":3872,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[3873].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["118->38@null","55->61@null"]},{"innovation":2,"count":2,"endpoints":["118->39@null","56->125@null"]},{"innovation":3,"count":2,"endpoints":["118->40@null","57->120@null"]}]},{"populationIndex":2,"nodeCount":184,"forwardConnectionCount":3920,"selfConnectionCount":0,"gateCount":960,"maxInnovation":3872,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[3873].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["118->38@null","55->61@null"]},{"innovation":2,"count":2,"endpoints":["118->39@null","56->125@null"]},{"innovation":3,"count":2,"endpoints":["118->40@null","57->120@null"]}]},{"populationIndex":3,"nodeCount":184,"forwardConnectionCount":3920,"selfConnectionCount":0,"gateCount":960,"maxInnovation":3872,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[3873].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["118->38@null","55->61@null"]},{"innovation":2,"count":2,"endpoints":["118->39@null","56->125@null"]},{"innovation":3,"count":2,"endpoints":["118->40@null","57->120@null"]}]},{"populationIndex":4,"nodeCount":184,"forwardConnectionCount":3920,"selfConnectionCount":0,"gateCount":960,"maxInnovation":3872,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[3873].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["118->38@null","55->61@null"]},{"innovation":2,"count":2,"endpoints":["118->39@null","56->125@null"]},{"innovation":3,"count":2,"endpoints":["118->40@null","57->120@null"]}]}]}

```

### LSTM

`error: Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)`

```
runtime.ts:106 [flappy-recurrent-debug] {"phase":"browser-profile-restart","fromArchitectureProfileId":"mlp","toArchitectureProfileId":"lstm"}
runtime.evolution-loop.service.ts:113 [flappy-recurrent-debug] {"phase":"browser-init-posted","architectureProfileId":"lstm","populationSize":5,"elitismCount":1,"inputSize":38,"outputSize":2}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"population-snapshot","phase":"runtime-init","architectureProfileId":"lstm","generation":0,"populationSize":5,"population":[{"populationIndex":0,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":1,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":2,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":3,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":4,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]}],"extra":{"populationSize":5,"elitismCount":1,"rngSeed":305441741}}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"marker","phase":"generation-request-received","architectureProfileId":"lstm","generation":0,"extra":{"currentPopulationSize":0,"warmStartApplied":false}}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"population-snapshot","phase":"generation-before-warm-start","architectureProfileId":"lstm","generation":0,"populationSize":5,"population":[{"populationIndex":0,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":1,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":2,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":3,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":4,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]}]}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"marker","phase":"warm-start-start","architectureProfileId":"lstm","generation":0,"extra":{"teacherStrategy":"rollout-only","trainingSetSize":512,"populationSize":5}}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"marker","phase":"warm-start-finished","architectureProfileId":"lstm","generation":0,"extra":{"teacherStrategy":"rollout-only","populationSize":5}}
flappy-evolution.worker.bundle.js:42 [flappy-recurrent-debug] {"kind":"population-snapshot","phase":"generation-after-warm-start","architectureProfileId":"lstm","generation":0,"populationSize":5,"population":[{"populationIndex":0,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":1,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":2,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":3,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]},{"populationIndex":4,"nodeCount":160,"forwardConnectionCount":4388,"selfConnectionCount":24,"gateCount":1080,"maxInnovation":4340,"strictGenomeOk":false,"strictGenomeError":"Connection innovations must stay unique across one strict genome. (connectionGenes[4341].innovation)","duplicateInnovations":[{"innovation":1,"count":2,"endpoints":["70->38@null","39->38@null"]},{"innovation":2,"count":2,"endpoints":["70->39@null","40->73@null"]},{"innovation":3,"count":2,"endpoints":["70->40@null","41->156@null"]}]}]}

```
