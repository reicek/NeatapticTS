# NEATchat Follow-up

**Status:** [PLANNED]

## Scope

Reopen NEATchat as a post-toy conversational-systems lane that builds on the
closed Phase 3 demo without rewriting its acceptance boundary. The follow-up
goal is to move NEATchat closer to the same development path as modern systems
while staying honest about scale and preserving the library's strengths:
inspectable recurrent networks, local adaptation, deterministic artifacts,
browser publishability, and source-owned training behavior.

This plan treats the archived Phase 3 NEATchat work as the stable teaching
baseline and defines the next lane: stronger pretrained seeds, persistent
memory, checkpointed personal state, worker-backed background adaptation,
hybrid retrieval-plus-generation behavior, and explicit evaluation contracts.

## Current state

- The original NEATchat plan is closed in
  `plans/completed/NEATchat.plans.md`; that baseline delivered a toy-scale
  recurrent chat demo with bounded pretraining, online adaptation, A/B
  comparison, and explicit documentation of its limits.
- The current implementation is still intentionally small: one-hot vocabulary,
  short context windows, lightweight next-token metrics, and optional local
  pretraining rather than a durable system identity or long-lived memory.
- The demo already has valuable assets worth preserving:
  - browser-hosted and docs-published visibility,
  - builder-backed LSTM/GRU/NARX seed networks,
  - session snapshot helpers,
  - inspectable visualization reuse via the Flappy Bird renderer,
  - explicit contracts instead of hidden runtime magic.
- The main gaps versus more modern conversational systems are now structural,
  not cosmetic:
  - no durable checkpointed identity or user memory,
  - no stronger imported pretrained base,
  - no retrieval or episodic recall layer,
  - no worker-backed background learning or candidate search,
  - no hybrid evaluation policy for evolve-plus-train experimentation,
  - no stronger regression harness than the current lightweight metrics.
- This reopen lane depends on earlier roadmap work that is not yet available
  as a NEATchat-owned implementation substrate:
  - `plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md` is
    [DONE], so the shared inference IR and transport APIs now exist as an
    archived baseline.
  - `plans/completed/Turnkey_Multithread_Evaluation_API.md` is [DONE], so the
    worker-pool evaluation surface already exists as an archived baseline for
    chat inference or background adaptation.
  - `plans/completed/Population_Save_Resume_and_Checkpointing.md` is [DONE],
    so a stable checkpoint and resume contract now exists for a durable
    NEATchat identity baseline.
  - `plans/Evolution_Training_Interoperability_Contracts.md` is still
    [PLANNED], so parameter-vector export/import and isolated fine-tune
    policies are not available yet.
  - `plans/ONNX_EXPORT_PLAN.md` is [WIP], but recurrent import hardening is
    still in progress and is not yet an honest foundation for the supported
    external-seed subset this lane wants to consume.
- Until those earlier lanes expose usable public seams, this file is a
  dependency-gated sequencing contract rather than an implementation-ready
  workstream.

## Dependency gates

Do not move this plan from [PLANNED] to [WIP] until all of the following are
true:

1. Checkpointing exposes a stable save/resume surface that can carry a
   NEATchat identity, memory bank, vocabulary metadata, and deterministic
   replay context.
2. Worker payload and multithread evaluation work expose a usable background
   execution path for chat inference, candidate scoring, and adaptation jobs,
   with a deterministic single-thread fallback.
3. Evolution-training interoperability exposes deterministic parameter-vector
   export/import or an equivalent isolated fine-tune contract for candidate
   comparison and promotion.
4. Recurrent ONNX import hardening can honestly support the exact external
   seed subset chosen for NEATchat, or an explicitly documented non-ONNX
   conversion path replaces it.

Roadmap sequencing note:

- In the agreed serial pre-NGE sequence, this plan stays after the archived Track 1 stop line in `plans/completed/Memory_Optimization.md`, after `plans/ONNX_EXPORT_PLAN.md`, and after `plans/Evolution_Training_Interoperability_Contracts.md`.
- Finishing Memory alone is not a handoff into NEATchat; the required stop line still passes through ONNX hardening and hybrid interoperability first.

Before those gates open, work on this file should stay limited to dependency
alignment, seed-subset reconnaissance, and acceptance-boundary tightening.

## Follow-up thesis

The next useful NEATchat is not "pretend to be a transformer." The next useful
NEATchat is a transparent, persistent, locally owned conversational system that
combines:

- a stronger recurrent pretrained base,
- explicit short-term and long-term memory layers,
- background adaptation that keeps learning between visible exchanges,
- retrieval-style grounding from saved user or corpus memory,
- hybrid policies that can compare and refine multiple candidate responses,
- checkpointed artifacts that make progress portable and inspectable.

That keeps the system on a modern path without turning the repo into an opaque
wrapper around an external hosted model.

## Goals

- G1: Replace the current internal warm-start snapshot with a meaningfully
  stronger pretrained seed while keeping the runtime local and inspectable.
- G2: Introduce persistent user or workspace memory that survives sessions and
  can be checkpointed, resumed, compared, and replayed deterministically.
- G3: Add retrieval-like memory augmentation so generation can draw from saved
  exchanges, curated corpora, and durable summaries instead of relying only on
  recurrent hidden state.
- G4: Support background adaptation in Node and browser workers so the visible
  chat surface stays responsive while training, reranking, or evaluation runs.
- G5: Define an honest external-seed import path for compatible recurrent ONNX
  or adjacent source models, then distill or convert them into native
  NEATchat-owned artifacts.
- G6: Add a model-routing layer that can compare base, personalized, and
  retrieval-grounded candidates instead of committing to one monolithic path.
- G7: Expand evaluation from toy next-token checks to a durable regression
  suite covering conversational quality, memory use, stability, and safety.
- G8: Keep the full system inspectable through visualization and artifact
  export, including what memory was recalled, which candidate won, and what
  changed during adaptation.

## Non-goals

- Claiming parity with frontier transformer-scale assistants.
- Turning NEATchat into a thin wrapper around a hosted API.
- Accepting arbitrary ONNX models as supported imports before the recurrent
  subset is stable and documented.
- Hiding major behavior behind prompt-only tricks that the library cannot own,
  serialize, replay, or test.
- Collapsing this lane back into the already closed Phase 3 toy-demo scope.

## Recommended agent + skill combo by workstream

Use `NEATchat Scout` + `neatchat-systems` as the default pairing for each
workstream below. The dependency-gated Phase 4 and Phase 6 plans named above
remain prerequisite owners until their public seams are actually available.

- Workstream 1 — `NEATchat Scout` + `neatchat-systems`
- Workstream 2 — `NEATchat Scout` + `neatchat-systems`
- Workstream 3 — `NEATchat Scout` + `neatchat-systems`
- Workstream 4 — `NEATchat Scout` + `neatchat-systems`
- Workstream 5 — `NEATchat Scout` + `neatchat-systems`
- Workstream 6 — `NEATchat Scout` + `neatchat-systems`

## Coverage backlog

### [PLANNED] Workstream 1 — Reopen on durable infrastructure

Build the substrate that lets NEATchat keep learning across sessions instead of
restarting from an ephemeral demo state.

This workstream opens only after checkpointing, worker payloads, multithread
evaluation, and parameter-vector contracts all expose usable public seams.

- Define `NEATchat` session snapshot v2 as a thin application-level layer on
  top of the checkpointing plan rather than a one-off ad hoc format.
- Reuse the worker-friendly inference payload work so chat inference,
  candidate scoring, and adaptation jobs can move off the main thread.
- Reuse the parameter-vector contract so personalized variants and candidate
  fine-tunes can be compared without mutating shared base state.
- Preserve exact replay where practical: same seed, same memory bank, same
  candidate-selection policy, same outputs within floating-point tolerance.

Acceptance:

- A saved NEATchat identity can be paused, resumed, and migrated across
  sessions without losing vocabulary, memory, candidate policies, or training
  counters.
- Background evaluation and adaptation can run in workers with deterministic
  task ordering and a single-thread fallback.

### [PLANNED] Workstream 2 — Stronger pretrained seed import and distillation

Replace the current bundled warm-start with a materially stronger seed model.

This workstream opens only after Workstream 1's substrate is available and the
supported recurrent import or conversion path is honest about what it can host.

- Pick a supported recurrent seed family first, not an arbitrary public chat
  model. The near-term target is a compact GRU or LSTM language model trained
  on open dialogue corpora and converted offline.
- Define the supported external-seed subset: recurrent next-token models only,
  bounded vocabulary, compatible activation families, and explicit fallback for
  unsupported operators.
- Add an offline conversion path:
  external checkpoint -> compatible ONNX or vector bridge -> native network ->
  NEATchat session snapshot.
- Treat imported models as teacher artifacts; when direct import is awkward,
  distill them into NEATchat-native recurrent builders instead of forcing the
  runtime to emulate a foreign stack.

Acceptance:

- The shipped default seed materially outperforms the current internal snapshot
  on held-out conversation quality and stability.
- The docs can name the supported external-seed subset honestly and point to a
  reproducible conversion flow.

### [PLANNED] Workstream 3 — Multi-tier memory and retrieval

Move beyond "whatever the recurrent state remembers" by adding explicit memory
layers with different time horizons.

- Short-term memory: current context window plus immediate exchange state.
- Episodic memory: past exchanges, corrections, and user-specific facts stored
  as retrievable examples.
- Semantic memory: compact summaries or distilled notes derived from repeated
  exchanges or imported corpora.
- Retrieval policy: rank which memory items matter for the current prompt and
  feed them into response generation or candidate scoring.

Potential implementation shape:

- retrieved exemplar sequences used as additional training or scoring context,
- summary nodes or memory-bank descriptors attached to a checkpointed profile,
- explicit provenance showing which memory items were selected for a reply.

Acceptance:

- The system can remember user-specific facts across sessions.
- Retrieved memories improve relevance without causing hidden irreversible
  drift.

### [PLANNED] Workstream 4 — Background adaptation and candidate search

Make learning continuous and less brittle by decoupling visible chat from slow
update loops.

- Run background fine-tuning, candidate scoring, and replay evaluation in
  workers.
- Maintain a frozen base model plus one or more candidate personalized deltas.
- Evaluate candidate replies using lightweight preference heuristics,
  regression prompts, or user feedback before promoting a new personalized
  state.
- Keep promotion explicit: no silent mutation of the shipped base artifact.

Acceptance:

- The visible chat UI remains responsive while adaptation jobs run.
- Users can inspect when a new personalized checkpoint was proposed, accepted,
  or rejected.

### [PLANNED] Workstream 5 — Hybrid routing and specialist submodels

Push NEATchat toward system-like behavior without pretending one network should
do every job equally well.

- Introduce a small routing layer that can choose among:
  - the shipped pretrained base,
  - a personalized adapted variant,
  - a retrieval-grounded response path,
  - optional domain-specialist heads or snapshots.
- Explore Lamarckian or evolutionary policy search for candidate selection,
  memory weighting, and specialist handoff logic once the training contract is
  stable.
- Keep routing outputs inspectable: which path won, which memories were used,
  and why the selected candidate beat the alternatives.

Acceptance:

- NEATchat can produce and compare multiple candidate responses before
  selecting one.
- Routing decisions are logged and reproducible enough for debugging and
  regression testing.

### [PLANNED] Workstream 6 — Evaluation, safety, and publishable product shape

Modern systems improve because they can measure themselves. This lane needs the
same discipline.

- Build a regression suite that goes beyond next-token accuracy:
  - factual consistency against saved profile memory,
  - repetition and collapse resistance,
  - response helpfulness on small curated tasks,
  - safe handling of unknowns,
  - stability after repeated online updates.
- Define failure buckets so we can tell whether a regression came from the base
  seed, retrieval, memory compression, routing, or background adaptation.
- Keep the browser-hosted surface publishable, but split product polish from
  experimental lanes so docs stay honest.

Acceptance:

- Follow-up releases can show whether the system actually improved, not just
  changed.
- The public demo remains truthful about what is live, what is experimental,
  and what depends on background jobs.

## Sequencing guidance

Recommended order:

1. Finish enough of checkpointing, worker payloads, multithread evaluation,
   and training-vector contracts to give NEATchat a durable substrate.
2. Finalize the supported recurrent external-seed subset and ship a stronger
   bundled pretrained base.
3. Add persistent episodic memory and retrieval before attempting aggressive
   personalized background learning.
4. Add candidate routing and specialist variants only after regression signals
   can explain why they help.

This lane should not reopen by inflating the closed toy-demo contract first.
It should reopen by standing on the active Phase 4 and Phase 6 foundations.

## Activation gates and first tasks

Before any NEATchat-owned implementation starts:

1. `plans/completed/Population_Save_Resume_and_Checkpointing.md` must remain
  the stable checkpoint surface that owns NEATchat identity and memory state.
2. `plans/completed/Worker_Friendly_Network_Serialization_Fastpath.md` and
  `plans/completed/Turnkey_Multithread_Evaluation_API.md` must remain the
  usable background-execution path with deterministic fallback behavior.
3. `plans/Evolution_Training_Interoperability_Contracts.md` must expose the
   parameter-vector or isolated fine-tune seam that personalized candidates
   depend on.
4. `plans/ONNX_EXPORT_PLAN.md` must harden the recurrent import subset, or the
   repo must choose and document a non-ONNX conversion path for the seed model.

Once those gates are open, the first NEATchat-owned tasks are:

1. Define the supported external-seed target precisely: likely single-layer or
   compact stacked GRU/LSTM next-token models with bounded vocabulary and no
   embedding-only assumptions that NEATchat cannot yet host.
2. Draft `NEATchat` snapshot/checkpoint v2 requirements on top of the active
   checkpointing plan.
3. Write a memory-bank contract covering episodic records, semantic summaries,
   retrieval ranking inputs, and promotion rules.
4. Define the first regression pack that the stronger pretrained seed must beat
   before it replaces the current bundled snapshot.

## Deferred questions

- Should the next lane keep pure one-hot token IO, or introduce an explicit
  embedding boundary as a separately planned runtime evolution?
- Should external seeds land through ONNX import first, or should a direct
  parameter-vector conversion path be allowed when it is more reliable?
- What is the right first memory primitive: retrieved exemplars, durable user
  fact cards, compressed summaries, or all three behind one interface?
- When routing chooses among multiple candidates, should promotion be purely
  heuristic, user-confirmed, or partially evolved?

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
Work on the planned NEATchat follow-up lane in plans/NEATchat.plans.md. Treat the archived Phase 3 toy-demo baseline in plans/completed/NEATchat.plans.md as closed scope. If the dependency gates in the active plan are not yet met, do not start NEATchat implementation; instead, only tighten the dependency contract or advance the blocking plans. The first NEATchat-owned implementation slice begins only after checkpointing, worker payloads, multithread evaluation, parameter-vector export/import, and recurrent ONNX seed import or conversion are usable enough to support an honest external-seed target.
```