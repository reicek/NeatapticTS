# NEATchat Reference Notes

This file stores paraphrased reference notes for the `neatchat-systems` skill.

These notes summarize upstream sources instead of copying them verbatim. Use the
linked sources for canonical wording and details.

## Source Map

### 1. NEATchat follow-up plan in this repo

- Source: `plans/NEATchat.plans.md`
- Why it matters:
  - This is the primary architectural source for the post-toy NEATchat lane.
  - It defines dependency gates, workstreams, activation boundaries, and explicit
    non-goals.
  - It frames NEATchat as a consumer of checkpointing, worker transport,
    multithread evaluation, interoperability, and recurrent import work rather
    than a replacement for those foundations.

### 2. Existing toy example README

- Source: `examples/neatChat/README.md`
- Why it matters:
  - It documents the current toy LSTM browser demo, bounded pretraining flow,
    reset behavior, and public-facing minimal surface.
  - It shows the baseline that the future system should preserve or intentionally
    supersede, not accidentally overwrite.

### 3. Repo memory: NEATchat warm-start and snapshot boundary

- Source: `/memories/repo/neatchat_warm_start_snapshot_boundary.md`
- Why it matters:
  - Warm-starting from a bundled sample chunk is the current no-corpus fallback.
  - Snapshot persistence is the most practical path for save or resume behavior.
  - Browser entry owns only the minimal snapshot UI; deeper snapshot semantics
    belong to the reusable implementation surface.

### 4. Wikipedia: dialogue system

- URL: https://en.wikipedia.org/wiki/Dialogue_system
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Dialogue systems are typically organized around a dialogue manager that sits
    between input understanding, task managers, and response generation.
  - This is a useful corrective when chat work starts collapsing every concern
    into one opaque model step.

### 5. Wikipedia: episodic memory

- URL: https://en.wikipedia.org/wiki/Episodic_memory
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Episodic memory emphasizes event-specific context, temporal order, and the
    distinction between remembering an episode and storing a durable fact.
  - That distinction maps well to chat history versus distilled user profile or
    semantic memory.

### 6. Wikipedia: information retrieval

- URL: https://en.wikipedia.org/wiki/Information_retrieval
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - Retrieval is about matching an information need against a collection and
    ranking possible results by relevance.
  - Sparse, dense, and hybrid retrieval language helps describe NEATchat memory
    lookup without pretending it is just one exact database hit.

## Practical Notes

### Persistent chat is a systems problem

- A useful NEATchat follow-up is not only a stronger model.
- It is a dialogue manager, memory stack, retrieval policy, checkpoint strategy,
  and adaptation workflow that remain inspectable together.

### Episodic and semantic memory should not collapse together

- Conversation turns are event-like and branch-specific.
- Stable user preferences or long-lived facts should usually be promoted from
  repeated evidence, not written immediately into durable profile state.

### Retrieval should be ranked and attributable

- Local memory lookup still needs scoring, provenance, and failure handling.
- A surfaced memory should have a visible reason for inclusion, not magic.

### Snapshot persistence is safer than live drift

- The repo already learned that save or resume behavior works best through
  explicit snapshots.
- That same lesson should shape background adaptation and profile branching.

### Foundation ownership matters

- If NEATchat needs worker payload support, checkpoint semantics, or vector import
  or export changes, those remain owner-local to the underlying foundation skill.
- The NEATchat layer should coordinate, not absorb, those responsibilities.

## Working Heuristics For This Repo

- Keep the toy example small and the follow-up system explicit.
- Favor branchable checkpoints over hidden in-place personalization.
- Promote facts cautiously from episodic evidence into durable profile state.
- Evaluate memory recall, reset fidelity, and retrieval provenance separately from
  response style.
- Keep every durable memory write inspectable and reversible.
