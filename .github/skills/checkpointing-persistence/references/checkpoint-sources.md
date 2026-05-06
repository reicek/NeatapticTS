# Checkpoint Reference Notes

This file stores paraphrased reference notes for the
`checkpointing-persistence` skill.

These notes summarize upstream material instead of copying it verbatim. Use the
linked sources for canonical wording and details.

## Source Map

### 1. Wikipedia overview of pseudorandom number generators

- URL: https://en.wikipedia.org/wiki/Pseudorandom_number_generator
- Authors: Wikipedia contributors
- License note: Wikipedia text is available under CC BY-SA 4.0.
- Why it matters:
  - A PRNG is fully determined by its internal state, not only by its original
    seed.
  - Reproducibility is one of the main practical reasons to use a deterministic
    generator.
  - For exact resume, persisting only the seed is not the same as persisting the
    current generator state after many draws.

### 2. Node.js `node:v8` serialization API

- URL: https://nodejs.org/api/v8.html#serialization-api
- Upstream repo: https://github.com/nodejs/node
- Why it matters:
  - The API is compatible with structured clone and is backward-compatible for
    stored data, but equal JavaScript values can still serialize to different
    byte sequences.
  - That makes it useful for internal tooling but a risky default for public,
    deterministic, cross-runtime checkpoint bytes.
  - The header and wire-format APIs reinforce the importance of explicit version
    handling.

### 3. Internal repo checkpoint-adjacent serialization memory

- Source: `/memories/repo/network_serialize_identity_boundary.md`
- Why it matters:
  - Network serialization already preserves innovation identity, endpoint gene
    identity, enabled state, and restore-counter synchronization.
  - Checkpoint work must build above that boundary, not duplicate it.
  - The orchestration checkpoint still needs evolution-level state that network
    serialization does not own.

## Practical Notes

### Seed vs state

- Seed answers: how was the generator initialized?
- State answers: where in the generator sequence is the run right now?
- Exact resume needs the second one.

### Public schema stability

- Public checkpoints should be explicit and versioned.
- Stable JSON is usually the best v1 contract because users can inspect it,
  diff it, and migrate it deliberately.
- Opaque binary blobs may be acceptable later, but only when migration and
  portability stories are clear.

### Full vs light checkpoints

- Full mode is for replay fidelity.
- Light mode is for useful restart state.
- Confusing these modes creates false reproducibility claims and hard-to-debug
  user trust issues.

## Working Heuristics For This Repo

- Reuse network serialization for graph identity, but own orchestration-level
  state separately.
- Keep strict restore failure explicit.
- Version every public checkpoint shape.
- Treat missing RNG state, missing counters, or missing adaptive state as a hard
  stop for exact replay.
- Reserve downstream metadata space so NEATchat and other applied systems can
  extend the checkpoint surface without forking it.