# NEATchat

**Status:** [PLANNED]

## Scope

Plan a deliberately small chatbot example that starts with near-zero knowledge,
learns incrementally from short user exchanges, and showcases the library's
sequence-model surface without overstating its scale. Reuse the Flappy Bird
network visualizer path so architecture inspection and interaction affordances
stay consistent across demos.

## Current state

- The roadmap places `NEATchat` in Phase 3 as part of the learnability lane,
  not in the Phase 7 research stack.
- The core dependency is the Phase 2 preconfigured-architectures boundary,
  especially the sequence builders covered by
  [Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md](Preconfigured_Architectures_MLP_LSTM_GRU_NARX.md).
- Browser packaging, persistence, and worker-backed background execution remain
  separate concerns that should reuse their own Phase 3 or Phase 4 plans rather
  than being normalized into the first `NEATchat` slice.
- The visualizer baseline should be reused from the active Flappy visualizer
  workstream in
  [flappy-network-visualizer-hover-highlight.plans.md](flappy-network-visualizer-hover-highlight.plans.md)
  so `NEATchat` does not fork a second visualization stack.
- The default vocabulary cap target is `3000` words as a practical sweet spot:
  large enough to reduce early out-of-vocabulary churn versus 1500, but still
  small enough for responsive local experimentation in Node and browser demos.

## Goals

- G1: Ship a small conversational demo that can start effectively blank and
  adapt from short supervised exchanges.
- G2: Keep the model language-agnostic at the token-stream level so the demo is
  not hard-coded to English.
- G3: Show how recurrent builders, state reset, and incremental training can be
  used in a concrete example.
- G4: Keep the runtime small enough for local Node execution and realistic
  browser experimentation.
- G5: Reuse the Flappy Bird neural-network visualizer contract for graph
  rendering, hover highlight behavior, and basic interaction patterns.
- G6: Provide an optional copy-paste pretraining path that accepts large text
  input in a target language and learns from a capped top-frequency vocabulary
  (default 3000 words, user-configurable for limit experiments).
- G7: Provide immediate preprocessing transparency (token counts, kept terms,
  and coverage) so users can understand whether the pasted corpus is suitable.
- G8: Make blank-start versus preseeded behavior directly comparable in one
  session via a simple A/B prompt flow.
- G9: Include lightweight quality metrics so users can evaluate whether
  pretraining improved behavior instead of relying on subjective impressions.

## Non-goals

- Training on web-scale corpora.
- Positioning the feature as an LLM, agent framework, or general-purpose
  assistant.
- Solving long-context memory, retrieval, or factual grounding in the first
  version.
- Treating a pure Markov-chain Daisy clone as part of the neural-network
  surface.
- Guaranteeing full ingestion of every pasted token when text exceeds runtime
  limits; the capped-vocabulary path is intentionally selective.
- Guaranteeing uniform quality across all writing systems in the first
  tokenizer version.

## Proposed approach

### Step 1 - Define the example contract

- Choose the smallest usable public entrypoint, likely a compact example under
  `examples/` that wraps an LSTM, GRU, or NARX builder.
- Keep the interaction surface simple: receive a user message, tokenize,
  perform one short online training step against the observed reply, then
  generate a short response.
- Include a visualization contract up front that points to the Flappy Bird
  network visualizer implementation as the default rendering path.

Acceptance:

- The example can explain its input format, reset behavior, and expected output
  in a few paragraphs.

### Step 2 - Constrain vocabulary and sequence length

- Use a deliberately tiny vocabulary budget and short context window so the
  demo remains tractable.
- Favor a simple tokenization rule that can operate on arbitrary languages,
  even if quality is limited.
- Add a configurable `topWordLimit` setting with a default of `3000`, plus a
  recommended experiment range of `300-5000`.
- Surface a lightweight runtime estimate before training starts (for example,
  estimated retained vocabulary size and expected pretraining duration bucket)
  so users can choose safer limits.
- Include explicit out-of-vocabulary handling with a stable `UNK` path.

Acceptance:

- The example remains fast enough for repeated local interactions.
- Users can intentionally vary `topWordLimit` and observe predictable
  performance-quality tradeoffs.

### Step 3 - Optional copy-paste pretraining

- Add an optional pretraining entrypoint that accepts user-pasted text in the
  target language.
- Normalize and tokenize the pasted text, compute word frequencies, keep only
  the top `topWordLimit` terms, and train on that bounded vocabulary slice.
- Keep this flow explicitly optional so users can compare blank-start behavior
  versus preseeded behavior.
- Process large pasted text in bounded chunks internally so memory stays
  controlled while computing frequency statistics and updates.

Acceptance:

- Users can paste large text blocks, run pretraining, and observe different
  output style compared with blank-start runs.
- Pretraining remains stable for large pasted input without unbounded memory
  growth.

### Step 4 - Add corpus report and controls

- After paste and before training, display a concise corpus report including:
  character count, total tokens, unique terms, retained terms after cap, and
  retained-token coverage percentage.
- Expose `topWordLimit` in the UI or CLI surface with a clear default and
  recommended range.

Acceptance:

- Users can see exactly what portion of their pasted text drives training.

### Step 5 - Add online-learning loop

- After each exchange, apply a narrow supervised update using the observed user
  response history.
- Keep the update path explicit so users can inspect when the model learns and
  when it only performs inference.

Acceptance:

- The bot's output changes measurably after repeated short conversations.

### Step 6 - Add A/B interaction and lightweight evaluation

- Add a one-session A/B mode where the same prompt can be run against blank
  start and pretrained state.
- Track lightweight metrics: held-out next-token accuracy on a tiny validation
  slice, repetition rate, and response-length stability.

Acceptance:

- Users can run an A/B comparison and view basic quality metrics after
  pretraining.

### Step 7 - Document boundaries honestly

- Explain that `NEATchat` is a toy sequence-learning demo and not a
  transformer-scale language model.
- State clearly that learning any language means "any token stream the small
  tokenizer can encode," not broad semantic mastery.
- Document the practical tradeoff: larger `topWordLimit` often improves lexical
  coverage but increases runtime and memory pressure.

Acceptance:

- The example docs set expectations without implying web-scale capability.

### Step 8 - Plan later follow-ons separately

- Defer persistence to
  [Population_Save_Resume_and_Checkpointing.md](Population_Save_Resume_and_Checkpointing.md).
- Defer background or worker-assisted learning to
  [Turnkey_Multithread_Evaluation_API.md](Turnkey_Multithread_Evaluation_API.md)
  and
  [Worker_Friendly_Network_Serialization_Fastpath.md](Worker_Friendly_Network_Serialization_Fastpath.md).
- Defer any formal training-plus-evolution hybrid experiments to
  [Evolution_Training_Interoperability_Contracts.md](Evolution_Training_Interoperability_Contracts.md).
- Defer any `NEATchat`-specific visualizer polish until reuse of the Flappy
  visualizer baseline is complete.

Acceptance:

- The first version stays narrow and the expansion seams are explicit.

## Risks and mitigations

- Risk: users may read "chatbot" and infer LLM-scale capability.
  - Mitigation: keep the docs explicit about vocabulary, context, and training
    limits.
- Risk: a blank-start model may look too unintelligent at first contact.
  - Mitigation: provide a tiny optional seed corpus while preserving the true
    from-scratch path.
- Risk: language-agnostic tokenization may be too naive for some scripts.
  - Mitigation: document the tokenizer contract and keep the first version easy
    to replace.
- Risk: very large pasted text may cause long pretraining time or memory spikes.
  - Mitigation: stream or chunk input internally, and always apply the
    top-frequency vocabulary cap before model updates.
- Risk: low vocabulary caps produce frequent unknown tokens and low-quality
  outputs.
  - Mitigation: use `UNK`, show retained-token coverage, and keep `topWordLimit`
    user-tunable with a higher default.

## Success criteria

- A user can run `NEATchat`, hold a short conversation, and observe the model
  adapting over repeated exchanges.
- The example teaches sequence builders and online learning more clearly than a
  synthetic benchmark alone.
- The docs make scale limits obvious enough that web-scale training is not
  implied.
- The demo renders its network through the reused Flappy visualizer path rather
  than a parallel custom renderer.
- Users can optionally paste training text and run capped-vocabulary
  pretraining with a visible `topWordLimit` setting (default 3000).
- The demo reports corpus statistics and retained-token coverage before
  pretraining begins.
- Users can compare blank-start versus pretrained responses side by side and
  inspect lightweight metrics after each run.

## Handoff query

```text
Continue from the current repo state and plan documents.
Work on the planned `NEATchat` Phase 3 example as a tiny online sequence-learning chatbot, not a large-scale transformer trainer. Reuse the preconfigured sequence builders, keep the first slice narrow, and reuse the Flappy Bird network visualizer baseline before adding any `NEATchat`-specific visualizer polish. Include the optional copy-paste pretraining flow with a configurable top-word cap (default 3000), corpus report fields, chunked ingestion, `UNK` handling, and an A/B plus lightweight-metrics loop so users can compare blank-start and preseeded behavior.
```
