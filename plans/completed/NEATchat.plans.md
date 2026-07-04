# NEATchat

**Status:** [DONE]

## Scope

Plan a deliberately small chatbot example that starts with near-zero knowledge,
learns incrementally from short user exchanges, and showcases the library's
sequence-model surface without overstating its scale. Reuse the Flappy Bird
network visualizer path so architecture inspection and interaction affordances
stay consistent across demos.

## Current state

- The roadmap places `NEATchat` in Phase 3 as part of the learnability lane,
  not in the Phase 7 research stack.
- The first public contract slice now lives under
  `examples/neatChat/`, with a compact Node entrypoint, a builder-backed seed
  network wrapper, a browser-hosted flagship-page preview that is publishable
  through `docs/examples/`, and explicit visualizer reuse metadata that points
  to the Flappy host plus network-view boundary instead of forking a second
  renderer.
- The shared contract and browser-hosted flagship page now also surface the
  Step 2 vocabulary budget directly: user-tunable `topWordLimit`, a short
  context-window contract, a stable `UNK` path, and a lightweight runtime
  estimate that stays aligned between Node and docs-published browser output.
- The published browser preview now also accepts an optional pasted corpus for
  Step 3, processes it in bounded chunks, prepares a retained-vocabulary slice
  for the preseeded transcript, and keeps that pretraining path optional so
  blank-start versus preseeded behavior can be compared from one page.
- The browser and Node-facing Step 4 surfaces now expose the pretraining
  controls more clearly: the published preview shows a concrete corpus report
  with character count, total tokens, unique terms, retained terms, and
  retained-token coverage, while the shared contract text calls out the
  default `topWordLimit` and recommended range directly.
- Step 7 documentation boundaries are now explicit in the NEATchat public
  surfaces: the README and browser-hosted flagship intro both state toy-scale
  scope, token-stream language semantics, the `topWordLimit` runtime-memory
  tradeoff, and the distinction between live behavior and contract-preview
  scaffolding.
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

### Step 1 - Define the example contract [DONE]

- Choose the smallest usable public entrypoint, likely a compact example under
  `examples/` that wraps an LSTM, GRU, or NARX builder.
- Mirror that contract through a browser-hosted page under `examples/neatChat/`
  so the docs publication flow can surface `NEATchat` in the flagship section
  even before the heavier learning loop is complete.
- Keep the interaction surface simple: receive a user message, tokenize,
  show a truthful browser-visible chat preview immediately, then grow toward
  one short online training step against the observed reply and a short
  generated response.
- Include a visualization contract up front that points to the Flappy Bird
  network visualizer implementation as the default rendering path.
- Expose visible progress on the browser page so users can inspect which
  slices are already shipped versus still planned without depending on the Node
  runner.

Acceptance:

- The example can explain its input format, reset behavior, and expected output
  in a few paragraphs.
- The published browser page shows a visible chat-shaped surface and current
  progress from the docs examples page.

### Step 2 - Constrain vocabulary and sequence length [DONE]

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
- Keep the browser-hosted flagship page aligned with the same limit defaults,
  runtime estimate language, and `UNK` behavior so docs-published users see
  the same vocabulary and runtime budget guidance as Node users.
- Surface those controls and estimates directly on the published browser page,
  not only in CLI or Node help text, so users can inspect the active budget
  and expected cost before they start pretraining.

Acceptance:

- The example remains fast enough for repeated local interactions.
- Users can intentionally vary `topWordLimit` and observe predictable
  performance-quality tradeoffs.
- The shared Node contract and published browser page expose the same
  vocabulary-cap, context-window, and runtime-estimate guidance.
- The published browser page exposes the same vocabulary cap guidance and
  runtime estimate cues as the Node entrypoint.

### Step 3 - Optional copy-paste pretraining [DONE]

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
- The published browser preview accepts pasted corpus text, builds the
  preseeded transcript from a bounded retained-vocabulary slice, and preserves
  the blank-start comparison path.

### Step 4 - Add corpus report and controls [DONE]

- After paste and before training, display a concise corpus report including:
  character count, total tokens, unique terms, retained terms after cap, and
  retained-token coverage percentage.
- Expose `topWordLimit` in the browser and CLI surfaces with a clear default
  and recommended range.

Acceptance:

- Users can see exactly what portion of their pasted text drives training.
- The published browser preview shows the full corpus report before training,
  and the Node contract text exposes the same top-word-cap default and
  recommended range.

### Step 5 - Add online-learning loop [DONE]

- After each exchange, apply a narrow supervised update using the observed user
  response history.
- Keep the update path explicit so users can inspect when the model learns and
  when it only performs inference.

Acceptance:

- The bot's output changes measurably after repeated short conversations.

### Step 6 - Add A/B interaction and lightweight evaluation [DONE]

- Add a one-session A/B mode where the same prompt can be run against blank
  start and pretrained state.
- Track lightweight metrics: held-out next-token accuracy on a tiny validation
  slice, repetition rate, and response-length stability.

Acceptance:

- Users can run an A/B comparison and view basic quality metrics after
  pretraining.

### Step 7 - Document boundaries honestly [DONE]

- Explain that `NEATchat` is a toy sequence-learning demo and not a
  transformer-scale language model.
- State clearly that learning any language means "any token stream the small
  tokenizer can encode," not broad semantic mastery.
- Document the practical tradeoff: larger `topWordLimit` often improves lexical
  coverage but increases runtime and memory pressure.
- Keep the published flagship page honest about which visible elements are live
  runtime behavior versus contract-preview scaffolding.

Acceptance:

- The example docs set expectations without implying web-scale capability.

### Step 8 - Plan later follow-ons separately [DONE]

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
- Keep the browser-hosted flagship surface publishable throughout the staged
  rollout so progress remains visible even while those later follow-ons stay
  deferred.

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
- The browser-hosted example is visible from the flagship section of
  `docs/examples/index.html`.
- Users can optionally paste training text and run capped-vocabulary
  pretraining with a visible `topWordLimit` setting (default 3000).
- The demo reports corpus statistics and retained-token coverage before
  pretraining begins.
- Users can compare blank-start versus pretrained responses side by side and
  inspect lightweight metrics after each run.

## Sample conversation

8227 characters, 1492 tokens, 522 unique terms, one interaction per line, no special tokens, and a mix of common and uncommon words.

Can be passed in sequence.

```json
[
 "Hi, how was your day?",
 "It was pretty good, thanks for asking. How about yours?",
 "Not too bad. I had a lot of meetings, but I managed to get through them.",
 "Meetings can be exhausting. Did anything interesting happen?",
 "Actually, yes. One of the meetings was about a new project we might start next month.",
 "That sounds exciting! What kind of project is it?",
 "It’s a collaboration with another department to develop a new app.",
 "Wow, that’s cool. Are you going to be leading the project?",
 "I’m not sure yet, but I hope so. I’ve always wanted to manage something like this.",
 "You’d be great at it. You’re organized and creative.",
 "Thanks, that means a lot. What about you? Any news at work?",
 "Nothing major, but I did finish a big report I’d been working on for weeks.",
 "That’s a relief, I bet. Was it stressful?",
 "A little, but mostly I just wanted to make sure everything was perfect.",
 "I know the feeling. Sometimes I double-check things too much.",
 "Better to be thorough than to miss something important.",
 "True. Do you have any plans for the weekend?",
 "Not really. I was thinking of just relaxing at home. You?",
 "I might go hiking if the weather’s nice. Want to join?",
 "That sounds fun! I haven’t been hiking in ages.",
 "It’s a great way to clear your mind. Plus, the views are amazing.",
 "Where do you usually go?",
 "There’s a trail about an hour from here. It’s not too difficult, but the scenery is beautiful.",
 "I’d love to check it out. Should we bring snacks?",
 "Definitely. I’ll pack some sandwiches and fruit.",
 "I can bring water and maybe some energy bars.",
 "Perfect. What time should we leave?",
 "How about 9 a.m.? That way we can avoid the midday heat.",
 "Sounds good to me. Should we invite anyone else?",
 "Maybe. Do you think Sarah would want to come?",
 "She might. I’ll text her and see if she’s interested.",
 "Great. The more, the merrier.",
 "Agreed. Do you remember the last time we all went hiking?",
 "Yeah, it was last spring. We got caught in that rainstorm.",
 "That was wild! We were soaked by the time we got back to the car.",
 "But it was still a lot of fun. We laughed the whole way.",
 "Sometimes the unexpected moments are the best.",
 "Absolutely. Speaking of unexpected, did you hear about the new restaurant opening downtown?",
 "No, I hadn’t. What kind of food do they serve?",
 "It’s a fusion place—mix of Asian and Mediterranean.",
 "That sounds delicious. We should try it sometime.",
 "I’m in! Maybe after our hike?",
 "Great idea. We’ll probably be hungry.",
 "For sure. I’ll make a reservation just in case.",
 "Good thinking. It’s probably going to be busy.",
 "Yeah, new places always are. I hope the food lives up to the hype.",
 "Me too. I love trying new things.",
 "Same here. It keeps life interesting.",
 "So, what else is new with you?",
 "Not much, honestly. I’ve been reading a lot lately.",
 "Anything good?",
 "I just finished a mystery novel. It was a real page-turner.",
 "I love mysteries. Who was the author?",
 "Her name is Lisa Gardner. Have you read any of her books?",
 "I don’t think so. Would you recommend it?",
 "Definitely. The plot twists kept me guessing until the end.",
 "I’ll have to check it out. I’ve been looking for something new to read.",
 "I can lend you my copy if you want.",
 "Thanks! I’ll take you up on that.",
 "No problem. What kind of books do you usually like?",
 "Mostly thrillers and historical fiction.",
 "Nice. I like those too. There’s something about getting lost in another time.",
 "Exactly. It’s like traveling without leaving your chair.",
 "Well said. Do you ever listen to audiobooks?",
 "Sometimes, especially when I’m driving.",
 "They’re great for long commutes. I listened to one last week that was really good.",
 "Which one?",
 "It was “The Night Circus” by Erin Morgenstern.",
 "I’ve heard of that! Did you like it?",
 "I loved it. The narration was fantastic.",
 "I’ll add it to my list. Thanks for the recommendation.",
 "Anytime. So, have you watched any good movies lately?",
 "I saw “Dune” last weekend. It was amazing.",
 "I’ve been meaning to watch that. Was it true to the book?",
 "Pretty much. They did a great job with the visuals.",
 "I’ll have to see it soon. Maybe we can watch it together.",
 "That would be fun. Movie night?",
 "Absolutely. Popcorn is on me.",
 "Deal. Do you prefer sweet or salty popcorn?",
 "Salty, with a little butter.",
 "Same here. Classic is best.",
 "Agreed. Do you want to watch at your place or mine?",
 "Let’s do mine. I just got a new sound system.",
 "Nice! I can’t wait to hear it.",
 "It makes a big difference, especially for action scenes.",
 "I bet. So, hiking in the morning, movie night in the evening?",
 "Sounds like a perfect day.",
 "I’m looking forward to it.",
 "Me too. It’s been a while since we hung out like this.",
 "Yeah, life gets busy. It’s good to slow down sometimes.",
 "Definitely. We should make it a regular thing.",
 "I’d like that. Maybe once a month?",
 "Let’s do it. We can take turns planning.",
 "Great idea. Keeps things interesting.",
 "Exactly. So,
 what’s your favorite way to relax?",
 "Honestly, just sitting outside with a cup of coffee.",
 "That sounds nice. Do you have a favorite spot?",
 "There’s a little park near my apartment. It’s quiet and peaceful.",
 "I’ll have to check it out sometime.",
 "You should. It’s a hidden gem.",
 "I love finding places like that.",
 "Me too. It’s important to have a place to unwind.",
 "Absolutely. Especially with how hectic life can be.",
 "Yeah. Sometimes I feel like I’m always rushing.",
 "Same here. That’s why I try to make time for myself.",
 "It’s important. Self-care isn’t selfish.",
 "Agreed. What do you do to take care of yourself?",
 "I like to cook. Trying new recipes is relaxing for me.",
 "That’s awesome. What’s your favorite dish to make?",
 "Probably homemade pasta. It’s a bit of work, but so worth it.",
 "I’ve never made pasta from scratch. Is it hard?",
 "Not really, just takes some patience.",
 "Maybe you can teach me sometime.",
 "I’d love to. We can have a cooking night.",
 "That sounds fun. I’ll bring dessert.",
 "Deal. What’s your specialty?",
 "Chocolate cake. It’s my grandmother’s recipe.",
 "Now I’m hungry.",
 "Me too. Maybe we should grab dinner.",
 "Good idea. Where should we go?",
 "How about that little Italian place on Main Street?",
 "Perfect. I’ve been craving pizza.",
 "Me too. Let’s go.",
 "I’ll drive.",
 "Thanks. I’ll get the door.",
 "After you.",
 "Thank you. So, what’s your favorite pizza topping?",
 "Mushrooms and olives. You?",
 "Pepperoni and extra cheese.",
 "Classic choice. Can’t go wrong.",
 "Exactly. Do you want to share a pizza?",
 "Sure. Half and half?",
 "Sounds good to me.",
 "Great. I’ll let the waiter know.",
 "Thanks. So, tell me more about your new project at work.",
 "Well, it’s still in the early stages, but we’re brainstorming ideas.",
 "That’s exciting. Do you have any concepts in mind?",
 "A few. We want to focus on user experience.",
 "That’s important. People appreciate intuitive design.",
 "Exactly. We’re thinking of doing some user testing.",
 "Smart move. Feedback is valuable.",
 "Definitely. I’ll keep you posted as things develop.",
 "Please do. I’m curious to see how it turns out.",
 "Thanks for your support.",
 "Always. That’s what friends are for.",
 "True. I appreciate it.",
 "Anytime. So, what’s one thing you want to do this year?",
 "Travel more. I haven’t been anywhere new in a while.",
 "Where would you go if you could pick anywhere?",
 "Japan. I’ve always wanted to see the cherry blossoms.",
 "That would be amazing. I hope you get to go.",
 "Me too. What about you?",
 "I’d like to learn a new language.",
 "Which one?",
 "Spanish, maybe. It’s useful and beautiful.",
 "Good choice. There are lots of resources online.",
 "Yeah, I’ve started using an app.",
 "How’s it going?",
 "Slow but steady. Practice makes perfect.",
 "Exactly. Maybe we can practice together.",
 "That would be great. Two heads are better than one.",
 "Agreed. Let’s set a goal.",
 "How about learning a new phrase each week?",
 "Perfect. We’ll quiz each other.",
 "Deal. This will be fun.",
 "I’m looking forward to it.",
 "Me too. It’s good to challenge ourselves.",
 "Absolutely. Keeps the mind sharp.",
 "So, what’s your favorite way to spend a Sunday?",
 "Sleeping in, then brunch with friends.",
 "That sounds perfect.",
 "It is. How about you?",
 "I like to go for a run, then read.",
 "Nice. Do you run often?",
 "A few times a week. It helps me clear my head.",
 "That’s great. I wish I had your motivation.",
 "You can join me anytime.",
 "Maybe I will. I could use the exercise.",
 "It’s more fun with a friend.",
 "Agreed. Let’s plan for next Sunday.",
 "Sounds good. I’ll remind you.",
 "Thanks. I’ll need it.",
 "No problem. So, what’s one thing you’re grateful for today?",
 "This conversation.",
 "Me too. It’s nice to connect.",
 "Absolutely. Thanks for being here.",
 "Always."
]
```
