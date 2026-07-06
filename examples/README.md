# Examples

This folder is a learning path, not a flat pile of demos.

Start with the smallest examples that show one idea clearly. Move to the browser-hosted starter pages once the core API makes sense. Then use the flagship demos to study what happens when the same library has to own runtime boundaries, fairness rules, shaping strategies, and inspectable browser surfaces.

The point of this folder is not just to prove that NeatapticTS can evolve agents. It is to show how different problem shapes change what the network should observe, how the training loop should be framed, and how much infrastructure a serious example needs around the controller.

## Recommended learning path

| Step | Start here                                                                                           | What it teaches                                 | Why it comes now                                                                                        |
| ---- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1    | [helloNetwork](./helloNetwork)                                                                       | One tiny forward pass through the public API    | It gives you a concrete output immediately, without evolution or browser setup.                         |
| 2    | [evolveXor](./evolveXor)                                                                             | The smallest useful NEAT loop                   | It adds selection, mutation, and generations without the extra complexity of a large environment.       |
| 3    | [sequenceReset](./sequenceReset)                                                                     | Recurrent state and `network.clear()` semantics | It shows the first stateful behavior boundary before you move to bigger runtime systems.                |
| 4    | [Starter browser pages](../docs/examples/index.html)                                                 | Lightweight browser-hosted walkthroughs         | It lets you stay on the same starter concepts while seeing the docs-published browser path.             |
| 5    | [flappy_bird](./flappy_bird), [asciiMaze](./asciiMaze), and [racing_curriculum](./racing_curriculum) | Full-system neuroevolution demos                | They are best read after the starter path, once the controller and runtime basics are already familiar. |

If you want the shortest useful route through the folder, follow that order exactly: inference first, minimal evolution second, sequence state third, browser quickstart fourth, then the flagship systems.

## Starter examples first

The starter tranche is intentionally small. Each example isolates one learning step and stays readable enough to rerun, inspect, and modify without first understanding a whole application.

### 1. Hello Network

[helloNetwork](./helloNetwork) is the first stop.

Use it to learn the public API shape, the input and output conventions, and what a tiny deterministic inference pass looks like when there is no training loop around it yet.

Best starting points:

- [helloNetwork/README.md](./helloNetwork/README.md)
- [helloNetwork/index.ts](./helloNetwork/index.ts)
- [helloNetwork/index.html](./helloNetwork/index.html)

### 2. Evolve XOR

[evolveXor](./evolveXor) is the first evolution checkpoint.

It keeps the task small enough that the evolutionary loop is still the main lesson. You can see population setup, evaluation, mutation, and a bounded solved outcome without the noise of a large simulation.

Best starting points:

- [evolveXor/README.md](./evolveXor/README.md)
- [evolveXor/index.ts](./evolveXor/index.ts)
- [evolveXor/index.html](./evolveXor/index.html)

### 3. Sequence Reset

[sequenceReset](./sequenceReset) is the first stateful example.

Read it when you want to understand the difference between a fresh recurrent run, a cleared recurrent run, and a carried-over recurrent run. It is the bridge between tiny feed-forward walkthroughs and the larger examples that depend on careful runtime-state boundaries.

Best starting points:

- [sequenceReset/README.md](./sequenceReset/README.md)
- [sequenceReset/index.ts](./sequenceReset/index.ts)
- [sequenceReset/index.html](./sequenceReset/index.html)

## Browser quickstart

Once the three starter examples make sense in source form, open the browser-hosted versions through [docs/examples/index.html](../docs/examples/index.html).

That page groups the starter examples separately from the flagship demos, so you can stay on the lightweight learning path before stepping into the larger systems. The starter browser pages are intentionally small: they render concrete results, reuse the shared docs publication flow, and avoid the heavier replay and control-panel surfaces used by the flagship demos.

Important browser note:

- the `index.html` files under each example are lightweight shells that load prebuilt assets from `docs/assets`.
- when you want the real browser implementation boundary, read the browser entry module in the example folder rather than only the HTML shell.
- when you change browser-facing example code and want the published starter pages to reflect it, run `npm run docs` or the relevant example build script.

## Flagship demos second

After the starter path, move to the larger examples that show NeatapticTS under more realistic system pressure.

### Flappy Bird

[flappy_bird](./flappy_bird) is the best next step if you want a browser-heavy control problem.

It shows how deterministic stepping, shared-seed evaluation, feed-forward local memory, worker-backed playback, and live network inspection fit together in one system. The lesson is not only how to evolve a bird controller. The lesson is how to keep a fast browser demo fair, inspectable, and architecturally readable.

Best starting points:

- [flappy_bird/README.md](./flappy_bird/README.md)
- [flappy_bird/trainFlappyBird.ts](./flappy_bird/trainFlappyBird.ts)
- [flappy_bird/index.html](./flappy_bird/index.html)

### ASCII Maze

[asciiMaze](./asciiMaze) is the best next step if you want a tighter observation budget and a more explicit shaping story.

It compresses the environment into a compact state summary, then makes the interesting work happen in navigation pressure, reward shaping, curriculum transfer, telemetry, and controller-level search overlays.

Best starting points:

- [asciiMaze/README.md](./asciiMaze/README.md)
- [asciiMaze/evolutionEngine.ts](./asciiMaze/evolutionEngine.ts)
- [asciiMaze/index.html](./asciiMaze/index.html)

### Racing Curriculum

[racing_curriculum](./racing_curriculum) is the best next step if you want to study coevolution, host/worker authority, and deterministic packed-snapshot streaming.

It runs two independent NEAT populations (Team A and Team B) against frozen opponent snapshots, keeps all simulation truth inside a dedicated worker, and streams compact typed-array race-step frames back to the host. The lesson is how to separate rendering authority from evolution authority and how to keep a competitive coevolution benchmark fair, replayable, and easy to extend with new neuromodulation, plasticity, or controller contracts.

Best starting points:

- [racing_curriculum/docs/coevolution-contract.md](https://github.com/reicek/NeatapticTS/blob/main/examples/racing_curriculum/docs/coevolution-contract.md)
- [racing_curriculum/workers/simulation-worker/README.md](./racing_curriculum/workers/simulation-worker/README.md)
- [racing_curriculum/browser-entry/browser-entry.ts](./racing_curriculum/browser-entry/browser-entry.ts)
- [racing_curriculum/index.html](./racing_curriculum/index.html)

## How the flagship demos differ

| Dimension         | Flappy Bird                                                                     | ASCII Maze                                                                              | Racing Curriculum                                                                    |
| ----------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Core challenge    | Reflex control under changing geometry                                          | Deliberate navigation toward a goal                                                     | Competitive coevolution across two teams                                             |
| Observation style | Broad temporal observation                                                      | Tight handcrafted state summary                                                         | Continuous sensor vector from track state                                            |
| Policy outputs    | 2 action scores                                                                 | 4 directional scores                                                                    | Car-control vector (steer, throttle, brake)                                          |
| Teaching emphasis | Evaluation fairness, feed-forward local memory, worker playback, inspectable UI | Reward shaping, curriculum transfer, telemetry-rich evolution, explicit search overlays | Host/worker authority, frozen-snapshot coevolution, zero-copy transfer, protocol FSM |
| Runtime flavor    | Browser-heavy and replay-oriented                                               | Console-browser hybrid and experiment-oriented                                          | Worker-authoritative simulation with host rendering                                  |

The quickest mental shortcut is simple:

- Flappy Bird is a fast control-systems lesson.
- ASCII Maze is a compact decision-making lesson.
- Racing Curriculum is a coevolution and runtime-authority lesson.

Together they show that a good neuroevolution example is not one fixed template. Observation design, scoring design, runtime ownership, and visualization strategy all change with the problem.

## Advanced follow-up

[`neatChat`](./neatChat) belongs after the starter path and the three existing flagship demos, but its browser-hosted page should publish alongside the flagship section so users can inspect progress without depending on Node.

Treat it as an advanced learnability follow-up for when you already understand the small examples, the browser publication flow, and the three larger system demos.

Best starting points:

- [neatChat/README.md](./neatChat/README.md)
- [neatChat/index.ts](./neatChat/index.ts)
- [neatChat/index.html](./neatChat/index.html)
- [neatChat/run.ts](./neatChat/run.ts)

## Why this folder matters

Libraries are easy to admire in abstraction. They are harder to trust until you see them under pressure.

This folder is where NeatapticTS stops being a list of features and starts behaving like a toolkit with opinions. Read in order, the examples move from tiny deterministic building blocks to larger runtime systems without asking a new reader to begin at the most complex end of the repo.
