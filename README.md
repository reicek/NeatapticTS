# NeatapticTS

<div class="badges">
	<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2c3963.svg" alt="License: MIT"/></a>
	<a href="docs/index.html"><img src="https://img.shields.io/badge/docs-generated-2c3963.svg" alt="Docs"/></a>
	<a href="https://www.npmjs.com/package/@reicek/neataptic-ts"><img src="https://img.shields.io/npm/v/%40reicek%2Fneataptic-ts?color=2c3963&label=npm" alt="npm version"/></a>
	<a href="package.json"><img src="https://img.shields.io/badge/node-22%2B-2c3963.svg" alt="Node 22+"/></a>
	<a href="https://github.com/reicek/NeatapticTS/actions/workflows/ci.yml"><img src="https://github.com/reicek/NeatapticTS/actions/workflows/ci.yml/badge.svg?branch=develop" alt="CI status"/></a>
	<a href="https://github.com/reicek/NeatapticTS/actions/workflows/deploy-pages.yml"><img src="https://github.com/reicek/NeatapticTS/actions/workflows/deploy-pages.yml/badge.svg?branch=develop" alt="Docs deploy status"/></a>
</div>

<img src="nn.jpg" width="784" alt="NeatapticTS cover art"/>

_Cover art for the NeatapticTS repository._

> A modern TypeScript NEAT library built to be read, tested, and extended.

NeatapticTS implements **NeuroEvolution of Augmenting Topologies (NEAT)** — an algorithm that discovers neural-network structure by evolution rather than by hand-design. Unlike gradient-based training, which assumes a fixed architecture and adjusts weights, NEAT simultaneously searches the _space of topologies_ and _the space of weights_. Connections grow, nodes split, and entire species of structurally different solutions compete in the same population until a solution emerges that could not have been specified in advance.

This is not a black-box experiment runner. The library is built to be inspected, extended, and learned from: typed primitives, deterministic seeds, rich telemetry, and documentation that explains the _why_ at every level — not just the API surface.

---

## The Key Idea: Evolving Topology

Most neural-network training starts by choosing an architecture — how many layers, how many neurons, what connections. NEAT removes that assumption. It starts with minimal networks and uses three coordinated mechanisms to search the combined space of weights _and_ structure:

**1. Historical markings** — Every structural innovation (a new node or connection) receives a globally unique _innovation number_. When two genomes from different structural lineages are crossed over, historical markings allow meaningful alignment: genes with matching innovation numbers describe the same structural feature, even if the genomes grew their shared ancestry by different mutation paths.

**2. Speciation** — New topology is immediately at a disadvantage against well-tuned incumbents. NEAT groups genomes into _species_ using a compatibility distance function, and fitness sharing ensures each species competes primarily against its own members rather than across topological families. This gives new structures the generations they need to prove useful before they are eliminated.

**3. Minimal complexification** — NEAT begins every run from the simplest possible network (inputs directly to outputs) and adds structure only when mutations suggest it. This keeps search in the tractable part of the topology space for as long as possible. See Stanley and Miikkulainen, [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02), for the original paper that established the algorithm.

### Compatibility Distance

The speciation boundary between any two genomes is determined by measuring how structurally different they are. If two genomes share most innovation numbers, they belong in the same species. If many innovations are disjoint or excess, they likely represent different evolutionary lineages.

The compatibility distance δ between two genomes is:

```
δ = (c₁ · E) / N  +  (c₂ · D) / N  +  c₃ · W̄
```

where **E** = excess gene count, **D** = disjoint gene count, **N** = larger genome length (normalizes for size), **W̄** = mean weight difference of matching genes, and **c₁, c₂, c₃** are coefficients that tune the relative importance of each term. Two genomes belong to the same species when δ < threshold. See Stanley and Miikkulainen, [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02), for the original derivation of this formula and the NEAT mechanics.

### The Evolutionary Loop

Each generation of a NEAT run follows the same five-step cycle:

```mermaid
flowchart LR
    subgraph Generation["One Generation"]
        direction LR
        Evaluate["Evaluate\nScore each genome\nagainst the task"]
        Speciate["Speciate\nGroup by compatibility\ndistance δ"]
        Select["Select\nFitness sharing\nwithin species"]
        Reproduce["Reproduce\nCrossover + mutation\npreserving innovations"]
        Grow["Grow\nAdd nodes/connections\nvia structural mutations"]
    end

    Evaluate --> Speciate --> Select --> Reproduce --> Grow --> Evaluate

    classDef base fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:2px;
    classDef accent fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;

    class Evaluate,Speciate,Select,Reproduce,Grow base;
```

---

## Why This Library

Many neuroevolution libraries are either convenient but opaque, or educational but too small to trust as a real reference. NeatapticTS is built to close that gap.

The project gives you:

- **Readable internals** — orchestration-first module structure, each boundary explained by its own chapter README.
- **Deterministic reproducibility** — seeded RNG, exportable state, replayable runs.
- **Modern TypeScript** — ES2023+ ergonomics, full type coverage, no runtime `any`.
- **Rich telemetry** — per-generation diversity, species history, Pareto fronts, novelty tracking.
- **Worker-backed evaluation** — parallel genome scoring for Node and browser environments.
- **ONNX export** — trained networks portable to ONNX-compatible inference runtimes.
- **NGE core** — seed-to-scale neuroevolution with canonical DNA envelopes, polyandric reproduction, and continuous juvenile growth validated beyond 8,000 neurons; deterministic checkpoints make growth replayable.
- **WebGPU inference** — opt-in GPU forward pass for eligible networks via `network.activate(input, { useGPU: true })`; automatic CPU fallback keeps classic NEAT behavior unchanged. See [WebGPU.md](./WebGPU.md).
- **Educational examples** — three full examples (Flappy Bird, ASCII Maze, Racing Curriculum) that teach architecture choices rather than hiding them.

---

## System Architecture

The library is organized into four cooperating layers. Reading them in order builds a complete picture from graph primitives up to the evolutionary controller:

```mermaid
flowchart TD
    subgraph Public["Public API  ·  src/neataptic.ts"]
        Neat["Neat\nevolutionary controller"]
        Network["Network\ngraph orchestration"]
        Methods["methods\nactivation · cost · selection · mutation"]
    end

    subgraph Architecture["src/architecture/  ·  Graph primitives"]
        Node["Node\nneuron with activation state"]
        Connection["Connection\nweighted directed edge"]
        Layer["Layer / Group\nstructured neuron sets"]
        NetworkImpl["Network internals\nactivate · train · serialize · mutate · ONNX"]
    end

    subgraph NEAT["src/neat/  ·  Evolutionary controller"]
        Init["init/\npopulation setup"]
        Evaluate["evaluate/\nscoring and objectives"]
        Evolve["evolve/\nselection and offspring"]
        Speciation["speciation/\ncompatibility grouping"]
        Mutation["mutation/\nstructural operators"]
        Telemetry["telemetry/\ndiversity and lineage"]
        RNG["rng/  export/  cache/\nreproducibility"]
    end

    subgraph Workers["src/multithreading/  ·  Parallel evaluation"]
        BrowserWorker["Browser worker"]
        NodeWorker["Node worker"]
    end

    Neat --> Init
    Neat --> Evaluate
    Neat --> Evolve
    Neat --> Speciation
    Neat --> Mutation
    Neat --> Telemetry
    Neat --> RNG
    Network --> NetworkImpl
    NetworkImpl --> Node
    NetworkImpl --> Connection
    NetworkImpl --> Layer
    Evaluate --> Workers

    classDef pub fill:#001522,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
    classDef arch fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
    classDef neat fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
    classDef work fill:#001522,stroke:#ff9a2e,color:#ffe6cc,stroke-width:1.5px;

    class Neat,Network,Methods pub;
    class Node,Connection,Layer,NetworkImpl arch;
    class Init,Evaluate,Evolve,Speciation,Mutation,Telemetry,RNG neat;
    class BrowserWorker,NodeWorker work;
```

---

## Start Here

| Goal                                               | Best place to start                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| Read the library architecture from the source side | [src/README.md](./src/README.md)                                          |
| Study the strongest end-to-end example             | [examples/flappy_bird/README.md](./examples/flappy_bird/README.md)        |
| Study curriculum learning and reward shaping       | [examples/asciiMaze/README.md](./examples/asciiMaze/README.md)            |
| Browse runnable example source directly            | [examples](./examples)                                                    |
| Review contribution standards                      | [CONTRIBUTING.md](./CONTRIBUTING.md) and [STYLEGUIDE.md](./STYLEGUIDE.md) |

## Reading Paths

### If you are new to NEAT or neuroevolution

1. Read the background section above — especially the three key mechanisms and the compatibility distance formula.
2. Open [src/neat/README.md](./src/neat/README.md) for the controller's public defaults and four-lane architecture.
3. Skim [examples/flappy_bird/README.md](./examples/flappy_bird/README.md) to see those concepts applied in a complete system.

### If you are new to this repo

1. Read [docs/index.html](./docs/index.html).
2. Read [src/README.md](./src/README.md).
3. Open one example:
   - [examples/flappy_bird/README.md](./examples/flappy_bird/README.md)
   - [examples/asciiMaze/README.md](./examples/asciiMaze/README.md)

### If you want library internals

- [src/neat/README.md](./src/neat/README.md) — the evolutionary controller
- [src/architecture/network/README.md](./src/architecture/network/README.md) — the network graph
- [src/architecture/network/onnx/README.md](./src/architecture/network/onnx/README.md) — ONNX portability
- [src/multithreading/README.md](./src/multithreading/README.md) — parallel evaluation

---

## Examples Worth Opening First

### Flappy Bird

[examples/flappy_bird](./examples/flappy_bird) is the best single example if you want to see how NeatapticTS feels in a real project. It combines deterministic environment stepping, evaluation designed to reduce lucky-rollout bias, worker-backed browser playback, live network inspection, and a modular architecture with explicit boundaries.

### ASCII Maze

[examples/asciiMaze](./examples/asciiMaze) is the companion example for studying curriculum progression, compact observations, reward shaping for sparse goals, and browser plus terminal visualization.

---

## Install

Runtime requirement: Node 22+.

```bash
npm install @reicek/neataptic-ts
```

Minimal example — evolve a network toward a target output in one generation loop:

```ts
import { Neat } from '@reicek/neataptic-ts';

const fitness = (network) => {
	const output = network.activate([1])[0];
	return -(output - 2) ** 2;
};

const neat = new Neat(1, 1, fitness, {
	popsize: 30,
	seed: 42,
	fastMode: true,
});

await neat.evaluate();
await neat.evolve();

console.log(neat.getBest()?.score);
```

For options, telemetry, multiobjective search, ONNX export, and subsystem details, continue in [docs/index.html](./docs/index.html) or [src/README.md](./src/README.md).

---

## Repo Map

| Path                                   | Purpose                                                           |
| -------------------------------------- | ----------------------------------------------------------------- |
| [src](./src)                           | Core library code and generated module docs                       |
| [examples](./examples)                 | Educational examples and demos                                    |
| [docs](./docs)                         | Generated documentation site and example assets                   |
| [scripts](./scripts)                   | Build and docs tooling, semantic index, and Cortex RAG MCP server |
| [rag_architecture](./rag_architecture) | Repo Cortex RAG architecture reference                            |
| [plans](./plans)                       | Architecture and planning material                              |

## Repo Cortex RAG

NeatapticTS ships a semantic retrieval layer — the **Repo Cortex** — that lets agents search the codebase by meaning rather than by substring. It is the primary search mechanism for the project's agents and skills: `grep`, `glob`, and `view` are fallbacks of last resort, not the routine path. The Cortex-First Search Policy in `copilot-instructions.md` codifies this preference.

The Cortex is exposed as a local `stdio` MCP server (`cortex`) with 18 read-only tools: `search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `load_parent_chunk`, `load_document`, `freshness_check`, `index_stats`, `ann_build_index`, `list_families`, `scan_code_quality`, `traverse_graph`, `expand_query`, `submit_feedback`, `parallel_search`, `multi_hop_search`, `turso_branch`, and `turso_pitr`. The `cortex` server is a lightweight lazy-load facade that spawns the real Repo Cortex server only when a tool is actually called.

Retrieval combines BM25 full-text search (FTS5) with dense vector search and fuses the two streams server-side via **Reciprocal Rank Fusion (RRF, k=60)** rather than a fixed alpha blend. Dense vectors are 8-bit quantized (`F8_BLOB`) and served through Turso's native DiskANN index (`vector_top_k`), so no embeddings are loaded into JavaScript memory at query time. A cross-encoder reranker and a query-expansion layer refine results further, and `search_context` assembles a token-bounded context window ready to hand to a language model.

The backing database is **Turso** (libSQL). See the next section for setup.

## Turso Setup

The Repo Cortex is backed by a Turso (libSQL) database. The MCP server connects through the async `@libsql/client` driver — there is no synchronous SQLite binding and no event-loop blocking.

### Environment variables

| Variable              | Purpose                                                                                                            | Example                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `TURSO_DATABASE_URL`  | Primary database URL. A `file:` URL uses a local embedded replica; a `libsql:` URL targets a cloud primary.        | `file:./rag-index/data/turso-replica.sqlite` or `libsql://my-db.turso.io` |
| `TURSO_AUTH_TOKEN`    | JWT auth token for cloud access. Optional for local `file:` URLs.                                                  | _keep secret_                                                             |
| `TURSO_SYNC_URL`      | Remote sync URL for embedded-replica mode. When set, the local `file:` database syncs from a remote Turso primary. | `libsql://my-db.turso.io`                                                 |
| `TURSO_SYNC_INTERVAL` | Embedded-replica sync interval, in seconds.                                                                        | `60`                                                                      |
| `TURSO_CONCURRENCY`   | Maximum in-flight queries for `parallel_search`.                                                                   | `20`                                                                      |

The MCP registrations in `.mcp.json` (Copilot CLI) and `.vscode/mcp.json` (VS Code) both supply the default env values for local development. They are kept in sync so the same six servers are available in either client:

```json
{
  "env": {
    "TURSO_DATABASE_URL": "file:./rag-index/data/turso-replica.sqlite",
    "TURSO_AUTH_TOKEN": "",
    "TURSO_SYNC_URL": "",
    "TURSO_SYNC_INTERVAL": "60"
  }
}
```

### Embedded replica config

For local development, set `TURSO_DATABASE_URL` to a `file:` path (default `file:./rag-index/data/turso-replica.sqlite`). This runs an **embedded replica** — a local libSQL database file that can operate standalone or sync from a remote Turso primary when `TURSO_SYNC_URL` is also set. Reads are served locally (sub-millisecond), and read-your-writes semantics apply. Run `node rag-index/build-index.mjs` to populate the corpus, then `node rag-index/embed-index.mjs` to generate and store `F8_BLOB` vectors.

For cloud-only operation, set `TURSO_DATABASE_URL` to a `libsql:` URL and provide `TURSO_AUTH_TOKEN`. All queries hit the cloud primary directly.

### Fallback behavior when Turso is unreachable

- **Embedded replica configured (`file:` URL):** local reads continue to be served from the replica file even if the remote primary (`TURSO_SYNC_URL`) is unreachable. Stale reads are possible until connectivity is restored and the replica syncs again. Writes that require the primary will fail until connectivity returns.
- **Cloud-only (`libsql:` URL, no replica):** if the cloud primary is unreachable, search and load calls return an error. There is no silent local fallback in this mode — configure an embedded replica if resilient local reads are required.

Schema versioning uses a `_schema_version` table rather than `PRAGMA user_version`, because Turso treats PRAGMA as read-only.

## Contributing

This repo treats documentation as part of the product. If you change behavior, examples, or public API shape, update the documentation surface that teaches that boundary.

Primary contribution entry points:

- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [STYLEGUIDE.md](./STYLEGUIDE.md)
- [RELEASE.md](./RELEASE.md)

## License

MIT.
