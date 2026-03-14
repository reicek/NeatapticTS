# NeatapticTS

<div class="badges">
	<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2c3963.svg" alt="License: MIT"/></a>
	<a href="docs/index.html"><img src="https://img.shields.io/badge/docs-generated-2c3963.svg" alt="Docs"/></a>
	<a href="https://www.npmjs.com/package/@reicek/neataptic-ts"><img src="https://img.shields.io/npm/v/%40reicek%2Fneataptic-ts?color=2c3963&label=npm" alt="npm version"/></a>
	<a href="package.json"><img src="https://img.shields.io/badge/node-22%2B-2c3963.svg" alt="Node 22+"/></a>
	<a href="https://github.com/reicek/NeatapticTS/actions/workflows/ci.yml"><img src="https://github.com/reicek/NeatapticTS/actions/workflows/ci.yml/badge.svg?branch=develop" alt="CI status"/></a>
	<a href="https://github.com/reicek/NeatapticTS/actions/workflows/deploy-pages.yml"><img src="https://github.com/reicek/NeatapticTS/actions/workflows/deploy-pages.yml/badge.svg?branch=develop" alt="Docs deploy status"/></a>
</div>

<img src="nn.jpg" width="784"/>

> A modern TypeScript NEAT library built to be read, tested, and extended.

NeatapticTS is a documentation-first evolution of Neataptic for people who want more than a black-box experiment runner. It combines typed neural-network primitives, NEAT-style topology evolution, reproducible runs, and educational examples that explain the architecture instead of hiding it.

This page is intentionally short. It is the entry map. The detailed API and subsystem documentation live in the generated docs and the focused READMEs throughout the repo.

## Why this exists

Many neuroevolution libraries are either convenient but opaque, or educational but too small to trust as a real reference. NeatapticTS is built to close that gap.

The project aims to give you:

- readable internals instead of magic,
- deterministic seeds and telemetry for reproducible experiments,
- modern TypeScript and ES2023+ ergonomics,
- examples that behave like reference systems rather than toy snippets.

If you want a library you can inspect, modify, and learn from while still running serious experiments, this is the point of the repo.

## Start here

| Goal | Best place to start |
| --- | --- |
| Read the architecture from the source side | [src/README.md](./src/README.md) |
| Study the strongest end-to-end example | [test/examples/flappy_bird/README.md](./test/examples/flappy_bird/README.md) |
| Study curriculum learning and reward shaping | [test/examples/asciiMaze/README.md](./test/examples/asciiMaze/README.md) |
| Browse runnable example source directly | [test/examples](./test/examples) |
| Review contribution standards | [CONTRIBUTING.md](./CONTRIBUTING.md) and [STYLEGUIDE.md](./STYLEGUIDE.md) |

## Reading paths

### If you are new to the repo

1. Read [docs/index.html](./docs/index.html).
2. Read [src/README.md](./src/README.md).
3. Open one example:
	 - [test/examples/flappy_bird/README.md](./test/examples/flappy_bird/README.md)
	 - [test/examples/asciiMaze/README.md](./test/examples/asciiMaze/README.md)

### If you want runnable source first

- [test/examples/flappy_bird](./test/examples/flappy_bird) for the clearest full-system example.
- [test/examples/asciiMaze](./test/examples/asciiMaze) for evolution orchestration, telemetry, and visualization.

### If you want library internals

- [src/neat/README.md](./src/neat/README.md)
- [src/architecture/network/README.md](./src/architecture/network/README.md)
- [src/architecture/network/onnx/README.md](./src/architecture/network/onnx/README.md)
- [src/multithreading/README.md](./src/multithreading/README.md)

## Examples worth opening first

### Flappy Bird

[test/examples/flappy_bird](./test/examples/flappy_bird) is the best single example if you want to understand how NeatapticTS feels in a real project.

It combines:

- deterministic environment stepping,
- evaluation designed to reduce lucky-rollout bias,
- worker-backed browser playback,
- live network inspection,
- a modular architecture with explicit boundaries.

### ASCII Maze

[test/examples/asciiMaze](./test/examples/asciiMaze) is the best companion example if you want to study curriculum progression, compact observations, reward shaping, and browser plus terminal visualization.

## Install

Runtime requirement: Node 22+.

```bash
npm install @reicek/neataptic-ts
```

Minimal example:

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

For options, telemetry, and subsystem details, continue in [docs/index.html](./docs/index.html).

## Repo map

| Path | Purpose |
| --- | --- |
| [src](./src) | Core library code and generated module docs |
| [test/examples](./test/examples) | Educational examples and demos |
| [docs](./docs) | Generated documentation site and example assets |
| [scripts](./scripts) | Build and docs tooling |
| [plans](./plans) | Architecture and roadmap material |

## Contributing

This repo treats documentation as part of the product. If you change behavior, examples, or public API shape, update the documentation surface that teaches that boundary.

Primary contribution entry points:

- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [STYLEGUIDE.md](./STYLEGUIDE.md)
- [srcLint.md](./srcLint.md)
- [RELEASE.md](./RELEASE.md)

## License

MIT.
