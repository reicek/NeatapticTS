### This library is being updated to TypeScrypt

<img src="https://cdn-images-1.medium.com/max/800/1*THG2__H9YHxYIt2sulzlTw.png" width="100%"/>

# NeatapticTS

NeatapticTS offers flexible neural networks; neurons and synapses can be removed with a single line of code. No fixed architecture is required for neural networks to function at all. This flexibility allows networks to be shaped for your dataset through neuro-evolution, which is done using multiple threads.

# Network Constructor Update

The `Network` class constructor now supports an optional third parameter for configuration:

```ts
new Network(input: number, output: number, options?: { minHidden?: number })
```
- `input`: Number of input nodes (required)
- `output`: Number of output nodes (required)
- `options.minHidden`: (optional) If set, enforces a minimum number of hidden nodes. If omitted or 0, no minimum is enforced. This allows true 1-1 (input-output only) networks.

**Example:**
```ts
// Standard 1-1 network (no hidden nodes)
const net = new Network(1, 1);

// Enforce at least 3 hidden nodes
const netWithHidden = new Network(2, 1, { minHidden: 3 });
```

# Neat Evolution minHidden Option

The `minHidden` option can also be passed to the `Neat` class to enforce a minimum number of hidden nodes in all evolved networks:

```ts
import Neat from './src/neat';
const neat = new Neat(2, 1, fitnessFn, { popsize: 50, minHidden: 5 });
```
- All networks created by the evolutionary process will have at least 5 hidden nodes.
- This is useful for ensuring a minimum network complexity during neuro-evolution.

See tests in `test/neat.ts` for usage and verification.

---

# ONNX Import/Export

NeatapticTS now supports exporting to and importing from ONNX format for strictly layered MLPs. This allows interoperability with other machine learning frameworks.

- Use `exportToONNX(network)` to export a network to ONNX.
- Use `importFromONNX(onnxModel)` to import a compatible ONNX model as a `Network` instance.

See tests in `test/network/onnx.export.test.ts` and `test/network/onnx.import.test.ts` for usage examples.

---

### Further notices

NeatapticTS is based on [Neataptic](https://github.com/wagenaartje/neataptic). Parts of [Synaptic](https://github.com/cazala/synaptic) where used to develop Neataptic.

The neuro-evolution algorithm used is the [Instinct](https://medium.com/@ThomasWagenaar/neuro-evolution-on-steroids-82bd14ddc2f6) algorithm.

##### Original [repository](https://github.com/wagenaartje/neataptic) in now [unmaintained](https://github.com/wagenaartje/neataptic/issues/112)
