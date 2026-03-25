/**
 * Architecture root chapter map and compatibility surface.
 *
 * This folder gathers the core runtime building blocks that most NeatapticTS
 * networks are composed from: `Network` orchestration, `Node` and
 * `Connection` graph primitives, `Group` and `Layer` composition helpers,
 * `Architect` presets, and the shared allocation pools that keep hot paths
 * leaner during training and inference.
 *
 * How to read this chapter:
 * - Start here for the high-level map of the architecture package.
 * - Continue into `network/` for the main orchestration surface and its
 *   runtime, training, mutation, serialization, and ONNX subchapters.
 * - Continue into `node/`, `connection/`, `group/`, and `layer/` for the graph
 *   building blocks.
 * - Continue into `architect/` for preset builders and into `nodePool/` and
 *   `activationArrayPool/` for allocation-oriented helpers.
 * - Treat the flat `src/architecture/*.ts` files as compatibility facades that
 *   keep public imports stable while the implementation lives in narrower,
 *   teachable chapters.
 */
export { default } from './network/network';