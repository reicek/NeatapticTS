/**
 * Import-owned types for ONNX runtime factory loading.
 *
 * These payloads describe the small runtime bootstrap contract that the import
 * flow uses to rebuild a perceptron scaffold and attach recurrent layer
 * constructors without making the parser own a hard-coded constructor shape.
 *
 * Example:
 * ```ts
 * const runtimeFactories: OnnxRuntimeFactories = {
 *   perceptronFactory,
 *   layerModule,
 * };
 * ```
 */

import type Layer from '../../../layer/layer';
import type Network from '../../network';

/** Runtime perceptron factory signature used by ONNX import orchestration. */
export type OnnxRuntimePerceptronFactory = (...sizes: number[]) => Network;

/** Runtime layer-constructor signature used for recurrent layer reconstruction. */
export type OnnxRuntimeLayerFactory = (size: number) => Layer;

/** Runtime layer module shape consumed by ONNX import orchestration. */
export type OnnxRuntimeLayerModule = {
  lstm: OnnxRuntimeLayerFactory;
  gru: OnnxRuntimeLayerFactory;
};

/** Runtime factories consumed during ONNX import network reconstruction. */
export type OnnxRuntimeFactories = {
  perceptronFactory: OnnxRuntimePerceptronFactory;
  layerModule: OnnxRuntimeLayerModule;
};

/** Validation context for perceptron size-list checks during ONNX import. */
export type OnnxPerceptronSizeValidationContext = {
  sizes: number[];
  minimumSizeCount: number;
  errorMessage: string;
};

/** Build context for mapping ONNX layer sizes into a Neataptic MLP factory call. */
export type OnnxPerceptronBuildContext = {
  sizes: number[];
  inputIndex: number;
  hiddenSliceStartIndex: number;
  hiddenSliceEndOffset: number;
  outputFallbackCount: number;
};
