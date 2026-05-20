/**
 * Decision router for ONNX layer emission.
 *
 * This boundary does not build tensors directly. Its job is to inspect one
 * export layer and choose the smallest valid emission strategy:
 * Conv when an explicit mapping is present, recurrent single-step when the
 * layer owns self-connections, compact dense export when activations are
 * homogeneous, or per-neuron decomposition when activations differ.
 *
 * ```mermaid
 * flowchart TD
 *   Start[Layer inputs] --> Conv{Conv mapping for layer?}
 *   Conv -->|Yes| ConvEmit[Emit Conv path]
 *   Conv -->|No| Recurrent{Self-recurrent hidden layer?}
 *   Recurrent -->|Yes| RecEmit[Emit recurrent single-step path]
 *   Recurrent -->|No| Mixed{Mixed activations?}
 *   Mixed -->|No| DenseEmit[Emit dense Gemm + activation]
 *   Mixed -->|Yes| PerNeuron[Emit per-neuron Gemm + activation + Concat]
 * ```
 */
import type NeatapticNode from '../../../../node';
import type {
  LayerActivationContext,
  LayerBuildContext,
  LayerRecurrentDecisionContext,
  LayerTraversalContext,
  OnnxExportOptions,
} from '../network.onnx.export.types';
import type { NodeInternals } from '../../network.onnx.utils.types';
import { NetworkOnnxRecurrentMixedActivationsUnsupportedError } from '../../network.onnx.errors';
import {
  appendConcatMergeMetadata,
  appendResidualAddMetadata,
  buildConcatMergeNodeName,
  buildConcatMergeOutputName,
  buildResidualBranchTensorName,
  buildResidualMergeNodeName,
  buildResidualMergeOutputName,
  resolveOneHopResidualSourceLayerIndex,
} from '../network.onnx.export-advanced-graph.utils';
import {
  emitDenseLayer,
  emitPerNeuronLayer,
  emitResidualAddLayer,
} from './network.onnx.export-dense.utils';
import { tryEmitConvLayer } from './network.onnx.export-conv.utils';
import { emitRecurrentLayer } from './network.onnx.export-recurrent.utils';

/**
 * Emit one export layer graph segment by routing the layer through the correct
 * ONNX emission strategy.
 *
 * Dispatch order matters:
 * - explicit Conv mappings win first,
 * - recurrent single-step export is considered only for hidden layers with
 *   self-connections,
 * - non-recurrent layers fall back to compact dense emission or mixed-activation
 *   per-neuron decomposition.
 *
 * Important invariants:
 * - recurrent mixed activations are rejected elsewhere rather than silently
 *   decomposed here,
 * - `allowMixedActivations` only affects the dense-family fallback path,
 * - the returned tensor name is the canonical input for the next layer.
 *
 * @param context Layer build context.
 * @returns Output tensor name produced by this layer.
 * @example
 * ```ts
 * const outputName = emitLayerGraph({
 *   model,
 *   layers,
 *   layerIndex: 2,
 *   previousOutputName: 'Layer_1',
 *   options: { allowMixedActivations: true },
 *   recurrentLayerIndices: [],
 *   batchDimension: false,
 *   legacyNodeOrdering: false,
 * });
 * ```
 */
export function emitLayerGraph(context: LayerBuildContext): string {
  const layerTraversalContext = createLayerTraversalContext(context);

  // Step 1: Attempt convolution emission branch first.
  const convOutputName = tryEmitConvolutionBranch(layerTraversalContext);
  if (convOutputName) {
    return convOutputName;
  }

  // Step 2: Build activation analysis context for dense/recurrent branching.
  const layerActivationContext = createLayerActivationContext(
    layerTraversalContext,
  );

  // Step 3: Fold to recurrent or dense/per-neuron branch output.
  return emitNonConvolutionBranch(
    layerTraversalContext,
    layerActivationContext,
  );

  /**
   * Build a compact traversal context with adjacent layers.
   *
   * @param input Base layer build context.
   * @returns Traversal context.
   */
  function createLayerTraversalContext(
    input: LayerBuildContext,
  ): LayerTraversalContext {
    const previousLayerNodes = input.layers[input.layerIndex - 1];
    const currentLayerNodes = input.layers[input.layerIndex];
    const isOutputLayer = input.layerIndex === input.layers.length - 1;
    return {
      ...input,
      previousLayerNodes,
      currentLayerNodes,
      isOutputLayer,
    };
  }

  /**
   * Attempt convolution emission and return produced output when mapped.
   *
   * @param traversalContext Layer traversal context.
   * @returns Convolution output name when emitted; otherwise null.
   */
  function tryEmitConvolutionBranch(
    traversalContext: LayerTraversalContext,
  ): string | undefined {
    return tryEmitConvLayer({
      hasLaterHiddenLayers:
        traversalContext.layerIndex < traversalContext.layers.length - 2,
      model: traversalContext.model,
      options: traversalContext.options,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: traversalContext.previousOutputName,
      previousLayerNodes: traversalContext.previousLayerNodes,
      currentLayerNodes: traversalContext.currentLayerNodes,
    });
  }

  /**
   * Build activation analysis context for non-convolution branches.
   *
   * @param traversalContext Layer traversal context.
   * @returns Activation analysis context.
   */
  function createLayerActivationContext(
    traversalContext: LayerTraversalContext,
  ): LayerActivationContext {
    return {
      hasMixedActivations: detectMixedActivations(
        traversalContext.currentLayerNodes,
        traversalContext.options,
      ),
    };
  }

  /**
   * Emit recurrent or dense/per-neuron branch output.
   *
   * @param traversalContext Layer traversal context.
   * @param activationContext Activation analysis context.
   * @returns Output tensor name.
   */
  function emitNonConvolutionBranch(
    traversalContext: LayerTraversalContext,
    activationContext: LayerActivationContext,
  ): string {
    if (
      shouldEmitRecurrentBranch(
        createRecurrentDecisionContext(traversalContext),
      )
    ) {
      return emitRecurrentBranch(traversalContext, activationContext);
    }
    return emitDenseFamilyBranch(traversalContext, activationContext);
  }

  /**
   * Build recurrent decision context with no extra parameters.
   *
   * @param traversalContext Layer traversal context.
   * @returns Recurrent decision context.
   */
  function createRecurrentDecisionContext(
    traversalContext: LayerTraversalContext,
  ): LayerRecurrentDecisionContext {
    return {
      recurrentLayerIndices: traversalContext.recurrentLayerIndices,
      layerIndex: traversalContext.layerIndex,
      isOutputLayer: traversalContext.isOutputLayer,
    };
  }

  /**
   * Determine whether recurrent single-step emission applies.
   *
   * @param decisionContext Recurrent branch decision context.
   * @returns Whether recurrent branch should be emitted.
   */
  function shouldEmitRecurrentBranch(
    decisionContext: LayerRecurrentDecisionContext,
  ): boolean {
    return (
      decisionContext.recurrentLayerIndices.includes(
        decisionContext.layerIndex,
      ) && !decisionContext.isOutputLayer
    );
  }

  /**
   * Emit recurrent layer branch with mixed-activation validation.
   *
   * @param traversalContext Layer traversal context.
   * @param activationContext Activation analysis context.
   * @returns Recurrent output tensor name.
   */
  function emitRecurrentBranch(
    traversalContext: LayerTraversalContext,
    activationContext: LayerActivationContext,
  ): string {
    ensureRecurrentSupportsActivations(
      traversalContext.layerIndex,
      activationContext,
    );
    return emitRecurrentLayer({
      model: traversalContext.model,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: traversalContext.previousOutputName,
      previousLayerNodes: traversalContext.previousLayerNodes,
      currentLayerNodes: traversalContext.currentLayerNodes,
      opset: traversalContext.options.opset ?? 18,
    });
  }

  /**
   * Emit dense or per-neuron layer branch from activation analysis.
   *
   * @param traversalContext Layer traversal context.
   * @param activationContext Activation analysis context.
   * @returns Output tensor name.
   */
  function emitDenseFamilyBranch(
    traversalContext: LayerTraversalContext,
    activationContext: LayerActivationContext,
  ): string {
    if (!activationContext.hasMixedActivations) {
      const concatOutputName =
        tryEmitExplicitConcatMergeBranch(traversalContext);
      if (concatOutputName) {
        return concatOutputName;
      }

      const residualOutputName = tryEmitResidualAddBranch(traversalContext);
      if (residualOutputName) {
        return residualOutputName;
      }
      return emitDenseBranch(traversalContext);
    }
    return emitPerNeuronBranch(traversalContext);
  }

  /**
   * Attempt the narrow one-hop residual-add subset before falling back.
   *
   * @param traversalContext Layer traversal context.
   * @returns Residual-add output tensor name when emitted; otherwise null.
   */
  function tryEmitResidualAddBranch(
    traversalContext: LayerTraversalContext,
  ): string | undefined {
    const residualSourceLayerIndex = resolveOneHopResidualSourceLayerIndex(
      traversalContext.currentLayerNodes,
      traversalContext.layers,
      traversalContext.layerIndex,
    );
    if (residualSourceLayerIndex === null) {
      return undefined;
    }

    const residualSourceOutputName =
      traversalContext.layerOutputNamesByLayerIndex.get(
        residualSourceLayerIndex,
      );
    if (!residualSourceOutputName) {
      return undefined;
    }

    const branchTensorName = buildResidualBranchTensorName(
      residualSourceLayerIndex,
      traversalContext.layerIndex,
    );
    const mergeNodeName = buildResidualMergeNodeName(
      traversalContext.layerIndex,
    );
    const mergeOutputName = buildResidualMergeOutputName(
      traversalContext.layerIndex,
    );
    const residualOutputName = emitResidualAddLayer({
      model: traversalContext.model,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: traversalContext.previousOutputName,
      residualSourceOutputName,
      previousLayerNodes: traversalContext.previousLayerNodes,
      residualSourceLayerNodes:
        traversalContext.layers[residualSourceLayerIndex],
      currentLayerNodes: traversalContext.currentLayerNodes,
      branchTensorName,
      mergeNodeName,
      mergeOutputName,
      options: traversalContext.options,
    });

    appendResidualAddMetadata(
      traversalContext.model,
      {
        sourceLayerIndex: residualSourceLayerIndex,
        targetLayerIndex: traversalContext.layerIndex,
        branchTensorName,
        mergeNodeName,
        mergeOutputName,
      },
      traversalContext.options.includeMetadata ?? false,
    );
    return residualOutputName;
  }

  /**
   * Attempt the narrow explicit concat subset before residual fallback.
   *
   * @param traversalContext Layer traversal context.
   * @returns Concat-merge output tensor name when emitted; otherwise undefined.
   */
  function tryEmitExplicitConcatMergeBranch(
    traversalContext: LayerTraversalContext,
  ): string | undefined {
    const concatMapping = traversalContext.options.concatMappings?.find(
      (mapping) => mapping.targetLayerIndex === traversalContext.layerIndex,
    );
    if (!concatMapping) {
      return undefined;
    }

    const inputOrder = concatMapping.inputOrder ?? 'previous_then_source';
    if (
      inputOrder !== 'previous_then_source' ||
      concatMapping.sourceLayerIndex >= traversalContext.layerIndex - 1
    ) {
      return undefined;
    }

    const concatSourceLayerNodes =
      traversalContext.layers[concatMapping.sourceLayerIndex];
    const concatSourceOutputName =
      traversalContext.layerOutputNamesByLayerIndex.get(
        concatMapping.sourceLayerIndex,
      );
    if (!concatSourceLayerNodes || !concatSourceOutputName) {
      return undefined;
    }

    const concatNodeName = buildConcatMergeNodeName(
      concatMapping.sourceLayerIndex,
      traversalContext.layerIndex,
    );
    const concatOutputName = buildConcatMergeOutputName(
      concatMapping.sourceLayerIndex,
      traversalContext.layerIndex,
    );

    traversalContext.model.graph.node.push({
      op_type: 'Concat',
      input: [traversalContext.previousOutputName, concatSourceOutputName],
      output: [concatOutputName],
      name: concatNodeName,
      attributes: [
        {
          name: 'axis',
          type: 'INT',
          i: traversalContext.batchDimension ? 1 : 0,
        },
      ],
    });

    const concatBranchOutputName = emitDenseLayer({
      model: traversalContext.model,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: concatOutputName,
      previousLayerNodes: [
        ...traversalContext.previousLayerNodes,
        ...concatSourceLayerNodes,
      ],
      currentLayerNodes: traversalContext.currentLayerNodes,
      legacyNodeOrdering: traversalContext.legacyNodeOrdering,
      options: traversalContext.options,
    });

    appendConcatMergeMetadata(
      traversalContext.model,
      {
        sourceLayerIndex: concatMapping.sourceLayerIndex,
        targetLayerIndex: traversalContext.layerIndex,
        concatNodeName,
        concatOutputName,
        inputOrder,
      },
      traversalContext.options.includeMetadata ?? false,
    );
    return concatBranchOutputName;
  }

  /**
   * Emit standard dense layer branch.
   *
   * @param traversalContext Layer traversal context.
   * @returns Dense output tensor name.
   */
  function emitDenseBranch(traversalContext: LayerTraversalContext): string {
    return emitDenseLayer({
      model: traversalContext.model,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: traversalContext.previousOutputName,
      previousLayerNodes: traversalContext.previousLayerNodes,
      currentLayerNodes: traversalContext.currentLayerNodes,
      legacyNodeOrdering: traversalContext.legacyNodeOrdering,
      options: traversalContext.options,
    });
  }

  /**
   * Emit per-neuron decomposition branch for mixed activations.
   *
   * @param traversalContext Layer traversal context.
   * @returns Per-neuron output tensor name.
   */
  function emitPerNeuronBranch(
    traversalContext: LayerTraversalContext,
  ): string {
    return emitPerNeuronLayer({
      model: traversalContext.model,
      layerIndex: traversalContext.layerIndex,
      previousOutputName: traversalContext.previousOutputName,
      previousLayerNodes: traversalContext.previousLayerNodes,
      currentLayerNodes: traversalContext.currentLayerNodes,
      options: traversalContext.options,
      batchDimension: traversalContext.batchDimension,
    });
  }

  /**
   * Ensure recurrent layers do not use unsupported mixed activations.
   *
   * @param layerIndex Layer index.
   * @param activationContext Activation analysis context.
   * @returns Nothing.
   */
  function ensureRecurrentSupportsActivations(
    layerIndex: number,
    activationContext: LayerActivationContext,
  ): void {
    if (!activationContext.hasMixedActivations) {
      return;
    }
    throw new NetworkOnnxRecurrentMixedActivationsUnsupportedError(
      `Recurrent export does not yet support mixed activations in hidden layer ${layerIndex}.`,
    );
  }

  /**
   * Determine whether a layer has mixed activation functions.
   *
   * @param currentLayerNodes Current layer nodes.
   * @param options Export options.
   * @returns Whether mixed activations are present and enabled.
   */
  function detectMixedActivations(
    currentLayerNodes: NeatapticNode[],
    options: OnnxExportOptions,
  ): boolean {
    if (!options.allowMixedActivations) {
      return false;
    }
    return collectActivationNames(currentLayerNodes).size > 1;
  }

  /**
   * Collect activation names for current-layer nodes.
   *
   * @param currentLayerNodes Current layer nodes.
   * @returns Activation name set.
   */
  function collectActivationNames(
    currentLayerNodes: NeatapticNode[],
  ): Set<string | undefined> {
    return new Set(
      currentLayerNodes.map((node) => resolveActivationName(node)),
    );
  }

  /**
   * Resolve the activation name for one node.
   *
   * @param node Current layer node.
   * @returns Activation name when present.
   */
  function resolveActivationName(node: NeatapticNode): string | undefined {
    const nodeInternal = node as NodeInternals;
    return nodeInternal.squash?.name;
  }
}
