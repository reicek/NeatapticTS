import type Network from '../../network/network';
import type Node from '../../node';
import type {
  NetworkStandaloneProps,
  NodeWithIndex,
  StandaloneGenerationContext as GenerationContext,
} from '../network.types';
import {
  NO_OUTPUT_NODES_ERROR,
  OUTPUT_NODE_TYPE,
} from './network.standalone.utils.types';
import { NetworkStandaloneNoOutputNodesError } from './network.standalone.errors';

/**
 * Cast a network instance to the internal standalone generation view.
 *
 * @param net Network instance to cast.
 * @returns Internal network properties used by the standalone generator.
 */
export function asStandaloneProps(net: Network): NetworkStandaloneProps {
  return net as unknown as NetworkStandaloneProps;
}

/**
 * Validate that the network has at least one output node.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Void.
 * @throws If no output node exists.
 */
export function ensureOutputNodesExist(
  standaloneProps: NetworkStandaloneProps,
): void {
  const hasOutputNode = standaloneProps.nodes.some(function hasOutput(
    node: Node,
  ) {
    return node.type === OUTPUT_NODE_TYPE;
  });

  if (!hasOutputNode) {
    throw new NetworkStandaloneNoOutputNodesError(NO_OUTPUT_NODES_ERROR);
  }
}

/**
 * Create a fresh generation context used across orchestration steps.
 *
 * @param standaloneProps Internal standalone network view.
 * @returns Initialized generation context.
 */
export function createGenerationContext(
  standaloneProps: NetworkStandaloneProps,
): GenerationContext {
  return {
    standaloneProps,
    emittedActivationSource: {},
    activationFunctionSources: [],
    activationFunctionIndexMap: {},
    nextActivationFunctionIndex: 0,
    initialActivations: [],
    initialStates: [],
    bodyLines: [],
  };
}

/**
 * Seed index, activation, and state arrays from network nodes.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function seedNodeIndexesAndState(
  generationContext: GenerationContext,
): void {
  const allNodes = generationContext.standaloneProps.nodes;

  for (
    let nodeTraversalIndex = 0;
    nodeTraversalIndex < allNodes.length;
    nodeTraversalIndex++
  ) {
    const currentNode = allNodes[nodeTraversalIndex] as NodeWithIndex;
    currentNode.index = nodeTraversalIndex;
    generationContext.initialActivations.push(currentNode.activation);
    generationContext.initialStates.push(currentNode.state);
  }
}
