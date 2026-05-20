import type { OnnxModel, OnnxNode } from '../schema/network.onnx.schema.types';

const ACTIVATION_NODE_NAME_PATTERN = /^act(?:_conv)?_l\d+(?:_n\d+)?$/i;

/**
 * Remove exporter-owned Identity activation nodes by rewiring their consumers.
 *
 * @param model - ONNX-like model to optimize in place.
 * @returns Nothing.
 */
export function pruneIdentityActivationNodes(model: OnnxModel): void {
  // Step 1: Collect activation-shaped Identity rewrites.
  const identityReplacements = collectIdentityReplacements(model.graph.node);
  if (identityReplacements.size === 0) {
    return;
  }

  // Step 2: Rewire node inputs through the replacement map.
  model.graph.node = model.graph.node
    .filter((graphNode) => !shouldPruneIdentityNode(graphNode))
    .map((graphNode) => ({
      ...graphNode,
      input: graphNode.input.map((inputName) =>
        resolveReplacementInputName(inputName, identityReplacements),
      ),
    }));

  // Step 3: Rewire graph outputs for completeness.
  model.graph.outputs = model.graph.outputs.map((valueInfo) => ({
    ...valueInfo,
    name: resolveReplacementInputName(valueInfo.name, identityReplacements),
  }));
}

function collectIdentityReplacements(
  graphNodes: OnnxNode[],
): Map<string, string> {
  return new Map(
    graphNodes
      .filter((graphNode) => shouldPruneIdentityNode(graphNode))
      .map((graphNode) => [graphNode.output[0], graphNode.input[0]]),
  );
}

function shouldPruneIdentityNode(graphNode: OnnxNode): boolean {
  return (
    graphNode.op_type === 'Identity' &&
    ACTIVATION_NODE_NAME_PATTERN.test(graphNode.name) &&
    graphNode.input.length === 1 &&
    graphNode.output.length === 1
  );
}

function resolveReplacementInputName(
  inputName: string,
  identityReplacements: Map<string, string>,
): string {
  let resolvedInputName = inputName;

  while (identityReplacements.has(resolvedInputName)) {
    resolvedInputName = identityReplacements.get(resolvedInputName)!;
  }

  return resolvedInputName;
}
