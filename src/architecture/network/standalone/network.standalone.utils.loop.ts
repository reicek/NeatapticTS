import type { StandaloneGenerationContext as GenerationContext } from '../network.types';
import {
  ACTIVATION_PRECISION_F16,
  MASK_MULTIPLIER_IDENTITY,
} from './network.standalone.utils.types';
import {
  ensureActivationFunctionIndex,
  resolveSquashName,
} from './network.standalone.utils.activation';
import {
  buildNodeSumExpression,
  formatOutputArrayValues,
} from './network.standalone.utils.graph';

/**
 * Append the generated input-copy loop to the standalone body.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function appendInputSeedLine(
  generationContext: GenerationContext,
): void {
  generationContext.inputNodeIndexes.forEach((nodeIndex, inputIndex) => {
    generationContext.bodyLines.push(
      buildStoredValueWriteStatement(
        generationContext,
        'A',
        nodeIndex,
        `input[${inputIndex}]`,
      ),
    );
  });
}

/**
 * Append compute lines for all non-input nodes.
 *
 * @param generationContext Mutable generation context.
 * @returns Void.
 */
export function appendAllNodeComputationLines(
  generationContext: GenerationContext,
): void {
  for (const nodeTraversalIndex of generationContext.activationNodeIndexes) {
    appendSingleNodeComputationLines(generationContext, nodeTraversalIndex);
  }
}

/**
 * Append generated return line for output activations.
 *
 * @param generationContext Mutable generation context.
 * @param outputIndexes Output node indexes.
 * @returns Void.
 */
export function appendOutputReturnLine(
  generationContext: GenerationContext,
  outputIndexes: number[],
): void {
  if (generationContext.resolvedActivationPrecision === ACTIVATION_PRECISION_F16) {
    generationContext.bodyLines.push(
      `return finalizeStoredOutput([${formatOutputArrayValues(generationContext, outputIndexes)}], WA, WS, A, S);`,
    );
    return;
  }

  generationContext.bodyLines.push(
    `return [${formatOutputArrayValues(generationContext, outputIndexes)}];`,
  );
}

/**
 * Append state and activation lines for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index currently being emitted.
 * @returns Void.
 */
function appendSingleNodeComputationLines(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
): void {
  const currentNode =
    generationContext.standaloneProps.nodes[nodeTraversalIndex];
  const squashName = resolveSquashName(currentNode, nodeTraversalIndex);
  const activationFunctionIndex = ensureActivationFunctionIndex(
    generationContext,
    squashName,
    currentNode.squash,
    nodeTraversalIndex,
  );
  const sumExpression = buildNodeSumExpression(
    generationContext,
    currentNode,
    nodeTraversalIndex,
  );
  appendStateLine(
    generationContext,
    nodeTraversalIndex,
    sumExpression,
    currentNode.bias,
  );
  appendActivationLine(
    generationContext,
    nodeTraversalIndex,
    activationFunctionIndex,
    currentNode.mask,
  );
}

/**
 * Append generated state assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param sumExpression Generated sum expression.
 * @param biasValue Node bias.
 * @returns Void.
 */
function appendStateLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  sumExpression: string,
  biasValue: number,
): void {
  generationContext.bodyLines.push(
    buildStoredValueWriteStatement(
      generationContext,
      'S',
      nodeTraversalIndex,
      `${sumExpression} + ${biasValue}`,
    ),
  );
}

/**
 * Append generated activation assignment line for one node.
 *
 * @param generationContext Mutable generation context.
 * @param nodeTraversalIndex Node index.
 * @param activationFunctionIndex Function table index.
 * @param maskValue Multiplicative mask.
 * @returns Void.
 */
function appendActivationLine(
  generationContext: GenerationContext,
  nodeTraversalIndex: number,
  activationFunctionIndex: number,
  maskValue: number,
): void {
  const maskSuffix = buildMaskSuffix(maskValue);
  const stateReadExpression = buildStoredValueReadExpression(
    generationContext,
    'S',
    nodeTraversalIndex,
  );
  generationContext.bodyLines.push(
    buildStoredValueWriteStatement(
      generationContext,
      'A',
      nodeTraversalIndex,
      `F[${activationFunctionIndex}](${stateReadExpression})${maskSuffix}`,
    ),
  );
}

/**
 * Build optional activation mask suffix for generated assignment line.
 *
 * @param maskValue Multiplicative mask value.
 * @returns Empty suffix for identity, otherwise multiplicative fragment.
 */
function buildMaskSuffix(maskValue: number): string {
  if (maskValue === MASK_MULTIPLIER_IDENTITY) {
    return '';
  }

  return ` * ${maskValue}`;
}

/**
 * Build one storage write statement for generated standalone buffers.
 *
 * @param generationContext Mutable generation context.
 * @param bufferName Generated buffer variable name.
 * @param nodeIndex Indexed storage slot.
 * @param valueExpression Numeric expression being stored.
 * @returns Native assignment or float16 encode statement.
 */
function buildStoredValueWriteStatement(
  generationContext: GenerationContext,
  bufferName: 'A' | 'S',
  nodeIndex: number,
  valueExpression: string,
): string {
  if (generationContext.resolvedActivationPrecision === ACTIVATION_PRECISION_F16) {
    return `${resolveStandaloneBufferName(bufferName)}[${nodeIndex}] = ${valueExpression};`;
  }

  return `${bufferName}[${nodeIndex}] = ${valueExpression};`;
}

/**
 * Build one storage read expression for generated standalone buffers.
 *
 * @param generationContext Mutable generation context.
 * @param bufferName Generated buffer variable name.
 * @param nodeIndex Indexed storage slot.
 * @returns Native read or float16 decode expression.
 */
function buildStoredValueReadExpression(
  generationContext: GenerationContext,
  bufferName: 'A' | 'S',
  nodeIndex: number,
): string {
  if (generationContext.resolvedActivationPrecision === ACTIVATION_PRECISION_F16) {
    return `${resolveStandaloneBufferName(bufferName)}[${nodeIndex}]`;
  }

  return `${bufferName}[${nodeIndex}]`;
}

/**
 * Resolve the generated working-buffer variable name for one standalone buffer.
 *
 * @param bufferName Persistent standalone storage name.
 * @returns Working-buffer name used during one float16 activation call.
 */
function resolveStandaloneBufferName(bufferName: 'A' | 'S'): 'WA' | 'WS' {
  return bufferName === 'A' ? 'WA' : 'WS';
}
