import {
  NGE_DNA_CPPN_INPUT_COUNT,
  NGE_DNA_CPPN_OUTPUT_COUNT,
} from './neat.nge-dna.constants';
import { NGE_DNA_CppnError } from './neat.nge-dna.errors';
import type {
  NgeCppnActivationKind,
  NgeCppnEdge,
  NgeCppnNode,
  NgeCppnProgram,
} from './neat.nge-dna.types';

const DEFAULT_OUTPUT_ACTIVATION_KIND: NgeCppnActivationKind = 'linear';
const DEFAULT_OUTPUT_BIAS = 0;

const CPPN_ACTIVATION_BY_KIND: Record<
  NgeCppnActivationKind,
  (inputValue: number) => number
> = {
  gaussian: (inputValue) => Math.exp(-(inputValue * inputValue)),
  linear: (inputValue) => inputValue,
  sigmoid: (inputValue) => 1 / (1 + Math.exp(-inputValue)),
  sine: (inputValue) => Math.sin(inputValue),
  tanh: (inputValue) => Math.tanh(inputValue),
};

/**
 * Evaluate one feedforward CPPN program over the canonical seven-dimensional input vector.
 *
 * @param program - Canonical CPPN descriptor carried by the DNA envelope.
 * @param inputVector - Ordered `[x1, y1, z1, x2, y2, z2, dist]` input vector.
 * @returns Fixed `[weight, enableBias]` output tuple.
 * @throws NGE_DNA_CppnError When the program shape is invalid or cyclic.
 */
export function evaluateCppnProgram(
  program: NgeCppnProgram,
  inputVector: readonly number[],
): readonly [number, number] {
  // Step 1: Validate the fixed evaluator contract before any graph work begins.
  validateInputVector(inputVector);
  validateProgramSignature(program);

  // Step 2: Resolve the non-input node registry and one deterministic topological order.
  const nonInputNodeRegistry = buildNonInputNodeRegistry(program);
  const orderedNonInputNodeIds = resolveTopologicalOrder(
    program,
    nonInputNodeRegistry,
  );
  const incomingEdgesByTarget = groupIncomingEdges(program.edges);

  // Step 3: Seed the input shelf, evaluate the non-input nodes, and fold the outputs.
  const nodeValues = new Map<string, number>(
    program.inputNodeIds.map((inputNodeId, inputIndex) => [
      inputNodeId,
      inputVector[inputIndex] as number,
    ]),
  );

  orderedNonInputNodeIds.forEach((nodeId) => {
    const nodeDescriptor = nonInputNodeRegistry.get(nodeId) as NgeCppnNode;
    const incomingEdges = incomingEdgesByTarget.get(nodeId) ?? [];
    const weightedInput =
      nodeDescriptor.bias +
      incomingEdges.reduce(
        (runningTotal, incomingEdge) =>
          runningTotal +
          incomingEdge.weight *
            (nodeValues.get(incomingEdge.sourceNodeId) as number),
        0,
      );

    nodeValues.set(
      nodeId,
      CPPN_ACTIVATION_BY_KIND[nodeDescriptor.activationKind](weightedInput),
    );
  });

  return [
    nodeValues.get(program.outputNodeIds[0] as string) as number,
    nodeValues.get(program.outputNodeIds[1] as string) as number,
  ];
}

function validateInputVector(inputVector: readonly number[]): void {
  if (inputVector.length !== NGE_DNA_CPPN_INPUT_COUNT) {
    throw new NGE_DNA_CppnError(
      `NGE_DNA CPPN evaluator requires ${NGE_DNA_CPPN_INPUT_COUNT} inputs, received ${inputVector.length}.`,
    );
  }
}

function validateProgramSignature(program: NgeCppnProgram): void {
  if (program.inputNodeIds.length !== NGE_DNA_CPPN_INPUT_COUNT) {
    throw new NGE_DNA_CppnError(
      `NGE_DNA CPPN program ${program.programId} must declare ${NGE_DNA_CPPN_INPUT_COUNT} input node ids.`,
    );
  }

  if (program.outputNodeIds.length !== NGE_DNA_CPPN_OUTPUT_COUNT) {
    throw new NGE_DNA_CppnError(
      `NGE_DNA CPPN program ${program.programId} must declare ${NGE_DNA_CPPN_OUTPUT_COUNT} output node ids.`,
    );
  }
}

function buildNonInputNodeRegistry(
  program: NgeCppnProgram,
): Map<string, NgeCppnNode> {
  const nonInputNodeRegistry = new Map<string, NgeCppnNode>(
    program.hiddenNodes.map((nodeDescriptor) => [
      nodeDescriptor.nodeId,
      nodeDescriptor,
    ]),
  );

  program.outputNodeIds.forEach((outputNodeId) => {
    if (!nonInputNodeRegistry.has(outputNodeId)) {
      nonInputNodeRegistry.set(outputNodeId, {
        activationKind: DEFAULT_OUTPUT_ACTIVATION_KIND,
        bias: DEFAULT_OUTPUT_BIAS,
        nodeId: outputNodeId,
      });
    }
  });

  validateEdgeEndpoints(
    program.edges,
    program.inputNodeIds,
    nonInputNodeRegistry,
  );
  return nonInputNodeRegistry;
}

function validateEdgeEndpoints(
  edges: readonly NgeCppnEdge[],
  inputNodeIds: readonly string[],
  nonInputNodeRegistry: Map<string, NgeCppnNode>,
): void {
  const knownSourceIds = new Set<string>([
    ...inputNodeIds,
    ...nonInputNodeRegistry.keys(),
  ]);
  const knownTargetIds = new Set<string>(nonInputNodeRegistry.keys());

  edges.forEach((edgeDescriptor) => {
    if (
      !knownSourceIds.has(edgeDescriptor.sourceNodeId) ||
      !knownTargetIds.has(edgeDescriptor.targetNodeId)
    ) {
      throw new NGE_DNA_CppnError(
        `NGE_DNA CPPN edge ${edgeDescriptor.sourceNodeId}->${edgeDescriptor.targetNodeId} references an unknown node id.`,
      );
    }
  });
}

function resolveTopologicalOrder(
  program: NgeCppnProgram,
  nonInputNodeRegistry: Map<string, NgeCppnNode>,
): string[] {
  const nodeIds = [...program.inputNodeIds, ...nonInputNodeRegistry.keys()];
  const inDegreeByNodeId = new Map<string, number>(
    nodeIds.map((nodeId) => [nodeId, 0]),
  );
  const outgoingTargetsBySourceId = new Map<string, string[]>();

  program.edges.forEach((edgeDescriptor) => {
    outgoingTargetsBySourceId.set(edgeDescriptor.sourceNodeId, [
      ...(outgoingTargetsBySourceId.get(edgeDescriptor.sourceNodeId) ?? []),
      edgeDescriptor.targetNodeId,
    ]);
    inDegreeByNodeId.set(
      edgeDescriptor.targetNodeId,
      (inDegreeByNodeId.get(edgeDescriptor.targetNodeId) as number) + 1,
    );
  });

  const queuedNodeIds = [
    ...program.inputNodeIds,
    ...[...nonInputNodeRegistry.keys()].filter(
      (nodeId) => (inDegreeByNodeId.get(nodeId) as number) === 0,
    ),
  ];
  const orderedNonInputNodeIds: string[] = [];
  let visitedNodeCount = 0;

  while (queuedNodeIds.length > 0) {
    const currentNodeId = queuedNodeIds.shift() as string;
    visitedNodeCount += 1;

    if (nonInputNodeRegistry.has(currentNodeId)) {
      orderedNonInputNodeIds.push(currentNodeId);
    }

    (outgoingTargetsBySourceId.get(currentNodeId) ?? []).forEach(
      (targetNodeId) => {
        const nextInDegree = (inDegreeByNodeId.get(targetNodeId) as number) - 1;
        inDegreeByNodeId.set(targetNodeId, nextInDegree);

        if (nextInDegree === 0) {
          queuedNodeIds.push(targetNodeId);
        }
      },
    );
  }

  if (visitedNodeCount !== nodeIds.length) {
    throw new NGE_DNA_CppnError(
      `NGE_DNA CPPN program ${program.programId} contains a cycle and cannot be evaluated feedforward.`,
    );
  }

  return orderedNonInputNodeIds;
}

function groupIncomingEdges(
  edges: readonly NgeCppnEdge[],
): Map<string, NgeCppnEdge[]> {
  return edges.reduce<Map<string, NgeCppnEdge[]>>(
    (incomingEdgesByTarget, edgeDescriptor) => {
      incomingEdgesByTarget.set(edgeDescriptor.targetNodeId, [
        ...(incomingEdgesByTarget.get(edgeDescriptor.targetNodeId) ?? []),
        edgeDescriptor,
      ]);
      return incomingEdgesByTarget;
    },
    new Map<string, NgeCppnEdge[]>(),
  );
}
