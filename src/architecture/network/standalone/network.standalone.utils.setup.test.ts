import Network from '../network';
import {
  resolveActivationTraversalNodes,
  resolveOrderedOutputNodes,
} from '../activate/network.activate.schedule.utils';
import {
  asStandaloneProps,
  createGenerationContext,
  ensureOutputNodesExist,
  resolveStandaloneExecutionMetadata,
  seedNodeIndexesAndState,
} from './network.standalone.utils.setup';
import { NetworkStandaloneNoOutputNodesError } from './network.standalone.errors';

jest.mock('../activate/network.activate.schedule.utils', () => ({
  resolveActivationTraversalNodes: jest.fn(),
  resolveOrderedOutputNodes: jest.fn(),
}));

const mockedResolveActivationTraversalNodes = jest.mocked(
  resolveActivationTraversalNodes,
);
const mockedResolveOrderedOutputNodes = jest.mocked(resolveOrderedOutputNodes);

function createStandaloneNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

function createSetupContext(network: Network) {
  return createGenerationContext(asStandaloneProps(network));
}

function findRequiredNode(
  network: Network,
  nodeType: 'input' | 'output',
  occurrence = 0,
) {
  const matchingNode = network.nodes.filter(
    (candidateNode) => candidateNode.type === nodeType,
  )[occurrence];

  if (!matchingNode) {
    throw new Error(`Expected ${nodeType} node at occurrence ${occurrence}.`);
  }

  return matchingNode;
}

describe('network standalone setup utility chapter', () => {
  afterEach(() => {
    jest.resetAllMocks();
  });

  describe('asStandaloneProps', () => {
    describe('when the runtime network is cast for standalone generation', () => {
      it('returns the same network reference', () => {
        // Arrange
        const network = createStandaloneNetwork(7501);

        // Act
        const standaloneProps = asStandaloneProps(network);

        // Assert
        expect(standaloneProps).toBe(network);
      });
    });
  });

  describe('ensureOutputNodesExist', () => {
    describe('when the standalone view still has one output node', () => {
      it('returns without throwing an error', () => {
        // Arrange
        const network = createStandaloneNetwork(7502);

        // Act
        const ensureOutputs = () =>
          ensureOutputNodesExist(asStandaloneProps(network));

        // Assert
        expect(ensureOutputs).not.toThrow();
      });
    });

    describe('when the standalone view has no output nodes', () => {
      it('throws the dedicated no-output-nodes error', () => {
        // Arrange
        const network = createStandaloneNetwork(7503);
        network.nodes = network.nodes.filter(
          (candidateNode) => candidateNode.type === 'input',
        );

        // Act
        const ensureOutputs = () =>
          ensureOutputNodesExist(asStandaloneProps(network));

        // Assert
        expect(ensureOutputs).toThrow(NetworkStandaloneNoOutputNodesError);
      });
    });
  });

  describe('createGenerationContext', () => {
    describe('when a fresh standalone generation pass begins', () => {
      it('creates an empty context for all generated arrays and maps', () => {
        // Arrange
        const network = createStandaloneNetwork(7504);
        const standaloneProps = asStandaloneProps(network);

        // Act
        const generationContext = createGenerationContext(standaloneProps);

        // Assert
        expect(generationContext).toEqual({
          standaloneProps,
          resolvedActivationPrecision: 'f64',
          inputNodeIndexes: [],
          activationNodeIndexes: [],
          outputNodeIndexes: [],
          emittedActivationSource: {},
          activationFunctionSources: [],
          activationFunctionIndexMap: {},
          nextActivationFunctionIndex: 0,
          initialActivations: [],
          initialStates: [],
          bodyLines: [],
        });
      });
    });
  });

  describe('seedNodeIndexesAndState', () => {
    describe('when the standalone context is seeded from runtime nodes', () => {
      it('assigns sequential indexes and captures the initial node values', () => {
        // Arrange
        const network = createStandaloneNetwork(7505);
        const generationContext = createSetupContext(network);

        // Act
        seedNodeIndexesAndState(generationContext);

        // Assert
        expect({
          initialActivations: generationContext.initialActivations,
          initialStates: generationContext.initialStates,
          nodeIndexes: generationContext.standaloneProps.nodes.map(
            (candidateNode) =>
              (candidateNode as { index?: number }).index ?? null,
          ),
        }).toEqual({
          initialActivations: network.nodes.map(
            (candidateNode) => candidateNode.activation,
          ),
          initialStates: network.nodes.map(
            (candidateNode) => candidateNode.state,
          ),
          nodeIndexes: [0, 1, 2],
        });
      });
    });
  });

  describe('resolveStandaloneExecutionMetadata', () => {
    describe('when scheduling metadata is missing but explicit input ids are complete', () => {
      it('recomputes execution order and keeps only indexed non-input traversal nodes', () => {
        // Arrange
        const network = createStandaloneNetwork(7506);
        const generationContext = createSetupContext(network);
        const firstInputNode = findRequiredNode(network, 'input', 0);
        const secondInputNode = findRequiredNode(network, 'input', 1);
        const outputNode = findRequiredNode(network, 'output');
        const computeTopoOrder = jest.fn();

        seedNodeIndexesAndState(generationContext);
        Reflect.set(network, '_computeTopoOrder', computeTopoOrder);
        Reflect.set(network, '_topoDirty', false);
        Reflect.set(network, '_activationSchedule', undefined);
        Reflect.set(network, '_topoOrder', undefined);
        mockedResolveActivationTraversalNodes.mockReturnValue([
          firstInputNode,
          outputNode,
          { type: 'hidden' } as unknown as Network['nodes'][number],
        ]);
        mockedResolveOrderedOutputNodes.mockReturnValue([
          outputNode,
          { type: 'output' } as unknown as Network['nodes'][number],
        ]);

        // Act
        resolveStandaloneExecutionMetadata(network, generationContext);

        // Assert
        expect({
          activationNodeIndexes: generationContext.activationNodeIndexes,
          computeTopoOrderCalls: computeTopoOrder.mock.calls.length,
          inputNodeIndexes: generationContext.inputNodeIndexes,
          outputNodeIndexes: generationContext.outputNodeIndexes,
        }).toEqual({
          activationNodeIndexes: [(outputNode as { index: number }).index],
          computeTopoOrderCalls: 1,
          inputNodeIndexes: [
            (firstInputNode as { index: number }).index,
            (secondInputNode as { index: number }).index,
          ],
          outputNodeIndexes: [(outputNode as { index: number }).index],
        });
      });
    });

    describe('when a cached schedule exists but explicit input ids are incomplete', () => {
      it('falls back to scanning indexed runtime input nodes without recomputing topology', () => {
        // Arrange
        const network = createStandaloneNetwork(7507);
        const generationContext = createSetupContext(network);
        const firstInputNode = findRequiredNode(network, 'input', 0);
        const secondInputNode = findRequiredNode(network, 'input', 1);
        const computeTopoOrder = jest.fn();

        Reflect.set(firstInputNode, 'index', 0);
        Reflect.set(secondInputNode, 'index', undefined);
        Reflect.set(network, '_computeTopoOrder', computeTopoOrder);
        Reflect.set(network, '_topoDirty', false);
        Reflect.set(network, '_activationSchedule', { mode: 'acyclic' });
        Reflect.set(network, '_inputNodeIds', [999999, firstInputNode.geneId]);
        mockedResolveActivationTraversalNodes.mockReturnValue([]);
        mockedResolveOrderedOutputNodes.mockReturnValue([]);

        // Act
        resolveStandaloneExecutionMetadata(network, generationContext);

        // Assert
        expect({
          computeTopoOrderCalls: computeTopoOrder.mock.calls.length,
          inputNodeIndexes: generationContext.inputNodeIndexes,
        }).toEqual({
          computeTopoOrderCalls: 0,
          inputNodeIndexes: [0],
        });
      });
    });

    describe('when topology is marked dirty even with cached scheduling data', () => {
      it('recomputes topology before resolving execution metadata', () => {
        // Arrange
        const network = createStandaloneNetwork(7508);
        const generationContext = createSetupContext(network);
        const firstInputNode = findRequiredNode(network, 'input', 0);
        const secondInputNode = findRequiredNode(network, 'input', 1);
        const computeTopoOrder = jest.fn();

        Reflect.set(firstInputNode, 'index', 0);
        Reflect.set(secondInputNode, 'index', 1);
        Reflect.set(network, '_computeTopoOrder', computeTopoOrder);
        Reflect.set(network, '_topoDirty', true);
        Reflect.set(network, '_activationSchedule', { mode: 'acyclic' });
        Reflect.set(network, '_inputNodeIds', [
          firstInputNode.geneId,
          secondInputNode.geneId,
        ]);
        mockedResolveActivationTraversalNodes.mockReturnValue([]);
        mockedResolveOrderedOutputNodes.mockReturnValue([]);

        // Act
        resolveStandaloneExecutionMetadata(network, generationContext);

        // Assert
        expect(computeTopoOrder).toHaveBeenCalledTimes(1);
      });
    });
  });
});
