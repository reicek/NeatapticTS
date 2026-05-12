import Network from '../network';
import Node from '../../node';
import {
  getActivationSchedulingDiagnostics,
  getLastGradClipGroupCount,
  getLossScale,
  getRawGradientNorm,
  getTrainingStats,
  resetDropoutMasks,
} from './network.runtime.diagnostics.utils';

const SCHEDULING_REFRESH_SUGGESTION =
  'Run activate() or noTraceActivate() after structural edits to refresh the compiled scheduling cache.';

function setRuntimeField(network: Network, key: string, value: unknown): void {
  Reflect.set(network, key, value);
}

function createCompiledAcyclicNetwork(seed: number): Network {
  const network = new Network(2, 1, {
    seed,
    enforceAcyclic: true,
  });
  network.activate([0.25, 0.75]);
  setRuntimeField(network, '_activationSchedulingDiagnostics', undefined);
  return network;
}

function createCompiledRecurrentNetwork(seed: number): Network {
  const network = new Network(1, 1, {
    seed,
    enforceAcyclic: false,
  });
  const inputNode = network.nodes[0];
  const outputNode = network.nodes[1];
  const hiddenNode = new Node('hidden');

  network.nodes = [inputNode, hiddenNode, outputNode];
  network.connections.slice().forEach((connection) => {
    network.disconnect(connection.from, connection.to);
  });

  network.connect(inputNode, hiddenNode);
  network.connect(hiddenNode, hiddenNode);
  network.connect(hiddenNode, outputNode);
  network.activate([1]);
  setRuntimeField(network, '_activationSchedulingDiagnostics', undefined);

  return network;
}

describe('network runtime diagnostics utility chapter', () => {
  describe('resetDropoutMasks', () => {
    describe('given a non-layered network contains nodes with and without mask fields', () => {
      it('restores only the defined masks to one', () => {
        // Arrange
        const network = new Network(2, 1, { seed: 611 });
        setRuntimeField(network, 'layers', undefined);
        Reflect.set(network.nodes[0], 'mask', 0);
        Reflect.deleteProperty(network.nodes[1], 'mask');
        Reflect.set(network.nodes[2], 'mask', 0.25);

        // Act
        resetDropoutMasks.call(network);
        const maskSnapshot = network.nodes.map((node) =>
          Reflect.has(node, 'mask') ? Reflect.get(node, 'mask') : undefined,
        );

        // Assert
        expect(maskSnapshot).toEqual([1, undefined, 1]);
      });
    });
  });

  describe('getTrainingStats', () => {
    describe('given mixed-precision overflow counters have not been recorded yet', () => {
      it('returns zeroed fallback counters in the training snapshot', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 612 });
        setRuntimeField(network, '_lastGradNorm', undefined);
        setRuntimeField(network, '_lastRawGradNorm', 12.5);
        setRuntimeField(network, '_mixedPrecision', { lossScale: 16 });
        setRuntimeField(network, '_optimizerStep', 9);
        setRuntimeField(network, '_mixedPrecisionState', {
          goodSteps: 3,
          badSteps: 1,
        });
        setRuntimeField(network, '_lastOverflowStep', 7);

        // Act
        const trainingStats = getTrainingStats.call(network);

        // Assert
        expect(trainingStats).toEqual({
          gradNorm: 0,
          gradNormRaw: 12.5,
          lossScale: 16,
          optimizerStep: 9,
          mp: {
            good: 3,
            bad: 1,
            overflowCount: 0,
            underflowCount: 0,
            scaleUps: 0,
            scaleDowns: 0,
            lastOverflowStep: 7,
            lastUnderflowStep: -1,
          },
        });
      });
    });
  });

  describe('simple runtime readers', () => {
    describe('given raw gradient, loss-scale, and grad-clip counters were recorded', () => {
      it('returns those three scalar runtime values directly', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 617 });
        setRuntimeField(network, '_lastRawGradNorm', 6.5);
        setRuntimeField(network, '_mixedPrecision', { lossScale: 32 });
        setRuntimeField(network, '_lastGradClipGroupCount', 4);

        // Act
        const readerSnapshot = {
          rawGradientNorm: getRawGradientNorm.call(network),
          lossScale: getLossScale.call(network),
          gradClipGroupCount: getLastGradClipGroupCount.call(network),
        };

        // Assert
        expect(readerSnapshot).toEqual({
          rawGradientNorm: 6.5,
          lossScale: 32,
          gradClipGroupCount: 4,
        });
      });
    });
  });

  describe('getActivationSchedulingDiagnostics', () => {
    describe('given a compiled acyclic schedule exists without cached diagnostics', () => {
      it('rebuilds the compiled acyclic snapshot from the cached schedule', () => {
        // Arrange
        const network = createCompiledAcyclicNetwork(613);

        // Act
        const diagnostics = getActivationSchedulingDiagnostics.call(network);

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: false,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled acyclic schedule with stable wave ordering.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 2,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [],
        });
      });
    });

    describe('given a compiled recurrent schedule exists without cached diagnostics', () => {
      it('rebuilds the compiled recurrent snapshot from the cached schedule', () => {
        // Arrange
        const network = createCompiledRecurrentNetwork(614);

        // Act
        const diagnostics = getActivationSchedulingDiagnostics.call(network);

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'unconstrained',
          requestedMode: 'recurrent',
          topologyDirty: false,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled recurrent schedule with explicit recurrent-component steps and carried recurrent state.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 3,
          recurrentComponentCount: 1,
          stateSemantics: 'carry',
          cycleNodeIds: [],
          suggestions: [
            'Call clear() before a new independent sequence when carried recurrent state should reset.',
          ],
        });
      });
    });

    describe('given no compiled schedule exists on a recurrent network yet', () => {
      it('reports the raw-node-order fallback with recurrent requested mode', () => {
        // Arrange
        const network = new Network(1, 1, {
          seed: 615,
          enforceAcyclic: false,
        });
        setRuntimeField(network, '_activationSchedulingDiagnostics', undefined);
        setRuntimeField(network, '_activationSchedule', undefined);
        setRuntimeField(network, '_topoDirty', false);

        // Act
        const diagnostics = getActivationSchedulingDiagnostics.call(network);

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'unconstrained',
          requestedMode: 'recurrent',
          topologyDirty: false,
          executionPath: 'raw-node-order',
          issue: 'schedule-missing',
          message:
            'No compiled activation schedule is cached yet, so execution will use raw node order until topology is rebuilt.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 0,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [
            'Run activate() or noTraceActivate() to rebuild scheduling state after topology changes.',
          ],
        });
      });
    });

    describe('given no compiled schedule exists on an acyclic network yet', () => {
      it('reports the raw-node-order fallback with acyclic requested mode', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 618,
          enforceAcyclic: true,
        });
        setRuntimeField(network, '_activationSchedulingDiagnostics', undefined);
        setRuntimeField(network, '_activationSchedule', undefined);
        setRuntimeField(network, '_topoDirty', false);

        // Act
        const diagnostics = getActivationSchedulingDiagnostics.call(network);

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: false,
          executionPath: 'raw-node-order',
          issue: 'schedule-missing',
          message:
            'No compiled activation schedule is cached yet, so execution will use raw node order until topology is rebuilt.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 0,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [
            'Run activate() or noTraceActivate() to rebuild scheduling state after topology changes.',
          ],
        });
      });
    });

    describe('given dirty cached diagnostics already include the scheduling refresh suggestion', () => {
      it('returns the suggestion list without duplicating that message', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 616,
          enforceAcyclic: true,
        });
        setRuntimeField(network, '_activationSchedulingDiagnostics', {
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: false,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled acyclic schedule with stable wave ordering.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 2,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [SCHEDULING_REFRESH_SUGGESTION],
        });
        setRuntimeField(network, '_topoDirty', true);

        // Act
        const diagnostics = getActivationSchedulingDiagnostics.call(network);

        // Assert
        expect(diagnostics).toEqual({
          topologyIntent: 'feed-forward',
          requestedMode: 'acyclic',
          topologyDirty: true,
          executionPath: 'compiled-schedule',
          issue: null,
          message:
            'Activation is using the compiled acyclic schedule with stable wave ordering. Topology is currently dirty, so the next activation will rebuild scheduling state before it runs.',
          inputNodeIds: network.inputNodeIds,
          outputNodeIds: network.outputNodeIds,
          stepCount: 2,
          recurrentComponentCount: 0,
          stateSemantics: null,
          cycleNodeIds: [],
          suggestions: [SCHEDULING_REFRESH_SUGGESTION],
        });
      });
    });
  });
});
