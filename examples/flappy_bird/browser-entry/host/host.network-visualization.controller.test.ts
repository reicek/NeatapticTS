/**
 * Tests for the live champion activation overlay on the host network
 * visualization controller.
 *
 * The host network panel controller exposes an `applyNetworkActivationOverlay`
 * handle so streamed playback frames can paint the frame winner's real node
 * activations onto the network instance that is currently visualized. Identity
 * and activation-length guards keep stale frames from mutating the wrong payload,
 * and every applied overlay requests a coalesced redraw so the painted labels refresh.
 */

import Network from '../../../../src/architecture/network';
import * as hostModule from './host';

/**
 * Structural contract for the controller handle exercised by these tests.
 *
 * The concrete `HostNetworkVisualizationController` type stays module-private
 * in `./host`, so the tests describe the minimal surface they rely on.
 */
interface HostNetworkVisualizationControllerContract {
  renderNetworkArchitecture: (
    network: Network | undefined,
    inputSize: number,
    outputSize: number,
  ) => void;
  applyNetworkActivationOverlay: (
    network: Network,
    winnerNodeActivations: Float32Array,
  ) => void;
}

type HostNetworkVisualizationControllerFactory = (
  networkCanvasHost: HTMLDivElement,
  networkCanvas: HTMLCanvasElement,
  networkContext: CanvasRenderingContext2D,
) => HostNetworkVisualizationControllerContract;

/**
 * Resolves the network-visualization controller factory from the host module.
 *
 * @returns The controller factory exported from the host module.
 */
function resolveCreateHostNetworkVisualizationController(): HostNetworkVisualizationControllerFactory {
  const candidate = (hostModule as Record<string, unknown>)[
    'createHostNetworkVisualizationController'
  ];

  if (typeof candidate !== 'function') {
    throw new Error(
      'createHostNetworkVisualizationController is not exported from ./host yet — implement and export per AC-212 in slice 02-host-overlay',
    );
  }

  return candidate as HostNetworkVisualizationControllerFactory;
}

/**
 * Creates a tooltip element stub covering every DOM surface the tooltip
 * lifecycle touches (style/attribute writes, child mounting, text updates).
 *
 * @returns A fresh element stub.
 */
function createStubTooltipElement(): Record<string, unknown> {
  return {
    style: {},
    dataset: {},
    textContent: '',
    offsetHeight: 12,
    setAttribute: jest.fn(),
    appendChild: jest.fn(),
    replaceChildren: jest.fn(),
    addEventListener: jest.fn(),
  };
}

/**
 * Creates a 2D-context stub matching the shape the network draw path reads
 * (canvas metrics, text measurement, and no-op paint primitives).
 *
 * @returns A canvas rendering context stub.
 */
function createStubNetworkContext(): CanvasRenderingContext2D {
  const canvas = {
    width: 800,
    height: 600,
    ownerDocument: { defaultView: { innerWidth: 800 } },
  } as unknown as HTMLCanvasElement;

  return {
    canvas,
    save: jest.fn(),
    restore: jest.fn(),
    fillRect: jest.fn(),
    strokeRect: jest.fn(),
    clearRect: jest.fn(),
    beginPath: jest.fn(),
    closePath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    quadraticCurveTo: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn(),
    fillText: jest.fn(),
    measureText: jest.fn((text: string) => ({
      width: text.length * 6,
      actualBoundingBoxAscent: 8,
      actualBoundingBoxDescent: 2,
    })),
    setLineDash: jest.fn(),
    arc: jest.fn(),
    rect: jest.fn(),
    clip: jest.fn(),
    getImageData: jest.fn(() => ({ data: [] })),
    putImageData: jest.fn(),
    createLinearGradient: jest.fn(() => ({ addColorStop: jest.fn() })),
    createPattern: jest.fn(() => null),
    drawImage: jest.fn(),
    translate: jest.fn(),
    rotate: jest.fn(),
    scale: jest.fn(),
    transform: jest.fn(),
    setTransform: jest.fn(),
    resetTransform: jest.fn(),
    createImageData: jest.fn(() => ({ data: [] })),
    getTransform: jest.fn(() => [1, 0, 0, 1, 0, 0]),
    isPointInPath: jest.fn(() => false),
    isPointInStroke: jest.fn(() => false),
    fillStyle: '',
    strokeStyle: '',
    font: '',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    lineWidth: 1,
    lineCap: 'butt',
    globalAlpha: 1,
    shadowBlur: 0,
    shadowColor: '',
  } as unknown as CanvasRenderingContext2D;
}

describe('createHostNetworkVisualizationController activation overlay', () => {
  let requestAnimationFrameCallbacks: Array<() => void>;
  let requestAnimationFrameCallCount: number;

  beforeEach(() => {
    requestAnimationFrameCallbacks = [];
    requestAnimationFrameCallCount = 0;

    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: {
        createElement: jest.fn(() => createStubTooltipElement()),
      },
    });
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      configurable: true,
      value: (callback: () => void): number => {
        requestAnimationFrameCallbacks.push(callback);
        requestAnimationFrameCallCount += 1;
        return requestAnimationFrameCallCount;
      },
    });
  });

  afterEach(() => {
    Reflect.deleteProperty(globalThis, 'document');
    Reflect.deleteProperty(globalThis, 'requestAnimationFrame');
    requestAnimationFrameCallbacks = [];
    requestAnimationFrameCallCount = 0;
  });

  /**
   * Drains every pending animation-frame callback (including ones scheduled
   * while draining) so a subsequent overlay can request a fresh redraw instead
   * of being coalesced into the render frame.
   */
  const flushPendingRedraws = (): void => {
    while (requestAnimationFrameCallbacks.length > 0) {
      const pendingCallbacks = requestAnimationFrameCallbacks.splice(
        0,
        requestAnimationFrameCallbacks.length,
      );
      pendingCallbacks.forEach((pendingCallback) => pendingCallback());
    }
  };

  /**
   * Builds the controller harness: stubbed DOM host/canvas, a deterministic
   * champion network, and the controller under test.
   *
   * @returns The controller and the champion network it visualizes.
   */
  const createControllerHarness = (): {
    controller: HostNetworkVisualizationControllerContract;
    championNetwork: Network;
  } => {
    const networkCanvasHost = {
      clientWidth: 800,
      clientHeight: 600,
      appendChild: jest.fn(),
      style: {},
      dataset: {},
    } as unknown as HTMLDivElement;
    const networkCanvas = {
      width: 0,
      height: 0,
      style: {},
      addEventListener: jest.fn(),
    } as unknown as HTMLCanvasElement;
    const championNetwork = new Network(2, 1, { seed: 42 });
    const controller = resolveCreateHostNetworkVisualizationController()(
      networkCanvasHost,
      networkCanvas,
      createStubNetworkContext(),
    );

    return { controller, championNetwork };
  };

  it('copies streamed winner activations into the visualized network nodes', () => {
    // Arrange — visualize the champion and settle its render frame.
    const { controller, championNetwork } = createControllerHarness();
    controller.renderNetworkArchitecture(championNetwork, 2, 1);
    flushPendingRedraws();
    const winnerNodeActivations = new Float32Array([0.5, -1.5, 2]);

    // Act — apply the streamed overlay for the visualized champion.
    controller.applyNetworkActivationOverlay(
      championNetwork,
      winnerNodeActivations,
    );

    // Assert — each visualized node carries the streamed activation.
    expect(championNetwork.nodes[0]!.activation).toBe(0.5);
    expect(championNetwork.nodes[1]!.activation).toBe(-1.5);
    expect(championNetwork.nodes[2]!.activation).toBe(2);
  });

  it('skips the overlay when the target network is not the visualized instance', () => {
    // Arrange — visualize the champion, then prepare an impostor payload.
    const { controller, championNetwork } = createControllerHarness();
    controller.renderNetworkArchitecture(championNetwork, 2, 1);
    flushPendingRedraws();
    const activationsBefore = championNetwork.nodes.map(
      (node) => node.activation,
    );
    const impostorNetwork = new Network(2, 1, { seed: 42 });
    const callCountBefore = requestAnimationFrameCallCount;

    // Act — apply the overlay for a network that is not visualized.
    controller.applyNetworkActivationOverlay(
      impostorNetwork,
      new Float32Array([0.5, -1.5, 2]),
    );

    // Assert — the visualized payload is untouched and no redraw is queued.
    expect(championNetwork.nodes.map((node) => node.activation)).toEqual(
      activationsBefore,
    );
    expect(requestAnimationFrameCallCount).toBe(callCountBefore);
  });

  it('skips the overlay when the activation length mismatches the node count', () => {
    // Arrange — visualize the champion, then truncate the activation stream.
    const { controller, championNetwork } = createControllerHarness();
    controller.renderNetworkArchitecture(championNetwork, 2, 1);
    flushPendingRedraws();
    const activationsBefore = championNetwork.nodes.map(
      (node) => node.activation,
    );
    const callCountBefore = requestAnimationFrameCallCount;

    // Act — apply an overlay shorter than the visualized node collection.
    controller.applyNetworkActivationOverlay(
      championNetwork,
      new Float32Array([0.5, -1.5]),
    );

    // Assert — the visualized payload is untouched and no redraw is queued.
    expect(championNetwork.nodes.map((node) => node.activation)).toEqual(
      activationsBefore,
    );
    expect(requestAnimationFrameCallCount).toBe(callCountBefore);
  });

  it('requests exactly one redraw when the overlay is applied', () => {
    // Arrange — visualize the champion and settle its render frame.
    const { controller, championNetwork } = createControllerHarness();
    controller.renderNetworkArchitecture(championNetwork, 2, 1);
    flushPendingRedraws();
    const callCountBefore = requestAnimationFrameCallCount;

    // Act — apply the streamed overlay for the visualized champion.
    controller.applyNetworkActivationOverlay(
      championNetwork,
      new Float32Array([0.5, -1.5, 2]),
    );

    // Assert — the overlay schedules a single coalesced repaint.
    expect(requestAnimationFrameCallCount).toBe(callCountBefore + 1);
  });
});