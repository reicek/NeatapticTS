import Network from '../../../src/architecture/network/network';
import { Activation } from '../../../src/methods/activation/activation';
import {
  evaluateConcurrentRacingAgents,
  type RacingAgentRequest,
} from '../../../src/architecture/network/gpu/network.gpu.racing';
import { createConcurrentBufferSet } from '../../../src/architecture/network/gpu/network.gpu.buffer';

const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');
const REQUEST_COUNT = 8;

function makeGPULogisticActivation(): (x: number, derivate?: boolean) => number {
  const workerKey = 'logisticActivation2';
  const fn = (x: number, derivate = false): number =>
    Activation.logistic(x, derivate);
  (fn as unknown as Record<symbol, string>)[ACTIVATION_KEY_SYMBOL] = workerKey;
  try {
    Object.defineProperty(fn, 'name', { value: workerKey, configurable: true });
  } catch {
    // ignore
  }
  return fn;
}

async function readAdapterInfo(
  adapter: GPUAdapter | null | undefined,
): Promise<Record<string, unknown> | null> {
  if (!adapter) return null;
  try {
    if (typeof adapter.requestAdapterInfo === 'function') {
      return (await adapter.requestAdapterInfo()) as Record<string, unknown>;
    }
    const info = adapter.info;
    if (info) {
      return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      };
    }
  } catch (error) {
    return { error: String(error) };
  }
  return null;
}

function determineBrowserVisibility(): {
  state: string;
  visibilityState: string;
  hidden: boolean;
  hasFocus: boolean;
} {
  const visibilityState = document.visibilityState;
  const hidden = document.hidden;
  const hasFocus = document.hasFocus();
  let state = 'visible-background';
  if (visibilityState === 'visible' && !hidden && hasFocus) {
    state = 'visible-foreground';
  }
  return { state, visibilityState, hidden, hasFocus };
}

async function awaitVisibleForeground(timeoutMs = 8000): Promise<ReturnType<typeof determineBrowserVisibility>> {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    window.focus?.();
    document.body?.focus?.();
    const v = determineBrowserVisibility();
    if (v.state === 'visible-foreground') {
      return v;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return determineBrowserVisibility();
}

export async function runWebGPURacingConcurrentSmoke(): Promise<Record<string, unknown>> {
  const startTime = performance.now();

  // Wait for the harness to bring the browser window to the foreground.
  const visibility = await awaitVisibleForeground();

  try {
    if (visibility.state !== 'visible-foreground') {
      return {
        success: false,
        browserVisibility: visibility.state,
        visibilityDetails: visibility,
        gpuDeviceBound: false,
        adapterInfo: null,
        maxAbsDiff: NaN,
        meanAbsDiff: NaN,
        error: 'Browser window is not visible-foreground; GPU measurement aborted.',
      };
    }

    const network = Network.createMLP(2, [3], 1);
    const gpuLogistic = makeGPULogisticActivation();
    for (const node of network.nodes) {
      if (typeof node.squash === 'function') {
        node.squash = gpuLogistic;
      }
    }

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    const gpuDeviceBound = Boolean(device);
    const adapterInfo = await readAdapterInfo(adapter);

    if (!device) {
      return {
        success: false,
        browserVisibility: visibility.state,
        visibilityDetails: visibility,
        gpuDeviceBound: false,
        adapterInfo,
        maxAbsDiff: NaN,
        meanAbsDiff: NaN,
        error: 'No WebGPU device available.',
      };
    }

    const freshBufferSet = createConcurrentBufferSet(device, network);
    const bufferSetKeys = [
      'connections',
      'nodes',
      'outputs',
      'params',
      'topoLevels',
      'inStart',
    ] as const;
    const bufferSetValid = bufferSetKeys.every(
      (key) => freshBufferSet[key] && typeof freshBufferSet[key].destroy === 'function',
    );
    for (const key of bufferSetKeys) {
      freshBufferSet[key].destroy();
    }
    if (!bufferSetValid) {
      return {
        success: false,
        browserVisibility: visibility.state,
        visibilityDetails: visibility,
        gpuDeviceBound,
        adapterInfo,
        maxAbsDiff: NaN,
        meanAbsDiff: NaN,
        error: 'createConcurrentBufferSet did not return a complete buffer set.',
      };
    }

    const requests: RacingAgentRequest[] = [];
    for (let i = 0; i < REQUEST_COUNT; i++) {
      requests.push({
        network,
        inputs: new Float32Array([
          0.1 + i * 0.15,
          -0.4 + i * 0.12,
        ]),
      });
    }

    const cpuReferences = requests.map((request) =>
      Array.from(network.activate(request.inputs)),
    );

    device.pushErrorScope('validation');
    const gpuOutputs = await evaluateConcurrentRacingAgents(device, requests);
    const validationError = await device.popErrorScope();

    if (validationError) {
      return {
        success: false,
        browserVisibility: visibility.state,
        visibilityDetails: visibility,
        gpuDeviceBound,
        adapterInfo,
        maxAbsDiff: NaN,
        meanAbsDiff: NaN,
        error: `WebGPU validation error: ${(validationError as Error).message ?? String(validationError)}`,
      };
    }

    let maxAbsDiff = 0;
    let totalAbsDiff = 0;
    let elementCount = 0;
    for (let i = 0; i < requests.length; i++) {
      const cpu = cpuReferences[i];
      const gpu = Array.from(gpuOutputs[i] ?? []);
      for (let j = 0; j < cpu.length; j++) {
        const diff = Math.abs(cpu[j] - (gpu[j] ?? 0));
        maxAbsDiff = Math.max(maxAbsDiff, diff);
        totalAbsDiff += diff;
        elementCount++;
      }
    }
    const meanAbsDiff = elementCount > 0 ? totalAbsDiff / elementCount : 0;

    const success =
      gpuDeviceBound &&
      maxAbsDiff < 1e-3 &&
      meanAbsDiff < 1e-4;

    return {
      success,
      browserVisibility: visibility.state,
      visibilityDetails: visibility,
      gpuDeviceBound,
      adapterInfo,
      maxAbsDiff,
      meanAbsDiff,
      requestCount: REQUEST_COUNT,
      durationMs: performance.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      browserVisibility: visibility.state,
      visibilityDetails: visibility,
      gpuDeviceBound: false,
      adapterInfo: null,
      maxAbsDiff: NaN,
      meanAbsDiff: NaN,
      error: String(error),
    };
  }
}
