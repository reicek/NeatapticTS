import Network from '../../../src/architecture/network/network';
import { createBatchInferenceQueue } from '../../../src/architecture/network/gpu/network.gpu.batched';

const ACTIVATION_KEY_SYMBOL = Symbol.for('neataptic.activation.key');

declare global {
  interface Window {
    Neataptic?: {
      methods?: {
        Activation?: {
          logistic?: (value: number, derivate?: boolean) => number;
        };
      };
    };
    batchedQueueSmokeResult: Record<string, unknown> | null;
  }
}

function makeGPULogisticActivation(): (value: number, derivate?: boolean) => number {
  const workerKey = 'logisticActivation2';
  const publicLogistic = window.Neataptic?.methods?.Activation?.logistic;

  function logisticActivation2(value: number, derivate = false): number {
    if (typeof publicLogistic === 'function') {
      return publicLogistic(value, derivate);
    }
    // Fallback inline logistic if the IIFE bundle is somehow absent.
    const fx = 1 / (1 + Math.exp(-value));
    return derivate ? fx * (1 - fx) : fx;
  }

  (logisticActivation2 as unknown as Record<symbol, string>)[ACTIVATION_KEY_SYMBOL] = workerKey;

  try {
    Object.defineProperty(logisticActivation2, 'name', {
      value: workerKey,
      configurable: true,
    });
  } catch (_) {
    // ignore
  }

  return logisticActivation2;
}

function buildEligibleNetwork(): Network {
  const network = Network.createMLP(2, [3], 1);
  const gpuLogistic = makeGPULogisticActivation();

  for (const node of network.nodes) {
    if (typeof node.squash === 'function') {
      node.squash = gpuLogistic;
    }
  }

  return network;
}

async function readAdapterInfo(adapter: GPUAdapter | null): Promise<Record<string, unknown> | null> {
  if (!adapter) {
    return null;
  }

  try {
    if (typeof adapter.requestAdapterInfo === 'function') {
      return (await adapter.requestAdapterInfo()) as Record<string, unknown>;
    }

    if (adapter.info && typeof adapter.info === 'object') {
      return {
        vendor: (adapter.info as GPUAdapterInfo).vendor ?? null,
        architecture: (adapter.info as GPUAdapterInfo).architecture ?? null,
        device: (adapter.info as GPUAdapterInfo).device ?? null,
        description: (adapter.info as GPUAdapterInfo).description ?? null,
      };
    }
  } catch (error) {
    return { error: String(error) };
  }

  return null;
}

function checkBrowserVisibility(): {
  visible: boolean;
  browserVisibility: string;
  visibilityState: string;
  hasFocus: boolean;
  reason?: string;
} {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return {
      visible: false,
      browserVisibility: 'not-a-browser-window',
      visibilityState: 'unknown',
      hasFocus: false,
      reason: 'No browser window context available.',
    };
  }

  const visibilityState = document.visibilityState ?? 'unknown';
  const hasFocus = typeof document.hasFocus === 'function' ? document.hasFocus() : false;
  const hidden = document.hidden ?? visibilityState !== 'visible';

  if (hidden || visibilityState !== 'visible') {
    return {
      visible: false,
      browserVisibility: visibilityState,
      visibilityState,
      hasFocus,
      reason: `Document visibilityState is ${visibilityState} and document.hidden is ${hidden}.`,
    };
  }

  if (!hasFocus) {
    return {
      visible: false,
      browserVisibility: 'not-focused',
      visibilityState,
      hasFocus,
      reason: 'Window is visible but does not have focus.',
    };
  }

  return {
    visible: true,
    browserVisibility: 'visible-foreground',
    visibilityState,
    hasFocus,
  };
}

async function runBatchedQueueSmoke(): Promise<Record<string, unknown>> {
  const visibility = checkBrowserVisibility();
  if (!visibility.visible) {
    return {
      success: false,
      error: visibility.reason,
      browserVisibility: visibility.browserVisibility,
      scenarioUrl: typeof window !== 'undefined' ? window.location.href : '',
    };
  }

  const startTime = performance.now();

  try {
    const networkA = buildEligibleNetwork();
    const networkB = buildEligibleNetwork();

    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    const gpuAdapterInfo = await readAdapterInfo(adapter);
    const gpuDeviceBound = Boolean(device);

    if (!device) {
      return {
        success: false,
        error: 'WebGPU device not available.',
        gpuDeviceBound: false,
        gpuAdapterInfo,
        browserVisibility: visibility.browserVisibility,
        durationMs: performance.now() - startTime,
        scenarioUrl: window.location.href,
      };
    }

    networkA.gpuDevice = device;
    networkB.gpuDevice = device;

    const queue = createBatchInferenceQueue(device);
    const inputA = [0.1, 0.2];
    const inputB = [0.3, 0.4];
    const idA = queue.enqueue({ network: networkA, inputs: inputA });
    const idB = queue.enqueue({ network: networkB, inputs: inputB });
    const sizeBeforeFlush = queue.size;
    const jobIdsMatch = idA === 0 && idB === 1 && sizeBeforeFlush === 2;

    const cpuOutputA = networkA.activate(inputA);
    const cpuOutputB = networkB.activate(inputB);

    const originalSubmit = device.queue.submit.bind(device.queue);
    let submitCount = 0;
    device.queue.submit = (...args: unknown[]) => {
      submitCount += 1;
      return originalSubmit(...args);
    };

    let gpuOutputs: Float32Array[] = [];
    let flushError: string | null = null;
    try {
      device.pushErrorScope('validation');
      gpuOutputs = await queue.flush();
      const validationError = await device.popErrorScope();
      if (validationError) {
        flushError = validationError.message ?? String(validationError);
      }
    } catch (error) {
      flushError = String(error);
    }

    const gpuOutputA = gpuOutputs[0] ? Array.from(gpuOutputs[0]) : [];
    const gpuOutputB = gpuOutputs[1] ? Array.from(gpuOutputs[1]) : [];

    let maxAbsDiff = 0;
    let totalAbsDiff = 0;
    let elementCount = 0;

    const compare = (cpu: number[], gpu: number[]) => {
      for (let index = 0; index < cpu.length; index += 1) {
        const diff = Math.abs(cpu[index] - (gpu[index] ?? 0));
        maxAbsDiff = Math.max(maxAbsDiff, diff);
        totalAbsDiff += diff;
        elementCount += 1;
      }
    };

    compare(cpuOutputA, gpuOutputA);
    compare(cpuOutputB, gpuOutputB);

    const meanAbsDiff = elementCount > 0 ? totalAbsDiff / elementCount : 0;

    const orderedOutputsMatch =
      gpuOutputA.length === cpuOutputA.length &&
      gpuOutputB.length === cpuOutputB.length &&
      gpuOutputA.every((value, index) => Math.abs(value - cpuOutputA[index]) <= 0.5) &&
      gpuOutputB.every((value, index) => Math.abs(value - cpuOutputB[index]) <= 0.5);

    const emptyQueue = createBatchInferenceQueue(device);
    const emptyFlushResult = await emptyQueue.flush();
    const emptyFlushOk = Array.isArray(emptyFlushResult) && emptyFlushResult.length === 0;

    const durationMs = performance.now() - startTime;

    const success =
      gpuDeviceBound &&
      jobIdsMatch &&
      submitCount === 1 &&
      orderedOutputsMatch &&
      emptyFlushOk &&
      !flushError &&
      maxAbsDiff <= 0.5 &&
      meanAbsDiff <= 0.1;

    return {
      success,
      gpuDeviceBound,
      gpuAdapterInfo,
      browserVisibility: visibility.browserVisibility,
      jobIdsMatch,
      submitCount,
      orderedOutputsMatch,
      emptyFlushOk,
      maxAbsDiff,
      meanAbsDiff,
      durationMs,
      scenarioUrl: window.location.href,
      cpuOutputA,
      gpuOutputA,
      cpuOutputB,
      gpuOutputB,
      sizeBeforeFlush,
      flushError,
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
      browserVisibility: visibility.browserVisibility,
      durationMs: performance.now() - startTime,
      scenarioUrl: typeof window !== 'undefined' ? window.location.href : '',
    };
  }
}

if (typeof window !== 'undefined') {
  window.batchedQueueSmokeResult = null;
  void runBatchedQueueSmoke().then((result) => {
    window.batchedQueueSmokeResult = result;
    const outputElement = document.getElementById('result');
    if (outputElement) {
      outputElement.textContent = JSON.stringify(result, null, 2);
    }
  });
}
