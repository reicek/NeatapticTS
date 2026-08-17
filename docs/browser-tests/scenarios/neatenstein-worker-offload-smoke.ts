import {
  createNeatensteinRendererBridge,
  type NeatensteinTier,
} from '../../../examples/neatenstein/browser-entry/host/renderer-bridge.ts';

interface TierResult {
  tier: NeatensteinTier;
  initialized: boolean;
  frameReceived: boolean;
  requestId: number | null;
  hasWallDistances: boolean;
  wallDistancesLength: number;
  format: string | null;
  timeout: boolean;
}

interface SmokeResult {
  success: boolean;
  browserVisibility: string;
  scenario: string;
  url: string;
  consoleErrorCount: number;
  gpuAdapterInfo: GPUAdapterInfo | { present: boolean; error?: string } | null;
  tiers: Record<NeatensteinTier, TierResult>;
  discrepancies: string[];
}

const WORKER_URL = './neatenstein-worker-offload-worker.bundle.mjs';
const SCENARIO = 'neatenstein-worker-offload-smoke';
const TIER_ORDER: NeatensteinTier[] = ['worker', 'cpu', 'gpu'];

function createResult(): SmokeResult {
  return {
    success: true,
    browserVisibility: 'visible-foreground',
    scenario: SCENARIO,
    url: location.href,
    consoleErrorCount: 0,
    gpuAdapterInfo: null,
    tiers: {} as Record<NeatensteinTier, TierResult>,
    discrepancies: [],
  };
}

function makeCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 360;
  return canvas;
}

function runTier(tier: NeatensteinTier): Promise<TierResult> {
  return new Promise((resolve) => {
    const result: TierResult = {
      tier,
      initialized: false,
      frameReceived: false,
      requestId: null,
      hasWallDistances: false,
      wallDistancesLength: 0,
      format: null,
      timeout: false,
    };

    const canvas = makeCanvas();
    const bridge = createNeatensteinRendererBridge({
      canvas,
      workerUrl: WORKER_URL,
      tier,
    });

    const timeoutId = window.setTimeout(() => {
      result.timeout = true;
      bridge.destroy();
      resolve(result);
    }, 5000);

    bridge.worker.onmessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data || typeof data !== 'object') return;

      if (data.type === 'initialized') {
        result.initialized = true;
        bridge.postSimState({ canvasWidth: 640, canvasHeight: 360, simTick: 1 });
      } else if (data.type === 'frame') {
        const frame = data.frame;
        result.frameReceived = true;
        result.requestId =
          frame && typeof frame.requestId === 'number' ? frame.requestId : null;
        result.format = frame?.format ?? null;
        result.hasWallDistances = frame?.wallDistances instanceof Float32Array;
        result.wallDistancesLength = frame?.wallDistances?.length ?? 0;
        window.clearTimeout(timeoutId);
        bridge.destroy();
        resolve(result);
      }
    };
  });
}

async function collectGpuAdapterInfo(): Promise<SmokeResult['gpuAdapterInfo']> {
  if (!('gpu' in navigator)) {
    return { present: false };
  }
  try {
    const adapter = await (navigator as NavigatorGPU).gpu.requestAdapter();
    if (!adapter) {
      return { present: true };
    }
    const info = await adapter.requestAdapterInfo();
    return info;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { present: true, error: message };
  }
}

async function main(): Promise<void> {
  const result = createResult();

  const originalError = console.error;
  console.error = (...args: unknown[]) => {
    result.consoleErrorCount += 1;
    originalError.apply(console, args);
  };
  window.addEventListener('error', () => {
    result.consoleErrorCount += 1;
  });

  for (const tier of TIER_ORDER) {
    result.tiers[tier] = await runTier(tier);
  }

  result.gpuAdapterInfo = await collectGpuAdapterInfo();

  for (const tier of TIER_ORDER) {
    const tierResult = result.tiers[tier];
    if (!tierResult.initialized) {
      result.success = false;
      result.discrepancies.push(`${tier}: init did not complete`);
    }
    if (!tierResult.frameReceived) {
      result.success = false;
      result.discrepancies.push(`${tier}: no frame message received`);
    }
    if (tierResult.requestId === null) {
      result.success = false;
      result.discrepancies.push(`${tier}: frame missing requestId`);
    }
    if (!tierResult.hasWallDistances) {
      result.success = false;
      result.discrepancies.push(`${tier}: frame missing wallDistances Float32Array`);
    }
    if (tierResult.timeout) {
      result.success = false;
      result.discrepancies.push(`${tier}: timed out waiting for frame`);
    }
  }

  const status = document.getElementById('status');
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }

  (window as unknown as Record<string, unknown>)[
    'neatensteinWorkerOffloadSmokeResult'
  ] = result;
}

main().catch((err) => {
  const errorResult = {
    success: false,
    scenario: SCENARIO,
    url: location.href,
    error: err instanceof Error ? err.message : String(err),
    stack: err instanceof Error ? err.stack : undefined,
  };
  (window as unknown as Record<string, unknown>)[
    'neatensteinWorkerOffloadSmokeResult'
  ] = errorResult;
  const status = document.getElementById('status');
  if (status) {
    status.textContent = JSON.stringify(errorResult, null, 2);
  }
});