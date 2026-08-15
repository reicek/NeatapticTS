// examples/neatenstein/browser-entry/constants.ts
var NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION = "neatenstein-frame-v1";

// examples/neatenstein/browser-entry/host/renderer-bridge.ts
function createNeatensteinRendererBridge(options) {
  const { canvas, workerUrl, tier } = options;
  const worker = new Worker(workerUrl, { type: "module" });
  let offscreen;
  const transferList = [];
  if (tier === "worker") {
    offscreen = canvas.transferControlToOffscreen();
    transferList.push(offscreen);
  }
  worker.postMessage(
    {
      type: "init",
      tier,
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      ...offscreen ? { canvas: offscreen } : {}
    },
    transferList
  );
  const bridge = {
    worker,
    requestId: 0,
    postSimState(state) {
      worker.postMessage({ type: "simState", state });
    },
    destroy() {
      worker.terminate();
    }
  };
  if (tier !== "worker") {
    worker.onmessage = (event) => {
      const data = event.data;
      if (data && typeof data === "object" && data.type === "frame" && data.frame && typeof data.frame.requestId === "number") {
        bridge.requestId = data.frame.requestId;
      }
    };
  }
  return bridge;
}

// docs/browser-tests/scenarios/neatenstein-worker-offload-smoke.ts
var WORKER_URL = "./neatenstein-worker-offload-worker.bundle.mjs";
var SCENARIO = "neatenstein-worker-offload-smoke";
var TIER_ORDER = ["worker", "cpu", "gpu"];
function createResult() {
  return {
    success: true,
    browserVisibility: "visible-foreground",
    scenario: SCENARIO,
    url: location.href,
    consoleErrorCount: 0,
    gpuAdapterInfo: null,
    tiers: {},
    discrepancies: []
  };
}
function makeCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 360;
  return canvas;
}
function runTier(tier) {
  return new Promise((resolve) => {
    const result = {
      tier,
      initialized: false,
      frameReceived: false,
      requestId: null,
      hasWallDistances: false,
      wallDistancesLength: 0,
      format: null,
      timeout: false
    };
    const canvas = makeCanvas();
    const bridge = createNeatensteinRendererBridge({
      canvas,
      workerUrl: WORKER_URL,
      tier
    });
    const timeoutId = window.setTimeout(() => {
      result.timeout = true;
      bridge.destroy();
      resolve(result);
    }, 5e3);
    bridge.worker.onmessage = (event) => {
      const data = event.data;
      if (!data || typeof data !== "object") return;
      if (data.type === "initialized") {
        result.initialized = true;
        bridge.postSimState({ canvasWidth: 640, canvasHeight: 360, simTick: 1 });
      } else if (data.type === "frame") {
        const frame = data.frame;
        result.frameReceived = true;
        result.requestId = frame && typeof frame.requestId === "number" ? frame.requestId : null;
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
async function collectGpuAdapterInfo() {
  if (!("gpu" in navigator)) {
    return { present: false };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
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
async function main() {
  const result = createResult();
  const originalError = console.error;
  console.error = (...args) => {
    result.consoleErrorCount += 1;
    originalError.apply(console, args);
  };
  window.addEventListener("error", () => {
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
  const status = document.getElementById("status");
  if (status) {
    status.textContent = JSON.stringify(result, null, 2);
  }
  window["neatensteinWorkerOffloadSmokeResult"] = result;
}
main().catch((err) => {
  const errorResult = {
    success: false,
    scenario: SCENARIO,
    url: location.href,
    error: err instanceof Error ? err.message : String(err),
    stack: err instanceof Error ? err.stack : void 0
  };
  window["neatensteinWorkerOffloadSmokeResult"] = errorResult;
  const status = document.getElementById("status");
  if (status) {
    status.textContent = JSON.stringify(errorResult, null, 2);
  }
});
//# sourceMappingURL=neatenstein-worker-offload-smoke.bundle.mjs.map
