import { describe, expect, it, jest } from '@jest/globals';
import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';
import {
  loadModule,
  workerSelf,
  sendInitMessage,
  sendSimStateMessage,
  sendRawMessage,
  sendResizeMessage,
  createMockCanvas,
  createMockContext,
  findPostByType,
  workerBeforeEach,
} from './display.worker.test-helpers';

describe('Neatenstein display worker', () => {
  workerBeforeEach();

  it('acknowledges an init message with the received tier and version', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    const initializedCall = findPostByType<{
      type: string;
      tier: string;
      version: string;
    }>(workerSelf.postMessage, 'initialized');
    expect(initializedCall).toEqual({
      type: 'initialized',
      tier: 'cpu',
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
  });

  it('AC-024: creates an MLP enemy population on init', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyPopulation?(): {
        kind: string;
        size: number;
        snapshot: () => { kind: string; weights: Float32Array };
      } | null;
    };

    sendInitMessage('cpu');
    const population = workerModule.__testOnlyGetEnemyPopulation?.();
    expect(population).not.toBeNull();
    expect(population?.kind).toBe('mlp');
    expect(population?.size).toBeGreaterThan(0);

    // The initial snapshot should have valid champion weights.
    const snapshot = population?.snapshot();
    expect(snapshot?.kind).toBe('mlp');
    expect(snapshot?.weights).toBeInstanceOf(Float32Array);
    expect(snapshot?.weights.length).toBeGreaterThan(0);
  });

  it('AC-024: enemy population is null before init', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetEnemyPopulation?(): unknown | null;
    };

    // Before init, the population should be null.
    const beforeInit = workerModule.__testOnlyGetEnemyPopulation?.();
    expect(beforeInit).toBeNull();
  });

  it('does not acknowledge init with an unrecognised tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({ type: 'init', tier: 'bad' });

    expect(
      findPostByType(workerSelf.postMessage, 'initialized'),
    ).toBeUndefined();
  });

  it('initializes with the default seed when mapSeed is omitted', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({
      type: 'init',
      tier: 'cpu',
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
    workerSelf.postMessage.mockClear();

    sendSimStateMessage();

    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeDefined();
  });

  it('uses the default version when the init message omits version', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendRawMessage({ type: 'init', tier: 'cpu', mapSeed: 42 });

    const initializedCall = findPostByType(
      workerSelf.postMessage,
      'initialized',
    );
    expect(initializedCall).toEqual(
      expect.objectContaining({
        type: 'initialized',
        tier: 'cpu',
        version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      }),
    );
  });

  it('does not render a worker frame when no canvas is transferred', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('worker');

    expect(() => sendSimStateMessage()).not.toThrow();
    expect(findPostByType(workerSelf.postMessage, 'frame')).toBeUndefined();
  });

  it('resizes the worker canvas to match the host-provided render size', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = {
      getContext: jest.fn(() => context),
      width: 100,
      height: 100,
    };
    sendInitMessage('worker', canvas);

    sendSimStateMessage();

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('resizes the worker canvas immediately when a resize message is received', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendResizeMessage(1024, 768);

    expect(canvas.width).toBe(1024);
    expect(canvas.height).toBe(768);
  });

  it('patches latestState dimensions when a resize message is received', async () => {
    jest.resetModules();
    const workerModule = (await loadModule('./display.worker.ts')) as {
      __testOnlyGetLatestState?():
        import('../renderer/frame').NeatensteinRenderState | null;
    };
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);
    sendSimStateMessage();
    workerSelf.postMessage.mockClear();

    sendResizeMessage(1024, 768);

    const latestState = workerModule.__testOnlyGetLatestState?.();
    expect(latestState?.canvasWidth).toBe(1024);
    expect(latestState?.canvasHeight).toBe(768);
  });

  it('ignores a resize message with invalid dimensions', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendResizeMessage(Number.NaN, Number.NaN);

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('ignores a resize message with non-number dimensions', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendRawMessage({ type: 'resize', width: '1024', height: '768' });

    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(360);
  });

  it('applies a pending resize to the worker canvas when init arrives later', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);

    sendResizeMessage(1024, 768);
    sendInitMessage('worker', canvas);

    expect(canvas.width).toBe(1024);
    expect(canvas.height).toBe(768);
  });

  it('reuses the worker 2D context across frames', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context } = createMockContext();
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendSimStateMessage();
    sendSimStateMessage();

    expect(canvas.getContext).toHaveBeenCalledTimes(1);
  });

  it('calls commit() on the worker 2D context when available', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const { context, putImageData } = createMockContext();
    const commit = jest.fn();
    (context as unknown as { commit: typeof commit }).commit = commit;
    const canvas = createMockCanvas(context);
    sendInitMessage('worker', canvas);

    sendSimStateMessage();

    expect(putImageData).toHaveBeenCalledTimes(1);
    expect(commit).toHaveBeenCalledTimes(1);
  });

  it('does not crash when the canvas returns no 2D context', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const canvas = {
      getContext: jest.fn(() => null),
      width: 640,
      height: 360,
    };
    sendInitMessage('worker', canvas);

    expect(() => sendSimStateMessage()).not.toThrow();
  });

  it('ignores non-object messages', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.postMessage.mock.calls.length;

    sendRawMessage('not an object');

    expect(workerSelf.postMessage.mock.calls.length).toBe(before);
  });

  it('ignores messages with an unknown type', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.postMessage.mock.calls.length;

    sendRawMessage({ type: 'unknown' });

    expect(workerSelf.postMessage.mock.calls.length).toBe(before);
  });
});
