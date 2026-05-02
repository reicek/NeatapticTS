import type { TestWorkerConstructor } from '../../multithreading/types';

/**
 * Resolve the browser worker wrapper from the Node-oriented environment shelf.
 *
 * Keeping this available preserves the current Jest and jsdom validation path,
 * where browser worker code is exercised inside a Node host runtime.
 *
 * @returns Browser worker constructor.
 */
export async function getBrowserTestWorker(): Promise<TestWorkerConstructor> {
  const { TestWorker } =
    await import('../../multithreading/workers/browser/testworker');
  return TestWorker;
}

/**
 * Resolve the Node worker wrapper from the Node-oriented environment shelf.
 *
 * @returns Node worker constructor.
 */
export async function getNodeTestWorker(): Promise<TestWorkerConstructor> {
  const { TestWorker } =
    await import('../../multithreading/workers/node/testworker');
  return TestWorker;
}
