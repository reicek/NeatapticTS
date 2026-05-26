import type { TestWorkerConstructor } from '../../multithreading/types';

const NODE_TEST_WORKER_UNAVAILABLE_MESSAGE =
  'Node test workers are unavailable in browser builds.';

/**
 * Resolve the browser worker wrapper from the browser-oriented environment shelf.
 *
 * @returns Browser worker constructor.
 */
export async function getBrowserTestWorker(): Promise<TestWorkerConstructor> {
  const { TestWorker } =
    await import('../../multithreading/workers/browser/testworker');
  return TestWorker;
}

/**
 * Reject Node worker loading from the browser-oriented environment shelf.
 *
 * Browser builds should fail with a clear environment error rather than trying
 * to pull `child_process` or `path` into the bundle.
 *
 * @returns Rejected promise describing the unsupported Node worker request.
 */
export async function getNodeTestWorker(): Promise<TestWorkerConstructor> {
  throw new Error(NODE_TEST_WORKER_UNAVAILABLE_MESSAGE);
}
