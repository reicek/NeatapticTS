import { getBrowserTestWorker, getNodeTestWorker } from './index';
import {
  getBrowserTestWorker as getBrowserAdapterBrowserTestWorker,
  getNodeTestWorker as getBrowserAdapterNodeTestWorker,
} from './browser/worker-loader';
import {
  getBrowserTestWorker as getNodeAdapterBrowserTestWorker,
  getNodeTestWorker as getNodeAdapterNodeTestWorker,
} from './node/worker-loader';
import { TestWorker as BrowserTestWorker } from '../multithreading/workers/browser/testworker';
import { TestWorker as NodeTestWorker } from '../multithreading/workers/node/testworker';

describe('environment worker adapters', () => {
  it('re-exports the browser worker loader through the default environment entry', async () => {
    await expect(getBrowserTestWorker()).resolves.toBe(BrowserTestWorker);
  });

  it('re-exports the node worker loader through the default environment entry', async () => {
    await expect(getNodeTestWorker()).resolves.toBe(NodeTestWorker);
  });

  it('keeps the node adapter browser loader available', async () => {
    await expect(getNodeAdapterBrowserTestWorker()).resolves.toBe(
      BrowserTestWorker,
    );
  });

  it('keeps the node adapter node loader available', async () => {
    await expect(getNodeAdapterNodeTestWorker()).resolves.toBe(NodeTestWorker);
  });

  it('keeps the browser adapter browser loader available', async () => {
    await expect(getBrowserAdapterBrowserTestWorker()).resolves.toBe(
      BrowserTestWorker,
    );
  });

  it('rejects node worker loading in the browser adapter', async () => {
    await expect(getBrowserAdapterNodeTestWorker()).rejects.toThrow(
      'Node test workers are unavailable in browser builds.',
    );
  });
});
