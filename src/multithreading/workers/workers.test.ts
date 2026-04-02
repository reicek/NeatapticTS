import { TestWorker as BrowserTestWorker } from './browser/testworker';
import { TestWorker as NodeTestWorker } from './node/testworker';
import { Workers } from './workers';

describe('multithreading worker loader chapter', () => {
  describe('getBrowserTestWorker', () => {
    describe('given the browser worker wrapper module is available', () => {
      it('resolves to the browser worker class', async () => {
        // Assert
        await expect(Workers.getBrowserTestWorker()).resolves.toBe(
          BrowserTestWorker,
        );
      });
    });
  });

  describe('getNodeTestWorker', () => {
    describe('given the node worker wrapper module is available', () => {
      it('resolves to the node worker class', async () => {
        // Assert
        await expect(Workers.getNodeTestWorker()).resolves.toBe(NodeTestWorker);
      });
    });
  });
});
