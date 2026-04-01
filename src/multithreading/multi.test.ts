import Multi from './multi';
import {
  ACTIVATION_FUNCTIONS,
  deserializeDataSet,
  serializeDataSet,
} from './multi.utils';
import { TestWorker as BrowserTestWorker } from './workers/browser/testworker';
import { TestWorker as NodeTestWorker } from './workers/node/testworker';
import { Workers } from './workers/workers';

describe('multithreading facade chapter', () => {
  describe('static shelves', () => {
    describe('given the root facade exposes the worker and activation contracts', () => {
      it('keeps those public references aligned with the lower-level shelves', () => {
        // Assert
        expect({
          workers: Multi.workers,
          activations: Multi.activations,
        }).toEqual({
          workers: Workers,
          activations: ACTIVATION_FUNCTIONS,
        });
      });
    });
  });

  describe('dataset facade helpers', () => {
    describe('given the caller round-trips a serialized dataset through the root facade', () => {
      it('matches the lower-level utility behavior exactly', () => {
        // Arrange
        const dataSet = [{ input: [1, 2], output: [3] }];

        // Act
        const facadeRoundTrip = Multi.deserializeDataSet(
          Multi.serializeDataSet(dataSet),
        );

        // Assert
        expect(facadeRoundTrip).toEqual(
          deserializeDataSet(serializeDataSet(dataSet)),
        );
      });
    });
  });

  describe('getBrowserTestWorker', () => {
    describe('given the browser wrapper module is available', () => {
      it('resolves to the browser worker class', async () => {
        // Assert
        await expect(Multi.getBrowserTestWorker()).resolves.toBe(
          BrowserTestWorker,
        );
      });
    });
  });

  describe('getNodeTestWorker', () => {
    describe('given the node wrapper module is available', () => {
      it('resolves to the node worker class', async () => {
        // Assert
        await expect(Multi.getNodeTestWorker()).resolves.toBe(NodeTestWorker);
      });
    });
  });
});
