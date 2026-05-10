import { Architect } from '../../../neataptic';
import {
  createGenomeFromNetwork,
  type NeatGenome,
} from '../../../neat/genome/genome';
import Network from '../network';
import {
  deserializeCompressedGenomeArchive,
  deserializeCompressedGenomeArchiveAsync,
  deserializeCompressedGenomeArchiveAsyncWithMetrics,
  deserializeCompressedGenomeArchiveWithMetrics,
  parseCompressedGenomeArchive,
  parseCompressedGenomeArchiveAsync,
  serializeCompressedGenomeArchive,
  serializeCompressedGenomeArchiveWithMetrics,
} from './network.serialize.genome.utils';

type NodeArchiveCompressionModule = {
  gunzipSync(input: Uint8Array): Uint8Array;
};

type ExtensionAwareNetwork = Network & {
  _reenableProb?: number;
};

function createExtensionAwareNetwork(seed: number): ExtensionAwareNetwork {
  const network = new Network(1, 1, { seed }) as ExtensionAwareNetwork;
  const outputNode = network.nodes.at(-1);

  if (!outputNode) {
    throw new Error('Expected one output node for genome archive coverage.');
  }

  network.connections[0].gain = 1.25;
  outputNode.response = 1.5;
  network._reenableProb = 0.6;

  return network;
}

function createExpectedStrictGenome(network: Network): NeatGenome {
  return createGenomeFromNetwork(network, {
    connectionGain: true,
    disabledConnectionReenableProbability: true,
    nodeResponse: true,
  });
}

function concatenateArchiveByteChunks(byteChunks: Uint8Array[]): Uint8Array {
  const totalByteLength = byteChunks.reduce((total, byteChunk) => {
    return total + byteChunk.length;
  }, 0);
  const concatenatedBytes = new Uint8Array(totalByteLength);
  let writeOffset = 0;

  byteChunks.forEach((byteChunk) => {
    concatenatedBytes.set(byteChunk, writeOffset);
    writeOffset += byteChunk.length;
  });

  return concatenatedBytes;
}

function createChunkedBrowserDecompressionStreamConstructor(): typeof DecompressionStream {
  const builtinModuleLoader = process.getBuiltinModule;

  if (typeof builtinModuleLoader !== 'function') {
    throw new Error(
      'Expected Node builtin loader while preparing the browser decode test shim.',
    );
  }

  const compressionModule = builtinModuleLoader(
    'node:zlib',
  ) as NodeArchiveCompressionModule;

  return class ChunkedBrowserDecompressionStream {
    public constructor(format: 'gzip') {
      if (format !== 'gzip') {
        throw new Error('Expected gzip format in the browser decode test shim.');
      }

      const compressedChunks: Uint8Array[] = [];

      return new TransformStream<Uint8Array, Uint8Array>({
        flush(controller) {
          const decompressedBytes = Uint8Array.from(
            compressionModule.gunzipSync(
              concatenateArchiveByteChunks(compressedChunks),
            ),
          );
          const firstChunkByteLength = Math.floor(
            decompressedBytes.length / 2,
          );

          if (firstChunkByteLength > 0) {
            controller.enqueue(
              decompressedBytes.subarray(0, firstChunkByteLength),
            );
          }

          if (firstChunkByteLength < decompressedBytes.length) {
            controller.enqueue(
              decompressedBytes.subarray(firstChunkByteLength),
            );
          }
        },
        transform(chunk) {
          compressedChunks.push(Uint8Array.from(chunk));
        },
      }) as unknown as ChunkedBrowserDecompressionStream;
    }
  } as unknown as typeof DecompressionStream;
}

describe('network serialize genome utilities chapter', () => {
  describe('serializeCompressedGenomeArchiveWithMetrics()', () => {
    describe('given one runtime carries extension-aware genome traits', () => {
      it('reports encode metrics while preserving the strict genome contract', () => {
        // Arrange
        const network = createExtensionAwareNetwork(8_100);
        const expectedGenome = createExpectedStrictGenome(network);

        // Act
        const archiveWithMetrics =
          serializeCompressedGenomeArchiveWithMetrics.call(network);
        const rebuiltGenome = parseCompressedGenomeArchive(
          archiveWithMetrics.archive,
        );

        // Assert
        expect({
          compressionRatioMatches:
            archiveWithMetrics.metrics.compressionRatio ===
            archiveWithMetrics.metrics.compressedByteLength /
              archiveWithMetrics.metrics.uncompressedByteLength,
          compressedByteLengthPositive:
            archiveWithMetrics.metrics.compressedByteLength > 0,
          encodeTimeMsFinite:
            Number.isFinite(archiveWithMetrics.metrics.encodeTimeMs) &&
            archiveWithMetrics.metrics.encodeTimeMs >= 0,
          rebuiltGenome,
          uncompressedByteLengthPositive:
            archiveWithMetrics.metrics.uncompressedByteLength > 0,
        }).toEqual({
          compressionRatioMatches: true,
          compressedByteLengthPositive: true,
          encodeTimeMsFinite: true,
          rebuiltGenome: expectedGenome,
          uncompressedByteLengthPositive: true,
        });
      });
    });
  });

  describe('serializeCompressedGenomeArchive()', () => {
    describe('given one runtime carries extension-aware genome traits', () => {
      it('round-trips the strict genome contract through the archive wrapper', () => {
        // Arrange
        const network = createExtensionAwareNetwork(8_101);
        const expectedGenome = createExpectedStrictGenome(network);

        // Act
        const rebuiltGenome = parseCompressedGenomeArchive(
          serializeCompressedGenomeArchive.call(network),
        );

        // Assert
        expect(rebuiltGenome).toEqual(expectedGenome);
      });
    });

    describe('given the zstd codec is requested explicitly', () => {
      it('round-trips the strict genome contract through the alternate archive codec', () => {
        // Arrange
        const network = createExtensionAwareNetwork(8_102);
        const expectedGenome = createExpectedStrictGenome(network);

        // Act
        const rebuiltGenome = parseCompressedGenomeArchive(
          serializeCompressedGenomeArchive.call(network, {
            compression: 'zstd',
          }),
        );

        // Assert
        expect(rebuiltGenome).toEqual(expectedGenome);
      });
    });
  });

  describe('parseCompressedGenomeArchive()', () => {
    describe('given the archive format tag is unknown', () => {
      it('throws an invalid-archive-format error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          parseCompressedGenomeArchive({
            compressedFormat: 'neat-genome-v1',
            compression: 'gzip',
            format: 'unsupported-genome-archive-format',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Invalid compressed genome archive format.',
        );
      });
    });

    describe('given the payload format tag is unknown', () => {
      it('throws an invalid-payload-format error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          parseCompressedGenomeArchive({
            compressedFormat: 'unsupported-genome-format',
            compression: 'gzip',
            format: 'neat-genome-archive-v1',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe('Invalid compressed genome payload format.');
      });
    });
  });

  describe('parseCompressedGenomeArchiveAsync()', () => {
    describe('given the archive format tag is unknown', () => {
      it('throws an invalid-archive-format error', async () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          await parseCompressedGenomeArchiveAsync({
            compressedFormat: 'neat-genome-v1',
            compression: 'gzip',
            format: 'unsupported-genome-archive-format',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Invalid compressed genome archive format.',
        );
      });
    });

    describe('given the payload format tag is unknown', () => {
      it('throws an invalid-payload-format error', async () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          await parseCompressedGenomeArchiveAsync({
            compressedFormat: 'unsupported-genome-format',
            compression: 'gzip',
            format: 'neat-genome-archive-v1',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe('Invalid compressed genome payload format.');
      });
    });
  });

  describe('deserializeCompressedGenomeArchive()', () => {
    describe('given one runtime carries output-affecting genome extensions', () => {
      it('reports decode metrics while rebuilding a runnable network with identical outputs', () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_099);
        const compressedArchive = serializeCompressedGenomeArchive.call(
          sourceNetwork,
        );
        const expectedOutput = sourceNetwork.activate(inputValues);

        // Act
        const rebuiltNetworkWithMetrics =
          deserializeCompressedGenomeArchiveWithMetrics(
            compressedArchive,
          ) as {
            metrics: {
              compressedByteLength: number;
              compressionRatio: number;
              decodeTimeMs: number;
              uncompressedByteLength: number;
            };
            value: ExtensionAwareNetwork;
          };

        // Assert
        expect({
          compressionRatioMatches:
            rebuiltNetworkWithMetrics.metrics.compressionRatio ===
            rebuiltNetworkWithMetrics.metrics.compressedByteLength /
              rebuiltNetworkWithMetrics.metrics.uncompressedByteLength,
          decodeTimeMsFinite:
            Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
            rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
          rebuiltOutput:
            rebuiltNetworkWithMetrics.value.activate(inputValues),
          response: rebuiltNetworkWithMetrics.value.nodes.at(-1)?.response ?? null,
        }).toEqual({
          compressionRatioMatches: true,
          decodeTimeMsFinite: true,
          rebuiltOutput: expectedOutput,
          response: 1.5,
        });
      });
    });

    describe('given one runtime carries output-affecting genome extensions', () => {
      it('rebuilds a runnable network with identical outputs and restored extension traits', () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_103);
        const expectedOutput = sourceNetwork.activate(inputValues);

        // Act
        const rebuiltNetwork = deserializeCompressedGenomeArchive(
          serializeCompressedGenomeArchive.call(sourceNetwork),
        ) as ExtensionAwareNetwork;

        // Assert
        expect({
          gain: rebuiltNetwork.connections[0].gain,
          output: rebuiltNetwork.activate(inputValues),
          reenableProb: rebuiltNetwork._reenableProb,
          response: rebuiltNetwork.nodes.at(-1)?.response ?? null,
        }).toEqual({
          gain: 1.25,
          output: expectedOutput,
          reenableProb: 0.6,
          response: 1.5,
        });
      });
    });

    describe('given one recurrent runtime carries temporal module descriptors', () => {
      it('preserves the extension bag when the archived genome is parsed', () => {
        // Arrange
        const sourceNetwork = Architect.lstm(1, 2, 1);
        const expectedGenome = createExpectedStrictGenome(sourceNetwork);

        // Act
        const rebuiltGenome = parseCompressedGenomeArchive(
          serializeCompressedGenomeArchive.call(sourceNetwork),
        );

        // Assert
        expect(rebuiltGenome.extensions).toEqual(expectedGenome.extensions);
      });
    });
  });

  describe('deserializeCompressedGenomeArchiveAsync()', () => {
    describe('given optional runtime hints and decode callbacks are omitted on the metrics helper', () => {
      it('rebuilds a runnable network through the default async metrics path', async () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_097);
        const compressedArchive = serializeCompressedGenomeArchive.call(
          sourceNetwork,
        );
        const expectedOutput = sourceNetwork.activate(inputValues);

        // Act
        const rebuiltNetworkWithMetrics =
          (await deserializeCompressedGenomeArchiveAsyncWithMetrics(
            compressedArchive,
          )) as {
            metrics: {
              decodeTimeMs: number;
              uncompressedByteLength: number;
            };
            value: ExtensionAwareNetwork;
          };

        // Assert
        expect({
          decodeTimeMsFinite:
            Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
            rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
          rebuiltOutput:
            rebuiltNetworkWithMetrics.value.activate(inputValues),
          uncompressedByteLengthPositive:
            rebuiltNetworkWithMetrics.metrics.uncompressedByteLength > 0,
        }).toEqual({
          decodeTimeMsFinite: true,
          rebuiltOutput: expectedOutput,
          uncompressedByteLengthPositive: true,
        });
      });
    });

    describe('given browser decompression emits multiple output chunks without a Node runtime', () => {
      it('reports decode metrics while preserving progress snapshots and output-affecting genome traits', async () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_098);
        const compressedArchive = serializeCompressedGenomeArchive.call(
          sourceNetwork,
        );
        const expectedOutput = sourceNetwork.activate(inputValues);
        const browserDecompressionStreamConstructor =
          createChunkedBrowserDecompressionStreamConstructor();
        const originalProcess = globalThis.process;
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );
        const progressSnapshots: Array<{ done: boolean }> = [];

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'DecompressionStream', {
          configurable: true,
          value: browserDecompressionStreamConstructor,
        });

        try {
          // Act
          const rebuiltNetworkWithMetrics =
            (await deserializeCompressedGenomeArchiveAsyncWithMetrics(
              compressedArchive,
              {},
              {
                onProgress(progressUpdate) {
                  progressSnapshots.push({
                    done: progressUpdate.done,
                  });
                },
              },
            )) as {
              metrics: {
                compressedByteLength: number;
                compressionRatio: number;
                decodeTimeMs: number;
                uncompressedByteLength: number;
              };
              value: ExtensionAwareNetwork;
            };

          // Assert
          expect({
            compressionRatioMatches:
              rebuiltNetworkWithMetrics.metrics.compressionRatio ===
              rebuiltNetworkWithMetrics.metrics.compressedByteLength /
                rebuiltNetworkWithMetrics.metrics.uncompressedByteLength,
            decodeTimeMsFinite:
              Number.isFinite(rebuiltNetworkWithMetrics.metrics.decodeTimeMs) &&
              rebuiltNetworkWithMetrics.metrics.decodeTimeMs >= 0,
            hasProgressSnapshots: progressSnapshots.length > 0,
            lastProgressDone: progressSnapshots.at(-1)?.done ?? false,
            rebuiltOutput:
              rebuiltNetworkWithMetrics.value.activate(inputValues),
          }).toEqual({
            compressionRatioMatches: true,
            decodeTimeMsFinite: true,
            hasProgressSnapshots: true,
            lastProgressDone: true,
            rebuiltOutput: expectedOutput,
          });
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
          Object.defineProperty(globalThis, 'DecompressionStream', {
            configurable: true,
            value: originalDecompressionStream,
          });
        }
      });
    });

    describe('given optional runtime hints and decode callbacks are omitted', () => {
      it('rebuilds a runnable network through the default async archive path', async () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_105);
        const compressedArchive = serializeCompressedGenomeArchive.call(
          sourceNetwork,
        );
        const expectedOutput = sourceNetwork.activate(inputValues);

        // Act
        const rebuiltNetwork =
          (await deserializeCompressedGenomeArchiveAsync(
            compressedArchive,
          )) as ExtensionAwareNetwork;

        // Assert
        expect(rebuiltNetwork.activate(inputValues)).toEqual(expectedOutput);
      });
    });

    describe('given browser decompression emits multiple output chunks without a Node runtime', () => {
      it('reports incremental decode progress while preserving output-affecting genome traits', async () => {
        // Arrange
        const inputValues = [0.35];
        const sourceNetwork = createExtensionAwareNetwork(8_104);
        const strictGenome = createExpectedStrictGenome(sourceNetwork);
        const compressedArchive = serializeCompressedGenomeArchive.call(
          sourceNetwork,
        );
        const expectedOutput = sourceNetwork.activate(inputValues);
        const expectedGenomeByteLength = new TextEncoder().encode(
          JSON.stringify(strictGenome),
        ).length;
        const expectedFirstChunkByteLength = Math.max(
          1,
          Math.floor(expectedGenomeByteLength / 2),
        );
        const browserDecompressionStreamConstructor =
          createChunkedBrowserDecompressionStreamConstructor();
        const originalProcess = globalThis.process;
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );
        const progressSnapshots: Array<{
          decodedByteLength: number;
          done: boolean;
        }> = [];

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'DecompressionStream', {
          configurable: true,
          value: browserDecompressionStreamConstructor,
        });

        try {
          // Act
          const rebuiltNetwork =
            (await deserializeCompressedGenomeArchiveAsync(
              compressedArchive,
              {},
              {
                onProgress(progressUpdate) {
                  progressSnapshots.push({
                    decodedByteLength: progressUpdate.decodedByteLength,
                    done: progressUpdate.done,
                  });
                },
              },
            )) as ExtensionAwareNetwork;

          // Assert
          expect({
            gain: rebuiltNetwork.connections[0].gain,
            output: rebuiltNetwork.activate(inputValues),
            progressSnapshots,
            reenableProb: rebuiltNetwork._reenableProb,
            response: rebuiltNetwork.nodes.at(-1)?.response ?? null,
          }).toEqual({
            gain: 1.25,
            output: expectedOutput,
            progressSnapshots: [
              {
                decodedByteLength: expectedFirstChunkByteLength,
                done: false,
              },
              {
                decodedByteLength: expectedGenomeByteLength,
                done: true,
              },
            ],
            reenableProb: 0.6,
            response: 1.5,
          });
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
          Object.defineProperty(globalThis, 'DecompressionStream', {
            configurable: true,
            value: originalDecompressionStream,
          });
        }
      });
    });
  });
});