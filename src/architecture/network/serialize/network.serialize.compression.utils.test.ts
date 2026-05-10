import type {
  CompressedSerializedNetwork,
  NetworkJSONConnection,
} from '../network.types';
import {
  COMPRESSED_NETWORK_ARCHIVE_FORMAT,
  COMPRESSED_WEIGHT_ENCODING,
  compressArchivePayloadBytes,
  compressSerializedConnections,
  createCompressedArchiveEncodeMetrics,
  createCompressedNetworkArchive,
  createCompressedNetworkArchiveAsync,
  decompressArchivePayloadBytesAsync,
  decompressSerializedConnections,
  estimateSerializedByteLength,
  parseCompressedNetworkArchive,
  parseCompressedNetworkArchiveAsync,
} from './network.serialize.compression.utils';

type NodeArchiveCompressionModule = {
  gunzipSync(input: Uint8Array): Uint8Array;
};

function createCompressedSerializedNetworkPayload(): CompressedSerializedNetwork {
  return {
    activations: [0.5, 0.75],
    architecture: {
      hasCycles: false,
      hiddenLayerSizes: [],
      source: 'graph-topology',
      totalConnections: 1,
      totalNodes: 2,
    },
    connections: compressSerializedConnections([
      {
        enabled: true,
        from: 0,
        gater: null,
        to: 1,
        weight: 0.5,
      },
    ]),
    dropout: 0,
    format: 'compact-compressed-v1',
    formatVersion: 4,
    input: 1,
    nodes: [
      {
        bias: 0,
        index: 0,
        squash: 'identity',
        type: 'input',
      },
      {
        bias: 0,
        index: 1,
        squash: 'identity',
        type: 'output',
      },
    ],
    output: 1,
    states: [0, 0],
    topologyIntent: 'feed-forward',
  };
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

describe('network serialize compression utilities chapter', () => {
  describe('parseCompressedNetworkArchiveAsync()', () => {
    describe('given browser decompression emits multiple output chunks without a Node runtime', () => {
      it('reports incremental decode progress while rebuilding the original payload', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const compressedArchive = createCompressedNetworkArchive(compressedPayload);
        const expectedPayloadByteLength = estimateSerializedByteLength(
          compressedPayload,
        );
        const expectedFirstChunkByteLength = Math.max(
          1,
          Math.floor(expectedPayloadByteLength / 2),
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
          const rebuiltPayload = await parseCompressedNetworkArchiveAsync(
            compressedArchive,
            {
              onProgress(progressUpdate) {
                progressSnapshots.push({
                  decodedByteLength: progressUpdate.decodedByteLength,
                  done: progressUpdate.done,
                });
              },
            },
          );

          // Assert
          expect({
            progressSnapshots,
            rebuiltPayload,
          }).toEqual({
            progressSnapshots: [
              {
                decodedByteLength: expectedFirstChunkByteLength,
                done: false,
              },
              {
                decodedByteLength: expectedPayloadByteLength,
                done: true,
              },
            ],
            rebuiltPayload: compressedPayload,
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

    describe('given browser decompression is used without explicit decode options', () => {
      it('rebuilds the original UTF-8 payload bytes through the default streamed path', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const payloadBytes = new TextEncoder().encode(
          JSON.stringify(compressedPayload),
        );
        const compressedBytes = compressArchivePayloadBytes(payloadBytes, 'gzip');
        const browserDecompressionStreamConstructor =
          createChunkedBrowserDecompressionStreamConstructor();
        const originalProcess = globalThis.process;
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );

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
          const rebuiltPayloadBytes = await decompressArchivePayloadBytesAsync(
            compressedBytes,
            'gzip',
          );

          // Assert
          expect(Array.from(rebuiltPayloadBytes)).toEqual(
            Array.from(payloadBytes),
          );
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

    describe('given browser decompression inflates an empty payload', () => {
      it('returns an empty byte buffer when the streamed decode finishes without chunks', async () => {
        // Arrange
        const payloadBytes = new Uint8Array(0);
        const compressedBytes = compressArchivePayloadBytes(payloadBytes, 'gzip');
        const browserDecompressionStreamConstructor =
          createChunkedBrowserDecompressionStreamConstructor();
        const originalProcess = globalThis.process;
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );

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
          const rebuiltPayloadBytes = await decompressArchivePayloadBytesAsync(
            compressedBytes,
            'gzip',
          );

          // Assert
          expect(Array.from(rebuiltPayloadBytes)).toEqual([]);
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

  describe('createCompressedNetworkArchiveAsync()', () => {
    describe('given browser compression streams are available without a Node runtime', () => {
      it('round-trips the original payload through the gzip archive wrapper', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalProcess = globalThis.process;

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const rebuiltPayload = await parseCompressedNetworkArchiveAsync(
            await createCompressedNetworkArchiveAsync(compressedPayload),
          );

          // Assert
          expect(rebuiltPayload).toEqual(compressedPayload);
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }
      });
    });

    describe('given browser compression streams are unavailable but Node exists', () => {
      it('falls back to the Node archive encoder', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalCompressionStream = Reflect.get(
          globalThis,
          'CompressionStream',
        );

        Object.defineProperty(globalThis, 'CompressionStream', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const rebuiltPayload = await parseCompressedNetworkArchiveAsync(
            await createCompressedNetworkArchiveAsync(compressedPayload),
          );

          // Assert
          expect(rebuiltPayload).toEqual(compressedPayload);
        } finally {
          Object.defineProperty(globalThis, 'CompressionStream', {
            configurable: true,
            value: originalCompressionStream,
          });
        }
      });
    });

    describe('given browser compression streams receive zstd without a Node runtime', () => {
      it('throws an unsupported-browser-compression error', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalProcess = globalThis.process;
        let errorMessage = '';

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            await createCompressedNetworkArchiveAsync(compressedPayload, {
              compression: 'zstd',
            });
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives only support gzip in browser runtimes.',
          );
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }
      });
    });

    describe('given no async gzip runtime is available', () => {
      it('throws an archive-runtime-required error', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalProcess = globalThis.process;
        const originalCompressionStream = Reflect.get(
          globalThis,
          'CompressionStream',
        );
        let errorMessage = '';

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'CompressionStream', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            await createCompressedNetworkArchiveAsync(compressedPayload);
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives require either Node.js zlib support or browser CompressionStream support.',
          );
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
          Object.defineProperty(globalThis, 'CompressionStream', {
            configurable: true,
            value: originalCompressionStream,
          });
        }
      });
    });
  });

  describe('createCompressedNetworkArchive()', () => {
    describe('given one compressed network payload is archived with default options', () => {
      it('emits gzip metadata around the base64 archive payload', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();

        // Act
        const compressedArchive = createCompressedNetworkArchive(
          compressedPayload,
        );

        // Assert
        expect({
          compressedFormat: compressedArchive.compressedFormat,
          compression: compressedArchive.compression,
          format: compressedArchive.format,
          payloadEncoding: compressedArchive.payloadEncoding,
        }).toEqual({
          compressedFormat: 'compact-compressed-v1',
          compression: 'gzip',
          format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
          payloadEncoding: 'base64',
        });
      });
    });

    describe('given one compressed network payload is archived with gzip', () => {
      it('round-trips the original payload through the archive wrapper', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();

        // Act
        const rebuiltPayload = parseCompressedNetworkArchive(
          createCompressedNetworkArchive(compressedPayload),
        );

        // Assert
        expect(rebuiltPayload).toEqual(compressedPayload);
      });
    });

    describe('given one compressed network payload is archived with zstd', () => {
      it('round-trips the original payload through the alternate codec', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();

        // Act
        const rebuiltPayload = parseCompressedNetworkArchive(
          createCompressedNetworkArchive(compressedPayload, {
            compression: 'zstd',
          }),
        );

        // Assert
        expect(rebuiltPayload).toEqual(compressedPayload);
      });
    });

    describe('given an unsupported compression label is requested', () => {
      it('throws an unsupported-archive-compression error', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        let errorMessage = '';

        // Act
        try {
          createCompressedNetworkArchive(compressedPayload, {
            compression: 'unsupported' as never,
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Unsupported compressed network archive compression.',
        );
      });
    });

    describe('given the runtime does not expose Node builtin modules', () => {
      it('throws a node-runtime-required error', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalGetBuiltinModule = process.getBuiltinModule;
        let errorMessage = '';

        Object.defineProperty(process, 'getBuiltinModule', {
          configurable: true,
          value: undefined,
        });

        // Act
        try {
          createCompressedNetworkArchive(compressedPayload);
        } catch (error) {
          errorMessage = (error as Error).message;
        } finally {
          Object.defineProperty(process, 'getBuiltinModule', {
            configurable: true,
            value: originalGetBuiltinModule,
          });
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed network archives require a Node.js runtime.',
        );
      });
    });

    describe('given the runtime does not expose a global process object', () => {
      it('throws a node-runtime-required error', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalProcess = globalThis.process;
        let errorMessage = '';

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        // Act
        try {
          createCompressedNetworkArchive(compressedPayload);
        } catch (error) {
          errorMessage = (error as Error).message;
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed network archives require a Node.js runtime.',
        );
      });
    });

    describe('given browser base64 helpers are unavailable but Buffer exists', () => {
      it('falls back to the Buffer encoder', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalBase64Encoder = Reflect.get(globalThis, 'btoa');

        Object.defineProperty(globalThis, 'btoa', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const compressedArchive = createCompressedNetworkArchive(
            compressedPayload,
          );

          // Assert
          expect(compressedArchive.payload.length > 0).toBe(true);
        } finally {
          Object.defineProperty(globalThis, 'btoa', {
            configurable: true,
            value: originalBase64Encoder,
          });
        }
      });
    });

    describe('given no runtime base64 encoder is available', () => {
      it('throws a base64-support-required error', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const originalBase64Encoder = Reflect.get(globalThis, 'btoa');
        const originalBuffer = Reflect.get(globalThis, 'Buffer');
        let errorMessage = '';

        Object.defineProperty(globalThis, 'btoa', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'Buffer', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            createCompressedNetworkArchive(compressedPayload);
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives require base64 support in the current runtime.',
          );
        } finally {
          Object.defineProperty(globalThis, 'btoa', {
            configurable: true,
            value: originalBase64Encoder,
          });
          Object.defineProperty(globalThis, 'Buffer', {
            configurable: true,
            value: originalBuffer,
          });
        }
      });
    });
  });

  describe('parseCompressedNetworkArchive()', () => {
    describe('given the archive format tag is unknown', () => {
      it('throws an invalid-archive-format error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          parseCompressedNetworkArchive({
            compressedFormat: 'compact-compressed-v1',
            compression: 'gzip',
            format: 'unsupported-compressed-archive-format',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Invalid compressed network archive format.',
        );
      });
    });

    describe('given the archive compression label is unknown', () => {
      it('throws an unsupported-archive-compression error during inflate', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          parseCompressedNetworkArchive({
            compressedFormat: 'compact-compressed-v1',
            compression: 'unsupported' as never,
            format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
            payload: '',
            payloadEncoding: 'base64',
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Unsupported compressed network archive compression.',
        );
      });
    });

    describe('given browser base64 helpers are unavailable but Buffer exists', () => {
      it('falls back to the Buffer decoder', () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const compressedArchive = createCompressedNetworkArchive(compressedPayload);
        const originalBase64Decoder = Reflect.get(globalThis, 'atob');

        Object.defineProperty(globalThis, 'atob', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const rebuiltPayload = parseCompressedNetworkArchive(compressedArchive);

          // Assert
          expect(rebuiltPayload).toEqual(compressedPayload);
        } finally {
          Object.defineProperty(globalThis, 'atob', {
            configurable: true,
            value: originalBase64Decoder,
          });
        }
      });
    });

    describe('given no runtime base64 decoder is available', () => {
      it('throws a base64-support-required error', () => {
        // Arrange
        const compressedArchive = {
          compressedFormat: 'compact-compressed-v1',
          compression: 'gzip' as const,
          format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
          payload: '',
          payloadEncoding: 'base64' as const,
        } as const;
        const originalBase64Decoder = Reflect.get(globalThis, 'atob');
        const originalBuffer = Reflect.get(globalThis, 'Buffer');
        let errorMessage = '';

        Object.defineProperty(globalThis, 'atob', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'Buffer', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            parseCompressedNetworkArchive(compressedArchive);
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives require base64 support in the current runtime.',
          );
        } finally {
          Object.defineProperty(globalThis, 'atob', {
            configurable: true,
            value: originalBase64Decoder,
          });
          Object.defineProperty(globalThis, 'Buffer', {
            configurable: true,
            value: originalBuffer,
          });
        }
      });
    });
  });

  describe('parseCompressedNetworkArchiveAsync()', () => {
    describe('given the archive format tag is unknown', () => {
      it('throws an invalid-archive-format error', async () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          await parseCompressedNetworkArchiveAsync({
            compressedFormat: 'compact-compressed-v1',
            compression: 'gzip',
            format: 'unsupported-compressed-archive-format',
            payload: '',
            payloadEncoding: 'base64',
          } as never);
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Invalid compressed network archive format.',
        );
      });
    });

    describe('given browser compression streams are unavailable but Node exists', () => {
      it('falls back to the Node archive decoder', async () => {
        // Arrange
        const compressedPayload = createCompressedSerializedNetworkPayload();
        const compressedArchive = createCompressedNetworkArchive(compressedPayload);
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );

        Object.defineProperty(globalThis, 'DecompressionStream', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          const rebuiltPayload = await parseCompressedNetworkArchiveAsync(
            compressedArchive,
          );

          // Assert
          expect(rebuiltPayload).toEqual(compressedPayload);
        } finally {
          Object.defineProperty(globalThis, 'DecompressionStream', {
            configurable: true,
            value: originalDecompressionStream,
          });
        }
      });
    });

    describe('given browser zstd inflate is requested without a Node runtime', () => {
      it('throws an unsupported-browser-compression error', async () => {
        // Arrange
        const originalProcess = globalThis.process;
        let errorMessage = '';

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            await parseCompressedNetworkArchiveAsync({
              compressedFormat: 'compact-compressed-v1',
              compression: 'zstd',
              format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
              payload: '',
              payloadEncoding: 'base64',
            });
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives only support gzip in browser runtimes.',
          );
        } finally {
          Object.defineProperty(globalThis, 'process', {
            configurable: true,
            value: originalProcess,
          });
        }
      });
    });

    describe('given no async gzip runtime is available', () => {
      it('throws an archive-runtime-required error', async () => {
        // Arrange
        const originalProcess = globalThis.process;
        const originalDecompressionStream = Reflect.get(
          globalThis,
          'DecompressionStream',
        );
        let errorMessage = '';

        Object.defineProperty(globalThis, 'process', {
          configurable: true,
          value: undefined,
        });
        Object.defineProperty(globalThis, 'DecompressionStream', {
          configurable: true,
          value: undefined,
        });

        try {
          // Act
          try {
            await parseCompressedNetworkArchiveAsync({
              compressedFormat: 'compact-compressed-v1',
              compression: 'gzip',
              format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
              payload: '',
              payloadEncoding: 'base64',
            });
          } catch (error) {
            errorMessage = (error as Error).message;
          }

          // Assert
          expect(errorMessage).toBe(
            'Compressed network archives require either Node.js zlib support or browser DecompressionStream support.',
          );
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

  describe('compressSerializedConnections()', () => {
    describe('given one connection collection contains contiguous disabled and zero-weight spans', () => {
      it('emits run-length metadata instead of a raw enabled-state vector', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            gater: null,
            to: 1,
            weight: 0.25,
          },
          {
            enabled: false,
            from: 1,
            gater: null,
            to: 2,
            weight: 0,
          },
          {
            enabled: false,
            from: 2,
            gater: null,
            to: 3,
            weight: 0,
          },
          {
            enabled: true,
            from: 3,
            gater: null,
            to: 4,
            weight: 0,
          },
          {
            enabled: true,
            from: 4,
            gater: null,
            to: 5,
            weight: -0.5,
          },
        ];

        // Act
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Assert
        expect({
          deltaWordCount: compressedConnections.weightWords.deltaWords.length,
          disabledRuns: compressedConnections.disabledRuns,
          enabledStatesPresent: compressedConnections.enabledStates !== undefined,
          zeroWeightRuns: compressedConnections.weightWords.zeroWeightRuns,
        }).toEqual({
          deltaWordCount: 4,
          disabledRuns: [{ length: 2, startIndex: 1 }],
          enabledStatesPresent: false,
          zeroWeightRuns: [{ length: 3, startIndex: 1 }],
        });
      });
    });

    describe('given one connection collection uses gain, gating, disabled flags, and historical ids', () => {
      it('round-trips the exact connection rows through the compressed payload', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            fromGeneId: 101,
            gain: 1.25,
            gater: 2,
            gaterGeneId: 202,
            innovation: 301,
            to: 1,
            toGeneId: 401,
            weight: 0.125,
          },
          {
            enabled: false,
            from: 1,
            fromGeneId: 402,
            gain: undefined,
            gater: null,
            gaterGeneId: undefined,
            innovation: 302,
            to: 2,
            toGeneId: 403,
            weight: -98765.4321,
          },
        ];
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Act
        const rebuiltConnections = decompressSerializedConnections(
          compressedConnections,
        );

        // Assert
        expect(rebuiltConnections).toEqual(serializedConnections);
      });
    });

    describe('given all optional fields stay neutral', () => {
      it('omits the optional compressed vectors', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            gater: null,
            to: 1,
            weight: 0.5,
          },
        ];

        // Act
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Assert
        expect({
          enabledStatesPresent: compressedConnections.enabledStates !== undefined,
          fromGeneIdsPresent: compressedConnections.fromGeneIds !== undefined,
          gainValuesPresent: compressedConnections.gainValues !== undefined,
          gaterGeneIdsPresent:
            compressedConnections.gaterGeneIds !== undefined,
          gaterIndicesPresent: compressedConnections.gaterIndices !== undefined,
          innovationIdsPresent:
            compressedConnections.innovationIds !== undefined,
          toGeneIdsPresent: compressedConnections.toGeneIds !== undefined,
        }).toEqual({
          enabledStatesPresent: false,
          fromGeneIdsPresent: false,
          gainValuesPresent: false,
          gaterGeneIdsPresent: false,
          gaterIndicesPresent: false,
          innovationIdsPresent: false,
          toGeneIdsPresent: false,
        });
      });

      it('rebuilds one neutral connection row without materializing optional metadata', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            gater: null,
            to: 1,
            weight: 0.5,
          },
        ];
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Act
        const rebuiltConnections = decompressSerializedConnections(
          compressedConnections,
        );

        // Assert
        expect(rebuiltConnections).toEqual(serializedConnections);
      });
    });

    describe('given no connections are present', () => {
      it('produces one empty compressed block that rebuilds to an empty list', () => {
        // Arrange
        const compressedConnections = compressSerializedConnections([]);

        // Act
        const rebuiltConnections = decompressSerializedConnections(
          compressedConnections,
        );

        // Assert
        expect({
          compressedConnections,
          rebuiltConnections,
        }).toEqual({
          compressedConnections: {
            connectionCount: 0,
            fromIndices: [],
            toIndices: [],
            weightWords: {
              deltaWords: [],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [],
            },
          },
          rebuiltConnections: [],
        });
      });
    });

    describe('given every serialized weight is exact positive zero', () => {
      it('rebuilds the full connection list from zero-weight run metadata alone', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            gater: null,
            to: 1,
            weight: 0,
          },
          {
            enabled: true,
            from: 1,
            gater: null,
            to: 2,
            weight: 0,
          },
          {
            enabled: true,
            from: 2,
            gater: null,
            to: 3,
            weight: 0,
          },
        ];
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Act
        const rebuiltConnections = decompressSerializedConnections(
          compressedConnections,
        );

        // Assert
        expect({
          compressedWeightWords: compressedConnections.weightWords,
          rebuiltConnections,
        }).toEqual({
          compressedWeightWords: {
            deltaWords: [],
            encoding: COMPRESSED_WEIGHT_ENCODING,
            firstWeightWords: [],
            zeroWeightRuns: [{ length: 3, startIndex: 0 }],
          },
          rebuiltConnections: serializedConnections,
        });
      });
    });

    describe('given exact positive-zero weights appear between non-zero weights', () => {
      it('rebuilds the mixed sequence from zero-weight runs plus non-zero deltas', () => {
        // Arrange
        const serializedConnections: NetworkJSONConnection[] = [
          {
            enabled: true,
            from: 0,
            gater: null,
            to: 1,
            weight: 0.25,
          },
          {
            enabled: true,
            from: 1,
            gater: null,
            to: 2,
            weight: 0,
          },
          {
            enabled: true,
            from: 2,
            gater: null,
            to: 3,
            weight: 0,
          },
          {
            enabled: true,
            from: 3,
            gater: null,
            to: 4,
            weight: -0.5,
          },
        ];
        const compressedConnections = compressSerializedConnections(
          serializedConnections,
        );

        // Act
        const rebuiltConnections = decompressSerializedConnections(
          compressedConnections,
        );

        // Assert
        expect(rebuiltConnections).toEqual(serializedConnections);
      });
    });
  });

  describe('decompressSerializedConnections()', () => {
    describe('given one disabled-run span exceeds the connection count', () => {
      it('throws an invalid-disabled-run-layout error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 2,
            disabledRuns: [{ length: 2, startIndex: 1 }],
            fromIndices: [0, 1],
            toIndices: [1, 2],
            weightWords: {
              deltaWords: [0, 0, 0, 0],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed connection field "disabledRuns" run layout is invalid.',
        );
      });
    });

    describe('given the weight encoding tag is unknown', () => {
      it('throws an unsupported-encoding error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 1,
            fromIndices: [0],
            toIndices: [1],
            weightWords: {
              deltaWords: [],
              encoding: 'unsupported' as never,
              firstWeightWords: [0, 0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe('Unsupported compressed weight encoding.');
      });
    });

    describe('given one required vector length is wrong', () => {
      it('throws an invalid-field-length error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 1,
            fromIndices: [],
            toIndices: [1],
            weightWords: {
              deltaWords: [],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed connection field "fromIndices" length is invalid.',
        );
      });
    });

    describe('given the base weight word count is wrong', () => {
      it('throws an invalid-base-word-count error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 1,
            fromIndices: [0],
            toIndices: [1],
            weightWords: {
              deltaWords: [],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed weight base word count is invalid.',
        );
      });
    });

    describe('given the delta word count is wrong', () => {
      it('throws an invalid-delta-word-count error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 2,
            fromIndices: [0, 1],
            toIndices: [1, 2],
            weightWords: {
              deltaWords: [0],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed weight delta word count is invalid.',
        );
      });
    });

    describe('given an all-zero payload still includes base words', () => {
      it('throws an invalid-base-word-count error for the zero-run-only path', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 2,
            fromIndices: [0, 1],
            toIndices: [1, 2],
            weightWords: {
              deltaWords: [],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0, 0],
              zeroWeightRuns: [{ length: 2, startIndex: 0 }],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed weight base word count is invalid.',
        );
      });
    });

    describe('given an all-zero payload still includes delta words', () => {
      it('throws an invalid-delta-word-count error for the zero-run-only path', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 2,
            fromIndices: [0, 1],
            toIndices: [1, 2],
            weightWords: {
              deltaWords: [0, 0, 0, 0],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [],
              zeroWeightRuns: [{ length: 2, startIndex: 0 }],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed weight delta word count is invalid.',
        );
      });
    });

    describe('given one zero-weight run exceeds the connection count', () => {
      it('throws an invalid-zero-run-layout error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 2,
            fromIndices: [0, 1],
            toIndices: [1, 2],
            weightWords: {
              deltaWords: [],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [0, 0, 0, 0],
              zeroWeightRuns: [{ length: 2, startIndex: 1 }],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed connection field "zeroWeightRuns" run layout is invalid.',
        );
      });
    });

    describe('given one disabled-run span overlaps the prior span', () => {
      it('throws an invalid-disabled-run-layout error', () => {
        // Arrange
        let errorMessage = '';

        // Act
        try {
          decompressSerializedConnections({
            connectionCount: 3,
            disabledRuns: [
              { length: 2, startIndex: 0 },
              { length: 1, startIndex: 1 },
            ],
            fromIndices: [0, 1, 2],
            toIndices: [1, 2, 3],
            weightWords: {
              deltaWords: [0, 0, 0, 0, 0, 0, 0, 0],
              encoding: COMPRESSED_WEIGHT_ENCODING,
              firstWeightWords: [1, 0, 0, 0],
            },
          });
        } catch (error) {
          errorMessage = (error as Error).message;
        }

        // Assert
        expect(errorMessage).toBe(
          'Compressed connection field "disabledRuns" run layout is invalid.',
        );
      });
    });
  });

  describe('estimateSerializedByteLength()', () => {
    describe('given one simple ASCII payload', () => {
      it('returns the UTF-8 byte length of the JSON form', () => {
        // Arrange
        const payload = { value: 'a' };

        // Act
        const byteLength = estimateSerializedByteLength(payload);

        // Assert
        expect(byteLength).toBe(13);
      });
    });
  });

  describe('createCompressedArchiveEncodeMetrics()', () => {
    describe('given the uncompressed payload is empty', () => {
      it('reports a zero compression ratio instead of dividing by zero', () => {
        // Arrange
        const uncompressedByteLength = 0;
        const compressedByteLength = 0;
        const encodeTimeMs = 1;

        // Act
        const metrics = createCompressedArchiveEncodeMetrics(
          uncompressedByteLength,
          compressedByteLength,
          encodeTimeMs,
        );

        // Assert
        expect(metrics).toEqual({
          compressedByteLength: 0,
          compressionRatio: 0,
          encodeTimeMs: 1,
          uncompressedByteLength: 0,
        });
      });
    });
  });
});