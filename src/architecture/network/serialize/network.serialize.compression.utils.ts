import type {
  CompressedSerializedConnectionBlock,
  CompressedSerializedNetwork,
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
  CompressedSerializedConnectionWeights,
  CompressedSerializedIndexRun,
  NetworkJSONConnection,
} from '../network.types';

/** Stable format tag identifying the compressed compact serialization payload version consumed by decompression utilities. */
export const COMPRESSED_NETWORK_FORMAT = 'compact-compressed-v1';

/** Stable format tag identifying the compressed archive wrapper payload version used by archive encode and decode utilities. */
export const COMPRESSED_NETWORK_ARCHIVE_FORMAT =
  'compact-compressed-archive-v1';

/** Base64 string encoding applied to archived compressed bytes for safe JSON transport and storage of binary payloads. */
export const COMPRESSED_NETWORK_ARCHIVE_ENCODING = 'base64';

/** Stable format tag identifying the IEEE-754 float64 signed-int16 delta encoding used for compact weight storage. */
export const COMPRESSED_WEIGHT_ENCODING = 'ieee754-f64-int16-delta-v1';

/** Default Node-side archive compression codec. */
const DEFAULT_COMPRESSED_NETWORK_ARCHIVE_COMPRESSION = 'gzip';

/** Sentinel used when a serialized connection has no gater node. */
const NULL_GATER_INDEX_SENTINEL = -1;

/** Number of signed 16-bit words used to represent one IEEE-754 float64 value. */
const FLOAT64_INT16_WORD_COUNT = 4;

/** Offset used to normalize signed 16-bit wrapped arithmetic. */
const SIGNED_INT16_WRAP_OFFSET = 0x8000;

/** Bit mask used to clamp wrapped arithmetic to 16 bits. */
const SIGNED_INT16_WRAP_MASK = 0xffff;

type ProcessWithBuiltinModule = NodeJS.Process & {
  getBuiltinModule?: (moduleName: string) => unknown;
};

type NodeCompressionModule = {
  gunzipSync(input: Uint8Array): Uint8Array;
  gzipSync(input: Uint8Array): Uint8Array;
  zstdCompressSync(input: Uint8Array): Uint8Array;
  zstdDecompressSync(input: Uint8Array): Uint8Array;
};

type BrowserCompressionFormat = 'gzip';

type BrowserCompressionStreamConstructor = new (
  format: BrowserCompressionFormat,
) => CompressionStream;

type BrowserDecompressionStreamConstructor = new (
  format: BrowserCompressionFormat,
) => DecompressionStream;

/** Shared byte-size and compression-ratio metrics recorded for a single archive encode or decode operation. */
export interface CompressedArchiveMetrics {
  /** Binary archive byte length after compression and before base64 wrapping. */
  compressedByteLength: number;
  /** Archive byte length divided by the uncompressed UTF-8 payload byte length. */
  compressionRatio: number;
  /** UTF-8 byte length before archive compression. */
  uncompressedByteLength: number;
}

/** Encode metrics extending the shared archive metrics with elapsed encode time in milliseconds. */
export interface CompressedArchiveEncodeMetrics extends CompressedArchiveMetrics {
  /** Elapsed encode time in milliseconds. */
  encodeTimeMs: number;
}

/** Decode metrics extending the shared archive metrics with elapsed decode time in milliseconds. */
export interface CompressedArchiveDecodeMetrics extends CompressedArchiveMetrics {
  /** Elapsed decode time in milliseconds. */
  decodeTimeMs: number;
}

/** Typed result wrapper pairing the encoded archive payload with its byte-size and timing metrics. */
export interface CompressedArchiveEncodeResult<Archive> {
  /** Archived payload emitted by the encode operation. */
  archive: Archive;
  /** Byte-size and timing metrics for the encode operation. */
  metrics: CompressedArchiveEncodeMetrics;
}

/** Typed result wrapper pairing the decoded value with byte-size and timing metrics from the decode operation. */
export interface CompressedArchiveDecodeResult<Value> {
  /** Decoded value rebuilt from the archive payload. */
  value: Value;
  /** Byte-size and timing metrics for the decode operation. */
  metrics: CompressedArchiveDecodeMetrics;
}

/** Progress snapshot emitted incrementally while an archive payload is being decoded so callers can surface decode progress. */
export interface CompressedArchiveDecodeProgress {
  /** Size of the most recently decoded UTF-8 chunk. */
  chunkByteLength: number;
  /** Total decoded UTF-8 bytes emitted so far. */
  decodedByteLength: number;
  /** Total compressed archive bytes read from the wrapper payload. */
  encodedByteLength: number;
  /** Whether this update represents the final decoded chunk. */
  done: boolean;
}

/** Optional progress and lifecycle callbacks supplied by the caller while an archive payload is being decoded. */
export interface CompressedArchiveDecodeOptions {
  /**
   * Called after each decoded chunk so browser callers can surface progress.
   *
   * Node fallback paths still emit one completed snapshot so callers do not
   * need runtime-specific branching.
   */
  onProgress?: (
    progress: CompressedArchiveDecodeProgress,
  ) => void | Promise<void>;
}

/**
 * Archive one compressed network payload with a Node-side binary codec.
 *
 * This is intentionally additive: the wrapped payload stays the exact JSON form
 * returned by `serializeCompressed`, then gzip or zstd is applied above it.
 *
 * @param compressedPayload - Existing compressed network payload.
 * @param options - Optional archive compression settings.
 * @returns Base64-wrapped compressed archive payload.
 */
export function createCompressedNetworkArchive(
  compressedPayload: CompressedSerializedNetwork,
  options: CompressedSerializedNetworkArchiveOptions = {},
): CompressedSerializedNetworkArchive {
  const compression =
    options.compression ?? DEFAULT_COMPRESSED_NETWORK_ARCHIVE_COMPRESSION;

  // Step 1: Encode the exact compressed payload JSON into UTF-8 bytes.
  const payloadBytes = new TextEncoder().encode(
    JSON.stringify(compressedPayload),
  );

  // Step 2: Apply the requested Node-side archive codec.
  const compressedBytes = compressArchivePayloadBytes(
    payloadBytes,
    compression,
  );

  // Step 3: Return a JSON-safe base64 wrapper for storage and transport.
  return {
    compressedFormat: compressedPayload.format,
    compression,
    format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
    payload: encodeArchivePayloadBase64(compressedBytes),
    payloadEncoding: COMPRESSED_NETWORK_ARCHIVE_ENCODING,
  };
}

/**
 * Archive one compressed network payload with the best available async runtime codec.
 *
 * Browser runtimes prefer `CompressionStream` with gzip so large payload work can
 * stay off the synchronous main-thread path. Node falls back to the existing zlib
 * owner when browser streams are unavailable.
 *
 * @param compressedPayload - Existing compressed network payload.
 * @param options - Optional archive compression settings.
 * @returns Base64-wrapped compressed archive payload.
 */
/**
 * Exported contract for createCompressedNetworkArchiveAsync.
 */
export async function createCompressedNetworkArchiveAsync(
  compressedPayload: CompressedSerializedNetwork,
  options: CompressedSerializedNetworkArchiveOptions = {},
): Promise<CompressedSerializedNetworkArchive> {
  const compression =
    options.compression ?? DEFAULT_COMPRESSED_NETWORK_ARCHIVE_COMPRESSION;

  // Step 1: Encode the exact compressed payload JSON into UTF-8 bytes.
  const payloadBytes = new TextEncoder().encode(
    JSON.stringify(compressedPayload),
  );

  // Step 2: Apply the best available async archive codec.
  const compressedBytes = await compressArchivePayloadBytesAsync(
    payloadBytes,
    compression,
  );

  // Step 3: Return a JSON-safe base64 wrapper for storage and transport.
  return {
    compressedFormat: compressedPayload.format,
    compression,
    format: COMPRESSED_NETWORK_ARCHIVE_FORMAT,
    payload: encodeArchivePayloadBase64(compressedBytes),
    payloadEncoding: COMPRESSED_NETWORK_ARCHIVE_ENCODING,
  };
}

/**
 * Rebuild one compressed network payload from its Node-side archive wrapper so persisted archives can be restored into deterministic compact serialization objects.
 *
 * @param compressedArchive - Base64-wrapped compressed archive payload.
 * @returns Restored compressed network payload.
 */
export function parseCompressedNetworkArchive(
  compressedArchive: CompressedSerializedNetworkArchive,
): CompressedSerializedNetwork {
  if (compressedArchive.format !== COMPRESSED_NETWORK_ARCHIVE_FORMAT) {
    throw new TypeError('Invalid compressed network archive format.');
  }

  // Step 1: Decode the base64 archive bytes back into the compressed buffer.
  const compressedBytes = decodeArchivePayloadBase64(compressedArchive.payload);

  // Step 2: Inflate the wrapped UTF-8 JSON payload with the matching codec.
  const payloadBytes = decompressArchivePayloadBytes(
    compressedBytes,
    compressedArchive.compression,
  );

  // Step 3: Parse the original compressed payload object.
  return JSON.parse(
    new TextDecoder().decode(payloadBytes),
  ) as CompressedSerializedNetwork;
}

/**
 * Rebuild one compressed network payload from its archive wrapper with async runtime codecs.
 *
 * Browser runtimes prefer `DecompressionStream` with gzip so archive hydration can
 * stay off the synchronous main-thread path. Node falls back to the existing zlib
 * owner when browser streams are unavailable.
 *
 * @param compressedArchive - Base64-wrapped compressed archive payload.
 * @returns Restored compressed network payload.
 */
/**
 * Exported contract for parseCompressedNetworkArchiveAsync.
 */
export async function parseCompressedNetworkArchiveAsync(
  compressedArchive: CompressedSerializedNetworkArchive,
  options: CompressedArchiveDecodeOptions = {},
): Promise<CompressedSerializedNetwork> {
  if (compressedArchive.format !== COMPRESSED_NETWORK_ARCHIVE_FORMAT) {
    throw new TypeError('Invalid compressed network archive format.');
  }

  // Step 1: Decode the base64 archive bytes back into the compressed buffer.
  const compressedBytes = decodeArchivePayloadBase64(compressedArchive.payload);

  // Step 2: Inflate the wrapped UTF-8 JSON payload with the best available async codec.
  const payloadBytes = await decompressArchivePayloadBytesAsync(
    compressedBytes,
    compressedArchive.compression,
    options,
  );

  // Step 3: Parse the original compressed payload object.
  return JSON.parse(
    new TextDecoder().decode(payloadBytes),
  ) as CompressedSerializedNetwork;
}

/**
 * Compress one serialized connection list into array-oriented exact payload fields.
 *
 * The weight channel stays lossless by delta-encoding the raw IEEE-754 float64
 * words rather than quantizing the numeric values.
 *
 * @param serializedConnections - Compact serialized connection rows.
 * @returns Compressed connection block.
 */
/**
 * Exported contract for compressSerializedConnections.
 */
export function compressSerializedConnections(
  serializedConnections: NetworkJSONConnection[],
): CompressedSerializedConnectionBlock {
  const connectionCount = serializedConnections.length;
  const weightWords = encodeExactConnectionWeights(
    serializedConnections.map(
      (serializedConnection) => serializedConnection.weight,
    ),
  );
  const disabledRuns = compressMatchingRuns(
    serializedConnections.map(
      (serializedConnection) => serializedConnection.enabled === false,
    ),
    (isDisabledConnection) => isDisabledConnection,
  );
  const gaterIndices = serializedConnections.some(
    (serializedConnection) => serializedConnection.gater !== null,
  )
    ? serializedConnections.map((serializedConnection) => {
        return serializedConnection.gater ?? NULL_GATER_INDEX_SENTINEL;
      })
    : undefined;
  return {
    connectionCount,
    disabledRuns,
    fromGeneIds: compressOptionalNumericSeries(
      serializedConnections.map(
        (serializedConnection) => serializedConnection.fromGeneId,
      ),
    ),
    fromIndices: serializedConnections.map(
      (serializedConnection) => serializedConnection.from,
    ),
    gaterGeneIds: compressOptionalNumericSeries(
      serializedConnections.map(
        (serializedConnection) => serializedConnection.gaterGeneId,
      ),
    ),
    gaterIndices,
    innovationIds: compressOptionalNumericSeries(
      serializedConnections.map(
        (serializedConnection) => serializedConnection.innovation,
      ),
    ),
    gainValues: compressOptionalNumericSeries(
      serializedConnections.map(
        (serializedConnection) => serializedConnection.gain,
      ),
    ),
    toGeneIds: compressOptionalNumericSeries(
      serializedConnections.map(
        (serializedConnection) => serializedConnection.toGeneId,
      ),
    ),
    toIndices: serializedConnections.map(
      (serializedConnection) => serializedConnection.to,
    ),
    weightWords,
  };
}

/**
 * Decompress one array-oriented connection block back into compact rows so archived connection payloads regain legacy-friendly per-edge field records.
 *
 * @param compressedConnections - Compressed connection block.
 * @returns Reconstructed serialized connection rows.
 */
export function decompressSerializedConnections(
  compressedConnections: CompressedSerializedConnectionBlock,
): NetworkJSONConnection[] {
  const connectionCount = compressedConnections.connectionCount;

  // Step 1: Validate owner-aligned vector widths before reconstruction.
  validateCompressedVectorLength(
    'fromIndices',
    compressedConnections.fromIndices,
    connectionCount,
  );
  validateCompressedVectorLength(
    'toIndices',
    compressedConnections.toIndices,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'gaterIndices',
    compressedConnections.gaterIndices,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'enabledStates',
    compressedConnections.enabledStates,
    connectionCount,
  );
  validateCompressedRuns(
    'disabledRuns',
    compressedConnections.disabledRuns,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'innovationIds',
    compressedConnections.innovationIds,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'gainValues',
    compressedConnections.gainValues,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'fromGeneIds',
    compressedConnections.fromGeneIds,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'toGeneIds',
    compressedConnections.toGeneIds,
    connectionCount,
  );
  validateCompressedOptionalVectorLength(
    'gaterGeneIds',
    compressedConnections.gaterGeneIds,
    connectionCount,
  );

  // Step 2: Decode the exact float64 weight sequence.
  const decodedWeights = decodeExactConnectionWeights(
    compressedConnections.weightWords,
    connectionCount,
  );
  const disabledMask = expandRunsToMask(
    compressedConnections.disabledRuns,
    connectionCount,
  );

  // Step 3: Rebuild compact serialized rows with legacy defaults.
  return Array.from({ length: connectionCount }, (_, connectionIndex) => {
    const encodedGaterIndex =
      compressedConnections.gaterIndices?.[connectionIndex];
    const isDisabledConnection = disabledMask?.[connectionIndex] === true;

    return {
      enabled: isDisabledConnection
        ? false
        : (compressedConnections.enabledStates?.[connectionIndex] ?? true),
      from: compressedConnections.fromIndices[connectionIndex]!,
      fromGeneId:
        compressedConnections.fromGeneIds?.[connectionIndex] ?? undefined,
      gain: compressedConnections.gainValues?.[connectionIndex] ?? undefined,
      gater:
        typeof encodedGaterIndex === 'number' &&
        encodedGaterIndex !== NULL_GATER_INDEX_SENTINEL
          ? encodedGaterIndex
          : null,
      gaterGeneId:
        compressedConnections.gaterGeneIds?.[connectionIndex] ?? undefined,
      innovation:
        compressedConnections.innovationIds?.[connectionIndex] ?? undefined,
      to: compressedConnections.toIndices[connectionIndex]!,
      toGeneId: compressedConnections.toGeneIds?.[connectionIndex] ?? undefined,
      weight: decodedWeights[connectionIndex]!,
    };
  });
}

/**
 * Estimate the UTF-8 byte length of one serialization payload so archive ratio metrics can compare compressed and uncompressed storage cost.
 *
 * @param payload - Payload to measure.
 * @returns UTF-8 byte length of the JSON string form.
 */
export function estimateSerializedByteLength(payload: unknown): number {
  const textEncoder = new TextEncoder();
  return textEncoder.encode(JSON.stringify(payload)).length;
}

/**
 * Create one metrics snapshot for an archive encode operation so callers can log compression efficiency and elapsed encoding cost consistently.
 *
 * @param uncompressedByteLength - UTF-8 byte length before archive compression.
 * @param compressedByteLength - Binary byte length after archive compression.
 * @param encodeTimeMs - Elapsed encode time in milliseconds.
 * @returns Archive encode metrics.
 */
export function createCompressedArchiveEncodeMetrics(
  uncompressedByteLength: number,
  compressedByteLength: number,
  encodeTimeMs: number,
): CompressedArchiveEncodeMetrics {
  return {
    compressedByteLength,
    compressionRatio: resolveCompressedArchiveRatio(
      compressedByteLength,
      uncompressedByteLength,
    ),
    encodeTimeMs,
    uncompressedByteLength,
  };
}

/**
 * Create one metrics snapshot for an archive decode operation so callers can inspect decompression efficiency and elapsed decoding cost consistently.
 *
 * @param uncompressedByteLength - UTF-8 byte length after archive inflation.
 * @param compressedByteLength - Binary byte length before archive inflation.
 * @param decodeTimeMs - Elapsed decode time in milliseconds.
 * @returns Archive decode metrics.
 */
export function createCompressedArchiveDecodeMetrics(
  uncompressedByteLength: number,
  compressedByteLength: number,
  decodeTimeMs: number,
): CompressedArchiveDecodeMetrics {
  return {
    compressedByteLength,
    compressionRatio: resolveCompressedArchiveRatio(
      compressedByteLength,
      uncompressedByteLength,
    ),
    decodeTimeMs,
    uncompressedByteLength,
  };
}

/**
 * Compress UTF-8 payload bytes with one supported Node-side archive codec so binary wrappers stay compact while preserving exact JSON payload semantics.
 *
 * @param payloadBytes - UTF-8 payload bytes.
 * @param compression - Archive compression codec.
 * @returns Compressed payload bytes.
 */
export function compressArchivePayloadBytes(
  payloadBytes: Uint8Array,
  compression: CompressedSerializedNetworkArchiveCompression,
): Uint8Array {
  const compressionModule = resolveNodeCompressionModule();

  switch (compression) {
    case 'gzip':
      return compressionModule.gzipSync(payloadBytes);
    case 'zstd':
      return compressionModule.zstdCompressSync(payloadBytes);
    default:
      throw new TypeError(
        'Unsupported compressed network archive compression.',
      );
  }
}

/**
 * Compress UTF-8 payload bytes with the best available async archive codec.
 *
 * @param payloadBytes - UTF-8 payload bytes.
 * @param compression - Archive compression codec.
 * @returns Compressed payload bytes.
 */
async function compressArchivePayloadBytesAsync(
  payloadBytes: Uint8Array,
  compression: CompressedSerializedNetworkArchiveCompression,
): Promise<Uint8Array> {
  if (compression === 'gzip') {
    const browserCompressionConstructor =
      resolveBrowserCompressionStreamConstructor();

    if (browserCompressionConstructor) {
      return transformArchivePayloadBytesWithStream(
        payloadBytes,
        browserCompressionConstructor,
      );
    }
  }

  if (hasNodeCompressionRuntime()) {
    return compressArchivePayloadBytes(payloadBytes, compression);
  }

  if (compression === 'zstd') {
    throw new TypeError(
      'Compressed network archives only support gzip in browser runtimes.',
    );
  }

  throw new TypeError(
    'Compressed network archives require either Node.js zlib support or browser CompressionStream support.',
  );
}

/**
 * Decompress archive payload bytes with one supported Node-side codec so compressed archive wrappers recover deterministic UTF-8 serialization payload bytes.
 *
 * @param payloadBytes - Compressed archive payload bytes.
 * @param compression - Archive compression codec.
 * @returns Inflated UTF-8 payload bytes.
 */
export function decompressArchivePayloadBytes(
  payloadBytes: Uint8Array,
  compression: CompressedSerializedNetworkArchiveCompression,
): Uint8Array {
  const compressionModule = resolveNodeCompressionModule();

  switch (compression) {
    case 'gzip':
      return compressionModule.gunzipSync(payloadBytes);
    case 'zstd':
      return compressionModule.zstdDecompressSync(payloadBytes);
    default:
      throw new TypeError(
        'Unsupported compressed network archive compression.',
      );
  }
}

/**
 * Decompress archive payload bytes with the best available async archive codec so browser and Node runtimes can share one non-blocking restore flow.
 *
 * @param payloadBytes - Compressed archive payload bytes.
 * @param compression - Archive compression codec.
 * @returns Inflated UTF-8 payload bytes.
 */
export async function decompressArchivePayloadBytesAsync(
  payloadBytes: Uint8Array,
  compression: CompressedSerializedNetworkArchiveCompression,
  options: CompressedArchiveDecodeOptions = {},
): Promise<Uint8Array> {
  if (compression === 'gzip') {
    const browserDecompressionConstructor =
      resolveBrowserDecompressionStreamConstructor();

    if (browserDecompressionConstructor) {
      return decompressArchivePayloadBytesWithBrowserStream(
        payloadBytes,
        browserDecompressionConstructor,
        options,
      );
    }
  }

  if (hasNodeCompressionRuntime()) {
    const decompressedBytes = decompressArchivePayloadBytes(
      payloadBytes,
      compression,
    );

    await emitArchiveDecodeProgress(options, {
      chunkByteLength: decompressedBytes.length,
      decodedByteLength: decompressedBytes.length,
      encodedByteLength: payloadBytes.length,
      done: true,
    });

    return decompressedBytes;
  }

  if (compression === 'zstd') {
    throw new TypeError(
      'Compressed network archives only support gzip in browser runtimes.',
    );
  }

  throw new TypeError(
    'Compressed network archives require either Node.js zlib support or browser DecompressionStream support.',
  );
}

/**
 * Decompress archive payload bytes through one browser stream while reporting progress.
 *
 * @param payloadBytes - Compressed archive payload bytes.
 * @param streamConstructor - Browser decompression stream constructor.
 * @param options - Optional incremental progress callbacks.
 * @returns Inflated UTF-8 payload bytes.
 */
async function decompressArchivePayloadBytesWithBrowserStream(
  payloadBytes: Uint8Array,
  streamConstructor: BrowserDecompressionStreamConstructor,
  options: CompressedArchiveDecodeOptions,
): Promise<Uint8Array> {
  const normalizedPayloadBytes = Uint8Array.from(payloadBytes);
  const transformedStream = new Blob([normalizedPayloadBytes])
    .stream()
    .pipeThrough(new streamConstructor('gzip'));

  return collectDecodedArchivePayloadBytes(
    transformedStream,
    normalizedPayloadBytes.length,
    options,
  );
}

/**
 * Transform archive payload bytes through one browser compression stream.
 *
 * @param payloadBytes - Source payload bytes.
 * @param streamConstructor - Browser stream constructor.
 * @returns Transformed payload bytes.
 */
async function transformArchivePayloadBytesWithStream(
  payloadBytes: Uint8Array,
  streamConstructor:
    BrowserCompressionStreamConstructor | BrowserDecompressionStreamConstructor,
): Promise<Uint8Array> {
  const normalizedPayloadBytes = Uint8Array.from(payloadBytes);
  const transformedStream = new Blob([normalizedPayloadBytes])
    .stream()
    .pipeThrough(new streamConstructor('gzip'));
  const transformedBuffer = await new Response(transformedStream).arrayBuffer();

  return new Uint8Array(transformedBuffer);
}

/**
 * Collect one decoded archive stream into bytes while surfacing chunk progress.
 *
 * The implementation buffers one chunk ahead so the final emitted snapshot can
 * mark `done: true` on the last real decoded chunk rather than on a synthetic
 * zero-byte completion event.
 *
 * @param decodedStream - Stream of decoded UTF-8 payload chunks.
 * @param encodedByteLength - Total compressed archive byte length.
 * @param options - Optional incremental progress callbacks.
 * @returns Concatenated decoded payload bytes.
 */
async function collectDecodedArchivePayloadBytes(
  decodedStream: ReadableStream<Uint8Array>,
  encodedByteLength: number,
  options: CompressedArchiveDecodeOptions,
): Promise<Uint8Array> {
  const decodedReader = decodedStream.getReader();
  const decodedChunks: Uint8Array[] = [];
  let decodedByteLength = 0;
  let pendingChunk: Uint8Array | undefined;

  while (true) {
    const nextRead = await decodedReader.read();

    if (nextRead.done) {
      if (pendingChunk) {
        decodedChunks.push(pendingChunk);
        decodedByteLength += pendingChunk.length;
        await emitArchiveDecodeProgress(options, {
          chunkByteLength: pendingChunk.length,
          decodedByteLength,
          encodedByteLength,
          done: true,
        });
      }

      break;
    }

    const currentChunk = Uint8Array.from(nextRead.value);

    if (pendingChunk) {
      decodedChunks.push(pendingChunk);
      decodedByteLength += pendingChunk.length;
      await emitArchiveDecodeProgress(options, {
        chunkByteLength: pendingChunk.length,
        decodedByteLength,
        encodedByteLength,
        done: false,
      });
    }

    pendingChunk = currentChunk;
  }

  return concatenateArchiveByteChunks(decodedChunks);
}

/**
 * Notify callers that one decoded archive chunk has been observed.
 *
 * @param options - Optional incremental progress callbacks.
 * @param progress - Progress payload for the decoded chunk.
 * @returns Nothing.
 */
async function emitArchiveDecodeProgress(
  options: CompressedArchiveDecodeOptions,
  progress: CompressedArchiveDecodeProgress,
): Promise<void> {
  if (typeof options.onProgress === 'function') {
    await options.onProgress(progress);
  }
}

/**
 * Concatenate one ordered set of archive byte chunks into a single buffer.
 *
 * @param byteChunks - Ordered archive byte chunks.
 * @returns One contiguous byte buffer.
 */
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

function resolveCompressedArchiveRatio(
  compressedByteLength: number,
  uncompressedByteLength: number,
): number {
  return uncompressedByteLength === 0
    ? 0
    : compressedByteLength / uncompressedByteLength;
}

/**
 * Resolve the Node builtin zlib module without introducing a static browser import.
 *
 * @returns Node compression module.
 */
function resolveNodeCompressionModule(): NodeCompressionModule {
  const compressionModule = resolveNodeCompressionModuleOrUndefined();

  if (!compressionModule) {
    throw new TypeError(
      'Compressed network archives require a Node.js runtime.',
    );
  }

  return compressionModule;
}

/**
 * Resolve the Node builtin zlib module when the runtime exposes it.
 *
 * @returns Node compression module when available.
 */
function resolveNodeCompressionModuleOrUndefined():
  NodeCompressionModule | undefined {
  const builtinModuleLoader =
    typeof process !== 'undefined'
      ? (process as ProcessWithBuiltinModule).getBuiltinModule
      : undefined;

  return typeof builtinModuleLoader === 'function'
    ? (builtinModuleLoader('node:zlib') as NodeCompressionModule)
    : undefined;
}

/**
 * Determine whether the current runtime can resolve the Node compression owner.
 *
 * @returns Whether Node zlib support is available.
 */
function hasNodeCompressionRuntime(): boolean {
  return resolveNodeCompressionModuleOrUndefined() !== undefined;
}

/**
 * Resolve the browser compression-stream constructor when one is available.
 *
 * @returns Browser compression-stream constructor.
 */
function resolveBrowserCompressionStreamConstructor():
  BrowserCompressionStreamConstructor | undefined {
  const compressionStreamConstructor = Reflect.get(
    globalThis,
    'CompressionStream',
  );

  return typeof compressionStreamConstructor === 'function'
    ? (compressionStreamConstructor as BrowserCompressionStreamConstructor)
    : undefined;
}

/**
 * Resolve the browser decompression-stream constructor when one is available.
 *
 * @returns Browser decompression-stream constructor.
 */
function resolveBrowserDecompressionStreamConstructor():
  BrowserDecompressionStreamConstructor | undefined {
  const decompressionStreamConstructor = Reflect.get(
    globalThis,
    'DecompressionStream',
  );

  return typeof decompressionStreamConstructor === 'function'
    ? (decompressionStreamConstructor as BrowserDecompressionStreamConstructor)
    : undefined;
}

/**
 * Encode archive payload bytes to base64 without assuming a specific runtime so archive wrappers remain portable across browser and Node environments.
 *
 * @param payloadBytes - Binary archive payload bytes.
 * @returns Base64-encoded payload text.
 */
export function encodeArchivePayloadBase64(payloadBytes: Uint8Array): string {
  const browserBase64Encoder = Reflect.get(globalThis, 'btoa');

  if (typeof browserBase64Encoder === 'function') {
    let binaryPayload = '';

    for (
      let payloadOffset = 0;
      payloadOffset < payloadBytes.length;
      payloadOffset += 0x8000
    ) {
      binaryPayload += String.fromCharCode(
        ...payloadBytes.subarray(payloadOffset, payloadOffset + 0x8000),
      );
    }

    return browserBase64Encoder(binaryPayload);
  }

  if (typeof Buffer !== 'undefined') {
    return Buffer.from(payloadBytes).toString(
      COMPRESSED_NETWORK_ARCHIVE_ENCODING,
    );
  }

  throw new TypeError(
    'Compressed network archives require base64 support in the current runtime.',
  );
}

/**
 * Decode archive payload bytes from base64 without assuming a specific runtime so compressed archives can hydrate reliably in browser and Node.
 *
 * @param payload - Base64-encoded payload text.
 * @returns Binary archive payload bytes.
 */
export function decodeArchivePayloadBase64(payload: string): Uint8Array {
  const browserBase64Decoder = Reflect.get(globalThis, 'atob');

  if (typeof browserBase64Decoder === 'function') {
    const binaryPayload = browserBase64Decoder(payload);
    const payloadBytes = new Uint8Array(binaryPayload.length);

    for (
      let payloadIndex = 0;
      payloadIndex < binaryPayload.length;
      payloadIndex += 1
    ) {
      payloadBytes[payloadIndex] = binaryPayload.charCodeAt(payloadIndex);
    }

    return payloadBytes;
  }

  if (typeof Buffer !== 'undefined') {
    return Uint8Array.from(
      Buffer.from(payload, COMPRESSED_NETWORK_ARCHIVE_ENCODING),
    );
  }

  throw new TypeError(
    'Compressed network archives require base64 support in the current runtime.',
  );
}

/**
 * Collapse optional numeric fields to `undefined` when no entries are present.
 *
 * @param values - Optional numeric series.
 * @returns Normalized nullable series or `undefined` when empty of numeric content.
 */
function compressOptionalNumericSeries(
  values: Array<number | null | undefined>,
): Array<number | null> | undefined {
  const normalizedValues = values.map((value) => {
    return typeof value === 'number' ? value : null;
  });

  return normalizedValues.some((value) => value !== null)
    ? normalizedValues
    : undefined;
}

/**
 * Encode float64 weights into one exact signed-16-bit delta stream.
 *
 * @param weights - Connection weights.
 * @returns Exact weight-word delta payload.
 */
function encodeExactConnectionWeights(
  weights: number[],
): CompressedSerializedConnectionWeights {
  const encodedWeightWords = weights.map((weightValue) => {
    return encodeFloat64ToSignedInt16Words(weightValue);
  });
  const zeroWeightRuns = compressMatchingRuns(
    encodedWeightWords,
    isPositiveZeroWeightWords,
  );
  const nonZeroWeightWords = encodedWeightWords.filter((currentWeightWords) => {
    return !isPositiveZeroWeightWords(currentWeightWords);
  });

  if (weights.length === 0) {
    return {
      deltaWords: [],
      encoding: COMPRESSED_WEIGHT_ENCODING,
      firstWeightWords: [],
    };
  }

  if (nonZeroWeightWords.length === 0) {
    return {
      deltaWords: [],
      encoding: COMPRESSED_WEIGHT_ENCODING,
      firstWeightWords: [],
      zeroWeightRuns,
    };
  }

  const firstWeightWords = nonZeroWeightWords[0]!;
  const deltaWords: number[] = [];
  let previousWeightWords = firstWeightWords;

  nonZeroWeightWords.slice(1).forEach((currentWeightWords) => {
    currentWeightWords.forEach((currentWeightWord, wordIndex) => {
      deltaWords.push(
        normalizeSignedInt16(
          currentWeightWord - previousWeightWords[wordIndex]!,
        ),
      );
    });

    previousWeightWords = currentWeightWords;
  });

  return {
    deltaWords,
    encoding: COMPRESSED_WEIGHT_ENCODING,
    firstWeightWords,
    zeroWeightRuns,
  };
}

/**
 * Decode one exact signed-16-bit delta stream back to float64 weights.
 *
 * @param weightWords - Encoded weight-word payload.
 * @param connectionCount - Expected number of weights.
 * @returns Exact reconstructed weights.
 */
function decodeExactConnectionWeights(
  weightWords: CompressedSerializedConnectionWeights,
  connectionCount: number,
): number[] {
  if (connectionCount === 0) {
    return [];
  }

  if (weightWords.encoding !== COMPRESSED_WEIGHT_ENCODING) {
    throw new TypeError('Unsupported compressed weight encoding.');
  }

  validateCompressedRuns(
    'zeroWeightRuns',
    weightWords.zeroWeightRuns,
    connectionCount,
  );

  const zeroWeightCount = countRunEntries(weightWords.zeroWeightRuns);
  const nonZeroWeightCount = connectionCount - zeroWeightCount;
  const zeroWeightMask = expandRunsToMask(
    weightWords.zeroWeightRuns,
    connectionCount,
  );

  if (nonZeroWeightCount === 0) {
    if (weightWords.firstWeightWords.length !== 0) {
      throw new TypeError('Compressed weight base word count is invalid.');
    }

    if (weightWords.deltaWords.length !== 0) {
      throw new TypeError('Compressed weight delta word count is invalid.');
    }

    return Array.from({ length: connectionCount }, () => 0);
  }

  if (weightWords.firstWeightWords.length !== FLOAT64_INT16_WORD_COUNT) {
    throw new TypeError('Compressed weight base word count is invalid.');
  }

  if (
    weightWords.deltaWords.length !==
    Math.max(0, nonZeroWeightCount - 1) * FLOAT64_INT16_WORD_COUNT
  ) {
    throw new TypeError('Compressed weight delta word count is invalid.');
  }

  const decodedNonZeroWeights = [
    decodeSignedInt16WordsToFloat64(weightWords.firstWeightWords),
  ];
  let previousWeightWords = weightWords.firstWeightWords.slice();

  for (
    let connectionIndex = 1;
    connectionIndex < nonZeroWeightCount;
    connectionIndex += 1
  ) {
    const deltaOffset = (connectionIndex - 1) * FLOAT64_INT16_WORD_COUNT;
    const currentWeightWords = previousWeightWords.map(
      (previousWeightWord, wordIndex) => {
        return normalizeSignedInt16(
          previousWeightWord + weightWords.deltaWords[deltaOffset + wordIndex]!,
        );
      },
    );

    decodedNonZeroWeights.push(
      decodeSignedInt16WordsToFloat64(currentWeightWords),
    );
    previousWeightWords = currentWeightWords;
  }

  return expandDecodedNonZeroWeights(
    decodedNonZeroWeights,
    zeroWeightMask,
    connectionCount,
  );
}

/**
 * Encode one float64 number into four signed 16-bit words.
 *
 * @param value - Numeric value to encode.
 * @returns Signed 16-bit little-endian words.
 */
function encodeFloat64ToSignedInt16Words(value: number): number[] {
  const wordBuffer = new ArrayBuffer(Float64Array.BYTES_PER_ELEMENT);
  const wordView = new DataView(wordBuffer);

  wordView.setFloat64(0, value, true);

  return Array.from({ length: FLOAT64_INT16_WORD_COUNT }, (_, wordIndex) => {
    return wordView.getInt16(wordIndex * Int16Array.BYTES_PER_ELEMENT, true);
  });
}

/**
 * Decode four signed 16-bit words back into one float64 number.
 *
 * @param words - Signed 16-bit little-endian words.
 * @returns Decoded numeric value.
 */
function decodeSignedInt16WordsToFloat64(words: number[]): number {
  const wordBuffer = new ArrayBuffer(Float64Array.BYTES_PER_ELEMENT);
  const wordView = new DataView(wordBuffer);

  words.forEach((wordValue, wordIndex) => {
    wordView.setInt16(
      wordIndex * Int16Array.BYTES_PER_ELEMENT,
      wordValue,
      true,
    );
  });

  return wordView.getFloat64(0, true);
}

/**
 * Normalize one integer through wrapped signed-16-bit arithmetic.
 *
 * @param value - Integer value to normalize.
 * @returns Wrapped signed 16-bit integer.
 */
function normalizeSignedInt16(value: number): number {
  return (
    ((value + SIGNED_INT16_WRAP_OFFSET) & SIGNED_INT16_WRAP_MASK) -
    SIGNED_INT16_WRAP_OFFSET
  );
}

/**
 * Collapse contiguous matching entries into run-length metadata.
 *
 * @param values - Ordered values to scan.
 * @param shouldCompressValue - Predicate deciding whether one value belongs to a run.
 * @returns Run metadata or `undefined` when no matching span exists.
 */
function compressMatchingRuns<Value>(
  values: Value[],
  shouldCompressValue: (value: Value) => boolean,
): CompressedSerializedIndexRun[] | undefined {
  const compressedRuns: CompressedSerializedIndexRun[] = [];
  let activeRunStartIndex: number | undefined;

  values.forEach((value, valueIndex) => {
    const isCompressedValue = shouldCompressValue(value);

    if (isCompressedValue && activeRunStartIndex === undefined) {
      activeRunStartIndex = valueIndex;
      return;
    }

    if (!isCompressedValue && activeRunStartIndex !== undefined) {
      compressedRuns.push({
        length: valueIndex - activeRunStartIndex,
        startIndex: activeRunStartIndex,
      });
      activeRunStartIndex = undefined;
    }
  });

  if (activeRunStartIndex !== undefined) {
    compressedRuns.push({
      length: values.length - activeRunStartIndex,
      startIndex: activeRunStartIndex,
    });
  }

  return compressedRuns.length > 0 ? compressedRuns : undefined;
}

/**
 * Count how many connection rows are covered by one run list.
 *
 * @param runs - Optional run metadata.
 * @returns Total covered row count.
 */
function countRunEntries(
  runs: CompressedSerializedIndexRun[] | undefined,
): number {
  return (
    runs?.reduce((coveredEntryCount, run) => {
      return coveredEntryCount + run.length;
    }, 0) ?? 0
  );
}

/**
 * Expand run-length metadata into one boolean mask aligned to connection order.
 *
 * @param runs - Optional run metadata.
 * @param entryCount - Total number of connection rows.
 * @returns Boolean run-membership mask or `undefined` when no runs exist.
 */
function expandRunsToMask(
  runs: CompressedSerializedIndexRun[] | undefined,
  entryCount: number,
): boolean[] | undefined {
  if (!runs || runs.length === 0) {
    return undefined;
  }

  const runMask = Array.from({ length: entryCount }, () => false);

  runs.forEach((run) => {
    for (
      let connectionIndex = run.startIndex;
      connectionIndex < run.startIndex + run.length;
      connectionIndex += 1
    ) {
      runMask[connectionIndex] = true;
    }
  });

  return runMask;
}

/**
 * Expand the decoded non-zero weight sequence back across zero-weight spans.
 *
 * @param decodedNonZeroWeights - Exact non-zero weights in encoded order.
 * @param zeroWeightMask - Optional zero-run membership mask.
 * @param connectionCount - Total number of serialized connections.
 * @returns Connection-aligned exact weight values.
 */
function expandDecodedNonZeroWeights(
  decodedNonZeroWeights: number[],
  zeroWeightMask: boolean[] | undefined,
  connectionCount: number,
): number[] {
  if (!zeroWeightMask) {
    return decodedNonZeroWeights;
  }

  const decodedWeights: number[] = [];
  let nonZeroWeightIndex = 0;

  for (
    let connectionIndex = 0;
    connectionIndex < connectionCount;
    connectionIndex += 1
  ) {
    decodedWeights.push(
      zeroWeightMask[connectionIndex] === true
        ? 0
        : decodedNonZeroWeights[nonZeroWeightIndex++]!,
    );
  }

  return decodedWeights;
}

/**
 * Determine whether one encoded float64 word sequence represents exact positive zero.
 *
 * @param encodedWeightWords - Signed 16-bit float64 words.
 * @returns Whether the encoded value is exact positive zero.
 */
function isPositiveZeroWeightWords(encodedWeightWords: number[]): boolean {
  return encodedWeightWords.every((encodedWeightWord) => {
    return encodedWeightWord === 0;
  });
}

/**
 * Validate one required vector width inside the compressed connection block.
 *
 * @param fieldName - Logical field name.
 * @param values - Vector to validate.
 * @param expectedLength - Required vector length.
 * @returns Nothing.
 */
function validateCompressedVectorLength(
  fieldName: string,
  values: unknown[],
  expectedLength: number,
): void {
  if (values.length !== expectedLength) {
    throw new TypeError(
      `Compressed connection field "${fieldName}" length is invalid.`,
    );
  }
}

/**
 * Validate one optional vector width when the vector is present.
 *
 * @param fieldName - Logical field name.
 * @param values - Optional vector.
 * @param expectedLength - Required vector length.
 * @returns Nothing.
 */
function validateCompressedOptionalVectorLength(
  fieldName: string,
  values: unknown[] | undefined,
  expectedLength: number,
): void {
  if (!values) {
    return;
  }

  validateCompressedVectorLength(fieldName, values, expectedLength);
}

/**
 * Validate run metadata for bounds, ordering, and overlap.
 *
 * @param fieldName - Logical field name.
 * @param runs - Optional run metadata.
 * @param entryCount - Total number of connection rows.
 * @returns Nothing.
 */
function validateCompressedRuns(
  fieldName: string,
  runs: CompressedSerializedIndexRun[] | undefined,
  entryCount: number,
): void {
  if (!runs) {
    return;
  }

  let nextAllowedStartIndex = 0;

  runs.forEach((run) => {
    const runEndIndex = run.startIndex + run.length;
    const hasInvalidBounds =
      !Number.isInteger(run.startIndex) ||
      !Number.isInteger(run.length) ||
      run.length <= 0 ||
      run.startIndex < nextAllowedStartIndex ||
      runEndIndex > entryCount;

    if (hasInvalidBounds) {
      throw new TypeError(
        `Compressed connection field "${fieldName}" run layout is invalid.`,
      );
    }

    nextAllowedStartIndex = runEndIndex;
  });
}
