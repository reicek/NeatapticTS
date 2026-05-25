import type {
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
} from '../network.types';
import type Network from '../network';
import {
  assertValidGenomeContract,
  createGenomeFromNetwork,
  createNetworkFromGenome,
  type GenomeMaterializationRuntimeHints,
  type NeatGenome,
  type NeatGenomeCaptureOptions,
} from '../../../neat/genome/genome';
import {
  COMPRESSED_NETWORK_ARCHIVE_ENCODING,
  createCompressedArchiveDecodeMetrics,
  createCompressedArchiveEncodeMetrics,
  type CompressedArchiveDecodeOptions,
  type CompressedArchiveDecodeResult,
  type CompressedArchiveEncodeResult,
  compressArchivePayloadBytes,
  decodeArchivePayloadBase64,
  decompressArchivePayloadBytes,
  decompressArchivePayloadBytesAsync,
  encodeArchivePayloadBase64,
  estimateSerializedByteLength,
} from './network.serialize.compression.utils';

/** Stable payload tag used for strict-genome archives. */
export const COMPRESSED_GENOME_FORMAT = 'neat-genome-v1';

/** Stable archive wrapper tag used for strict-genome archives. */
export const COMPRESSED_GENOME_ARCHIVE_FORMAT = 'neat-genome-archive-v1';

/**
 * Archive options for strict-genome compression that combine binary codec choices with capture-time genome field toggles.
 * These options let callers tune payload size and fidelity without introducing phenotype-only runtime state.
 */
export interface CompressedSerializedGenomeArchiveOptions extends CompressedSerializedNetworkArchiveOptions {
  /** Optional strict-genome capture switches applied before compression. */
  captureOptions?: NeatGenomeCaptureOptions;
}

/** JSON-safe archive wrapper for one strict genome contract. */
export interface CompressedSerializedGenomeArchive {
  /** Stable archive wrapper tag for strict-genome payloads. */
  format: typeof COMPRESSED_GENOME_ARCHIVE_FORMAT;
  /** Stable payload tag for the wrapped strict genome JSON. */
  compressedFormat: typeof COMPRESSED_GENOME_FORMAT;
  /** Archive compression codec applied above the strict genome JSON bytes. */
  compression: CompressedSerializedNetworkArchiveCompression;
  /** Binary-to-text encoding used for the archived bytes. */
  payloadEncoding: typeof COMPRESSED_NETWORK_ARCHIVE_ENCODING;
  /** Encoded archive bytes. */
  payload: string;
}

/** Default strict-genome capture switches for archive round-trips. */
const DEFAULT_GENOME_CAPTURE_OPTIONS: Required<NeatGenomeCaptureOptions> = {
  connectionGain: true,
  disabledConnectionReenableProbability: true,
  nodeResponse: true,
};

/** Default archive compression codec for strict-genome payloads. */
const DEFAULT_COMPRESSED_GENOME_ARCHIVE_COMPRESSION = 'gzip';

/**
 * Archive one runtime phenotype through the strict `NeatGenome` contract.
 *
 * This keeps the payload structural and replay-safe: runtime activation traces,
 * slab allocations, and other phenotype-only state remain outside the archive.
 *
 * @param options - Optional archive codec and genome-capture settings.
 * @returns Base64-wrapped compressed strict-genome archive.
 *
 * @example
 * ```ts
 * const archive = serializeCompressedGenomeArchive.call(network);
 * const rebuiltNetwork = deserializeCompressedGenomeArchive(archive);
 * ```
 */
export function serializeCompressedGenomeArchive(
  this: Network,
  options: CompressedSerializedGenomeArchiveOptions = {},
): CompressedSerializedGenomeArchive {
  const compression =
    options.compression ?? DEFAULT_COMPRESSED_GENOME_ARCHIVE_COMPRESSION;
  const strictGenome = createGenomeFromNetwork(
    this,
    resolveGenomeCaptureOptions(options.captureOptions),
  );

  // Step 1: Encode the strict genome JSON into UTF-8 bytes.
  const payloadBytes = new TextEncoder().encode(JSON.stringify(strictGenome));

  // Step 2: Apply the requested archive codec above the genome payload.
  const compressedBytes = compressArchivePayloadBytes(
    payloadBytes,
    compression,
  );

  // Step 3: Return a JSON-safe base64 wrapper for storage and transport.
  return {
    compressedFormat: COMPRESSED_GENOME_FORMAT,
    compression,
    format: COMPRESSED_GENOME_ARCHIVE_FORMAT,
    payload: encodeArchivePayloadBase64(compressedBytes),
    payloadEncoding: COMPRESSED_NETWORK_ARCHIVE_ENCODING,
  };
}

/**
 * Archive one runtime phenotype through the strict genome contract and report deterministic encode metrics for observability.
 * The metrics payload helps compare codec and capture-option tradeoffs without changing archive semantics.
 *
 * @param options - Optional archive codec and genome-capture settings.
 * @returns Archived strict-genome payload plus encode metrics.
 */
export function serializeCompressedGenomeArchiveWithMetrics(
  this: Network,
  options: CompressedSerializedGenomeArchiveOptions = {},
): CompressedArchiveEncodeResult<CompressedSerializedGenomeArchive> {
  const startedAt = performance.now();
  const archive = serializeCompressedGenomeArchive.call(this, options);
  const encodeTimeMs = performance.now() - startedAt;
  const strictGenome = createGenomeFromNetwork(
    this,
    resolveGenomeCaptureOptions(options.captureOptions),
  );

  return {
    archive,
    metrics: createCompressedArchiveEncodeMetrics(
      estimateSerializedByteLength(strictGenome),
      decodeArchivePayloadBase64(archive.payload).length,
      encodeTimeMs,
    ),
  };
}

/**
 * Parse one archived strict genome contract back into validated JSON state before any runtime materialization begins.
 * This boundary enforces archive tags and schema validity so malformed payloads fail with clear diagnostics.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @returns Restored strict genome contract.
 */
export function parseCompressedGenomeArchive(
  compressedArchive: CompressedSerializedGenomeArchive,
): NeatGenome {
  if (compressedArchive.format !== COMPRESSED_GENOME_ARCHIVE_FORMAT) {
    throw new TypeError('Invalid compressed genome archive format.');
  }

  if (compressedArchive.compressedFormat !== COMPRESSED_GENOME_FORMAT) {
    throw new TypeError('Invalid compressed genome payload format.');
  }

  // Step 1: Decode the archived bytes back into the compressed binary buffer.
  const compressedBytes = decodeArchivePayloadBase64(compressedArchive.payload);

  // Step 2: Inflate the wrapped strict genome JSON bytes with the matching codec.
  const payloadBytes = decompressArchivePayloadBytes(
    compressedBytes,
    compressedArchive.compression,
  );
  const strictGenome = JSON.parse(
    new TextDecoder().decode(payloadBytes),
  ) as NeatGenome;

  // Step 3: Validate the restored contract before exposing it to callers.
  assertValidGenomeContract(strictGenome);

  return strictGenome;
}

/**
 * Parse one archived strict genome contract with async runtime codecs and progress callbacks.
 *
 * Browser runtimes prefer the streaming decode path so large archives can emit
 * incremental progress while `DecompressionStream` inflates the wrapped UTF-8
 * JSON bytes. Node falls back to one completed snapshot.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @param options - Optional incremental decode callbacks.
 * @returns Restored strict genome contract.
 */
export async function parseCompressedGenomeArchiveAsync(
  compressedArchive: CompressedSerializedGenomeArchive,
  options: CompressedArchiveDecodeOptions = {},
): Promise<NeatGenome> {
  if (compressedArchive.format !== COMPRESSED_GENOME_ARCHIVE_FORMAT) {
    throw new TypeError('Invalid compressed genome archive format.');
  }

  if (compressedArchive.compressedFormat !== COMPRESSED_GENOME_FORMAT) {
    throw new TypeError('Invalid compressed genome payload format.');
  }

  // Step 1: Decode the archived bytes back into the compressed binary buffer.
  const compressedBytes = decodeArchivePayloadBase64(compressedArchive.payload);

  // Step 2: Inflate the wrapped strict genome JSON bytes with the best available async codec.
  const payloadBytes = await decompressArchivePayloadBytesAsync(
    compressedBytes,
    compressedArchive.compression,
    options,
  );
  const strictGenome = JSON.parse(
    new TextDecoder().decode(payloadBytes),
  ) as NeatGenome;

  // Step 3: Validate the restored contract before exposing it to callers.
  assertValidGenomeContract(strictGenome);

  return strictGenome;
}

/**
 * Materialize one runnable phenotype from a compressed strict-genome archive using validated contract data.
 * Runtime hints are applied only after strict-genome restoration so deterministic genotype state stays authoritative.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @param runtimeHints - Optional phenotype-only metadata to restore.
 * @returns Rebuilt executable runtime phenotype.
 */
export function deserializeCompressedGenomeArchive(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
): Network {
  return createNetworkFromGenome(
    parseCompressedGenomeArchive(compressedArchive),
    runtimeHints,
  );
}

/**
 * Materialize one runnable phenotype from a compressed strict-genome archive and report decode metrics for runtime analysis.
 * This variant keeps decode telemetry alongside the rebuilt network for reproducibility and performance audits.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @param runtimeHints - Optional phenotype-only metadata to restore.
 * @returns Rebuilt network plus decode metrics.
 */
export function deserializeCompressedGenomeArchiveWithMetrics(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
): CompressedArchiveDecodeResult<Network> {
  const compressedByteLength = decodeArchivePayloadBase64(
    compressedArchive.payload,
  ).length;
  const startedAt = performance.now();
  const strictGenome = parseCompressedGenomeArchive(compressedArchive);
  const rebuiltNetwork = createNetworkFromGenome(strictGenome, runtimeHints);
  const decodeTimeMs = performance.now() - startedAt;

  return {
    metrics: createCompressedArchiveDecodeMetrics(
      estimateSerializedByteLength(strictGenome),
      compressedByteLength,
      decodeTimeMs,
    ),
    value: rebuiltNetwork,
  };
}

/**
 * Materialize one runnable phenotype from a compressed strict-genome archive with async decode progress callbacks.
 * This path is suited for browser or streaming contexts where large payload inflation should stay responsive.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @param runtimeHints - Optional phenotype-only metadata to restore.
 * @param options - Optional incremental decode callbacks.
 * @returns Rebuilt executable runtime phenotype.
 */
export async function deserializeCompressedGenomeArchiveAsync(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
  options: CompressedArchiveDecodeOptions = {},
): Promise<Network> {
  return createNetworkFromGenome(
    await parseCompressedGenomeArchiveAsync(compressedArchive, options),
    runtimeHints,
  );
}

/**
 * Materialize one runnable phenotype from a compressed strict-genome archive with async decode metrics and progress support.
 * The returned telemetry helps compare streaming decode strategies while preserving the same strict reconstruction contract.
 *
 * @param compressedArchive - Base64-wrapped strict-genome archive payload.
 * @param runtimeHints - Optional phenotype-only metadata to restore.
 * @param options - Optional incremental decode callbacks.
 * @returns Rebuilt network plus decode metrics.
 */
export async function deserializeCompressedGenomeArchiveAsyncWithMetrics(
  compressedArchive: CompressedSerializedGenomeArchive,
  runtimeHints: GenomeMaterializationRuntimeHints = {},
  options: CompressedArchiveDecodeOptions = {},
): Promise<CompressedArchiveDecodeResult<Network>> {
  const compressedByteLength = decodeArchivePayloadBase64(
    compressedArchive.payload,
  ).length;
  const startedAt = performance.now();
  const strictGenome = await parseCompressedGenomeArchiveAsync(
    compressedArchive,
    options,
  );
  const rebuiltNetwork = createNetworkFromGenome(strictGenome, runtimeHints);
  const decodeTimeMs = performance.now() - startedAt;

  return {
    metrics: createCompressedArchiveDecodeMetrics(
      estimateSerializedByteLength(strictGenome),
      compressedByteLength,
      decodeTimeMs,
    ),
    value: rebuiltNetwork,
  };
}

function resolveGenomeCaptureOptions(
  captureOptions: NeatGenomeCaptureOptions | undefined,
): Required<NeatGenomeCaptureOptions> {
  return {
    connectionGain:
      captureOptions?.connectionGain ??
      DEFAULT_GENOME_CAPTURE_OPTIONS.connectionGain,
    disabledConnectionReenableProbability:
      captureOptions?.disabledConnectionReenableProbability ??
      DEFAULT_GENOME_CAPTURE_OPTIONS.disabledConnectionReenableProbability,
    nodeResponse:
      captureOptions?.nodeResponse ??
      DEFAULT_GENOME_CAPTURE_OPTIONS.nodeResponse,
  };
}
