export interface FreshnessManifestFamilyEntry {
  fresh: boolean;
  stalePaths: string[];
  lastReindex: number;
  maxSyncWaitMs: number;
  max_age_ms?: number | null;
  gated: boolean;
}

export interface FreshnessManifest {
  lastReindex: number;
  updatedBy: string;
  families: Record<string, FreshnessManifestFamilyEntry>;
}

export interface SemanticIndexValidationInput {
  documents?: Array<Record<string, number | string>>;
  freshnessChecks?: Array<Record<string, number | string>>;
  chunks?: number;
  minDocuments?: number;
  minChunks?: number;
  now?: number;
  familyAssignments?: Map<string, string>;
  manifest?: FreshnessManifest | null;
}

export interface SemanticIndexValidationResult {
  ok: boolean;
  pass: boolean;
  failures: string[];
  documents: number;
  chunks: number;
  stale_paths: string[];
  missing_paths: string[];
  warnings: string[];
  family_fresh: Record<string, { fresh: boolean; stalePaths: string[] }>;
  fixHint: string | null;
}

export function validateSemanticIndex(
  input?: SemanticIndexValidationInput,
): Promise<SemanticIndexValidationResult>;
export function validateDatabase(
  options?: Record<string, unknown>,
): Promise<SemanticIndexValidationResult>;
export function loadFreshnessManifest(
  manifestPath?: string,
): FreshnessManifest | null;
export function writeFreshnessManifest(
  familyFresh: Record<string, { fresh: boolean; stalePaths: string[] }>,
  outputPath?: string,
  updatedBy?: string,
  previousManifest?: FreshnessManifest | null,
): Promise<FreshnessManifest>;
