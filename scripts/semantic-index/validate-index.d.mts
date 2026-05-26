export interface SemanticIndexValidationInput {
  documents?: Array<Record<string, number | string>>;
  freshnessChecks?: Array<Record<string, number | string>>;
  chunks?: number;
  minDocuments?: number;
  minChunks?: number;
  now?: number;
  maxStalenessMs?: number;
}

export interface SemanticIndexValidationResult {
  ok: boolean;
  pass: boolean;
  failures: string[];
  documents: number;
  chunks: number;
}

export function validateSemanticIndex(input?: SemanticIndexValidationInput): Promise<SemanticIndexValidationResult>;
export function validateDatabase(options?: Record<string, number | string>): Promise<SemanticIndexValidationResult>;