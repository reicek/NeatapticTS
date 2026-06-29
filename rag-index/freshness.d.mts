export interface FreshnessProof {
  mtime_ms: number;
  size: number;
  sha256: string;
}

export interface StoredFreshnessProof {
  mtime_ms: number;
  file_size?: number;
  size?: number;
  sha256: string;
}

export function getFreshnessProof(filePath: string): Promise<FreshnessProof>;
export function isFreshDocument(
  documentRow: StoredFreshnessProof | null | undefined,
  freshnessProof: StoredFreshnessProof | null | undefined,
): boolean;
