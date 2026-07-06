import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';

export async function getFreshnessProof(filePath) {
  const [fileStats, fileBuffer] = await Promise.all([
    stat(filePath),
    readFile(filePath),
  ]);

  return {
    mtime_ms: Math.trunc(fileStats.mtimeMs),
    size: fileBuffer.byteLength,
    sha256: createHash('sha256').update(fileBuffer).digest('hex'),
  };
}

export function isFreshDocument(documentRow, freshnessProof) {
  return Boolean(
    documentRow &&
    freshnessProof &&
    Number(documentRow.mtime_ms) === Number(freshnessProof.mtime_ms) &&
    Number(documentRow.file_size) ===
      Number(freshnessProof.size ?? freshnessProof.file_size) &&
    documentRow.sha256 === freshnessProof.sha256,
  );
}
