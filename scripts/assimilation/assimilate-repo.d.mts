/**
 * Type declarations for the ESM assimilation script.
 *
 * These types allow TypeScript test files to import the ESM module without
 * resorting to `@ts-nocheck`.
 */

export interface AssimilationReport {
  /** Absolute or relative workspace folder where the repo was saved. */
  baseFolder: string;
  /** Repository name. */
  repo: string;
  /** Repository owner. */
  owner: string;
  /** Git ref that was fetched. */
  ref: string;
  /** Successfully downloaded blobs. */
  filesDownloaded: Array<{ path: string; size: number }>;
  /** Paths that could not be downloaded. */
  filesFailed: string[];
  /** Detected LICENSE file, if any. */
  license: { file?: string; type?: string; text?: string } | null;
  /** GitHub API rate limit remaining after the tree call. */
  rateLimitRemaining: number;
}

/**
 * Parse a GitHub repository URL into owner, repo, and ref components.
 *
 * Supports plain repository URLs (`https://github.com/owner/repo`) and
 * tree/branch URLs (`https://github.com/owner/repo/tree/main`).
 *
 * @param url - Public GitHub repository URL.
 * @returns Parsed repo coordinates.
 * @throws when the URL is not a valid GitHub repository URL.
 */
export function parseRepoUrl(url: string): {
  owner: string;
  repo: string;
  ref: string;
};

/**
 * Derive the local output folder name for an assimilation run.
 *
 * @param repoUrl - Public GitHub repository URL.
 * @param outputFolder - Optional parent folder; the repo name is appended.
 * @returns Local base folder and repo name.
 */
export function deriveFolderName(
  repoUrl: string,
  outputFolder?: string,
): { baseFolder: string; repo: string };

/**
 * Assimilate a public GitHub repository into the local workspace.
 *
 * @param url - Public GitHub repository URL.
 * @param options - Optional settings.
 * @returns Report describing what was downloaded.
 */
export function assimilateRepo(
  url: string,
  options?: { outputFolder?: string; token?: string },
): Promise<AssimilationReport>;
