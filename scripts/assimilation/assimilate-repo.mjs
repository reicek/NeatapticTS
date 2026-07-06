import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Buffer } from 'node:buffer';

/**
 * Parse a GitHub repository URL into owner, repo, and ref components.
 *
 * Supports plain repository URLs (`https://github.com/owner/repo`) and
 * tree/branch URLs (`https://github.com/owner/repo/tree/main`).
 *
 * @param {string} url - Public GitHub repository URL.
 * @returns {{ owner: string; repo: string; ref: string }} Parsed repo coordinates.
 * @throws {Error} when the URL is not a valid GitHub repository URL.
 */
export function parseRepoUrl(url) {
  const parsed = new URL(url);
  const parts = parsed.pathname.split('/').filter(Boolean);

  if (parts.length < 2) {
    throw new Error(`Invalid GitHub repository URL: ${url}`);
  }

  const [owner, repo] = parts;
  let ref = 'HEAD';

  if (parts.length >= 4 && parts[2] === 'tree') {
    ref = parts[3];
  }

  return { owner, repo, ref };
}

/**
 * Derive the local output folder name for an assimilation run.
 *
 * @param {string} repoUrl - Public GitHub repository URL.
 * @param {string} [outputFolder] - Optional parent folder; the repo name is appended.
 * @returns {{ baseFolder: string; repo: string }} Local base folder and repo name.
 */
export function deriveFolderName(repoUrl, outputFolder) {
  const { repo } = parseRepoUrl(repoUrl);
  const baseFolder = outputFolder ? path.join(outputFolder, repo) : repo;

  return { baseFolder, repo };
}

/**
 * Assimilate a public GitHub repository into the local workspace.
 *
 * Fetches the recursive tree, downloads each blob through the raw/contents
 * ladder, preserves verbatim copies, logs failures, detects the LICENSE file,
 * and writes a summary report.
 *
 * @param {string} url - Public GitHub repository URL.
 * @param {{ outputFolder?: string; token?: string }} [options] - Optional settings.
 * @returns {Promise<AssimilationReport>} Report describing what was downloaded.
 */
export async function assimilateRepo(url, options = {}) {
  const { owner, repo, ref } = parseRepoUrl(url);
  const { baseFolder } = deriveFolderName(url, options.outputFolder);

  const authHeaders = options.token
    ? { Authorization: `Bearer ${options.token}` }
    : {};

  const treeUrl = `https://api.github.com/repos/${owner}/${repo}/git/trees/${ref}?recursive=1`;
  const treeResponse = await fetch(treeUrl, {
    headers: {
      Accept: 'application/vnd.github+json',
      ...authHeaders,
    },
  });

  if (!treeResponse.ok) {
    throw new Error(
      `GitHub tree API failed for ${owner}/${repo}: ${treeResponse.status} ${treeResponse.statusText}`,
    );
  }

  const rateLimitRemaining =
    parseInt(treeResponse.headers.get('x-ratelimit-remaining') ?? '0', 10) || 0;
  const treeData = await treeResponse.json();

  await mkdir(path.join(baseFolder, 'verbatim'), { recursive: true });
  await mkdir(path.join(baseFolder, 'notes'), { recursive: true });

  const filesDownloaded = [];
  const filesFailed = [];
  let license = null;

  const entries = treeData.tree || [];
  const blobs = entries.filter((entry) => entry.type === 'blob');

  for (const entry of blobs) {
    const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/${ref}/${entry.path}`;
    const contentsUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${entry.path}?ref=${ref}`;

    let content = null;
    let downloaded = false;

    try {
      const rawResponse = await fetch(rawUrl, { headers: { ...authHeaders } });
      if (rawResponse.ok) {
        content = await rawResponse.text();
        downloaded = true;
      }
    } catch {
      // Fall through to contents API.
    }

    if (!downloaded) {
      try {
        const contentsResponse = await fetch(contentsUrl, {
          headers: {
            Accept: 'application/vnd.github+json',
            ...authHeaders,
          },
        });
        if (contentsResponse.ok) {
          const contentsData = await contentsResponse.json();
          if (typeof contentsData.content === 'string') {
            content = Buffer.from(contentsData.content, 'base64').toString(
              'utf8',
            );
            downloaded = true;
          }
        }
      } catch {
        // Fall through to failure logging.
      }
    }

    if (downloaded) {
      const filePath = path.join(baseFolder, 'verbatim', entry.path);
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, content, 'utf8');

      filesDownloaded.push({ path: entry.path, size: content.length });

      if (entry.path.toLowerCase() === 'license') {
        license = {
          file: entry.path,
          type: detectLicenseType(content),
          text: content,
        };
      }
    } else {
      filesFailed.push(entry.path);
    }
  }

  await writeFile(
    path.join(baseFolder, 'notes', 'fetch-failures.md'),
    formatFetchFailures(filesFailed),
    'utf8',
  );

  await writeFile(
    path.join(baseFolder, 'notes', 'license-attribution.md'),
    formatLicenseAttribution(license),
    'utf8',
  );

  await writeFile(
    path.join(baseFolder, `${repo}.md`),
    formatRepoSummary(
      url,
      owner,
      repo,
      ref,
      filesDownloaded,
      filesFailed,
      rateLimitRemaining,
    ),
    'utf8',
  );

  return {
    baseFolder,
    repo,
    owner,
    ref,
    filesDownloaded,
    filesFailed,
    license,
    rateLimitRemaining,
  };
}

/**
 * @typedef {object} AssimilationReport
 * @property {string} baseFolder - Absolute or relative workspace folder.
 * @property {string} repo - Repository name.
 * @property {string} owner - Repository owner.
 * @property {string} ref - Git ref that was fetched.
 * @property {Array<{ path: string; size: number }>} filesDownloaded - Successfully downloaded blobs.
 * @property {string[]} filesFailed - Paths that could not be downloaded.
 * @property {{ file?: string; type?: string; text?: string }|null} license - Detected LICENSE file, if any.
 * @property {number} rateLimitRemaining - GitHub API rate limit remaining after the tree call.
 */

// Minimal CLI entry point so the file can be executed directly.
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))
) {
  const [url, outputFolder] = process.argv.slice(2);
  assimilateRepo(url, { outputFolder })
    .then((report) => {
      console.log(JSON.stringify(report, null, 2));
    })
    .catch((error) => {
      console.error(error.message);
      process.exit(1);
    });
}

/**
 * Guess the license family from the LICENSE file text.
 *
 * @param {string} text - LICENSE file contents.
 * @returns {string} Detected license family or 'unknown'.
 */
function detectLicenseType(text) {
  if (/MIT/i.test(text)) return 'MIT';
  if (/Apache/i.test(text)) return 'Apache';
  if (/BSD/i.test(text)) return 'BSD';
  if (/GPL/i.test(text)) return 'GPL';
  if (/Mozilla/i.test(text)) return 'MPL';
  return 'unknown';
}

/**
 * Format the fetch-failures note file.
 *
 * @param {string[]} filesFailed - Paths that failed to download.
 * @returns {string} Markdown content.
 */
function formatFetchFailures(filesFailed) {
  const items =
    filesFailed.length > 0
      ? filesFailed.map((filePath) => `- ${filePath}`).join('\n')
      : 'None.';
  return `# Fetch Failures\n\n${items}\n`;
}

/**
 * Format the license-attribution note file.
 *
 * @param {{ file?: string; type?: string; text?: string }|null} license - Detected license.
 * @returns {string} Markdown content.
 */
function formatLicenseAttribution(license) {
  if (!license) {
    return '# License Attribution\n\nNo LICENSE file detected.\n';
  }

  return `# License Attribution\n\nFile: ${license.file}\nType: ${license.type}\n\n\`\`\`\n${license.text}\n\`\`\`\n`;
}

/**
 * Format the repository summary file.
 *
 * @param {string} url - Repository URL.
 * @param {string} owner - Repository owner.
 * @param {string} repo - Repository name.
 * @param {string} ref - Git ref.
 * @param {Array<{ path: string; size: number }>} filesDownloaded - Downloaded files.
 * @param {string[]} filesFailed - Failed paths.
 * @param {number} rateLimitRemaining - Remaining API calls.
 * @returns {string} Markdown content.
 */
function formatRepoSummary(
  url,
  owner,
  repo,
  ref,
  filesDownloaded,
  filesFailed,
  rateLimitRemaining,
) {
  return `# ${repo}\n\nAssimilated from ${url}.\n\n- Owner: ${owner}\n- Ref: ${ref}\n- Files downloaded: ${filesDownloaded.length}\n- Files failed: ${filesFailed.length}\n- Rate limit remaining: ${rateLimitRemaining}\n`;
}
