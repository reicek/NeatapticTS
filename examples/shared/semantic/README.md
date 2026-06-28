# Semantic Snapshot — Shared Browser Modules

Browser-friendly utilities for loading and searching the NeatapticTS Repo Cortex semantic
snapshot. Any demo under `examples/` can import these modules to show contextual
documentation alongside its UI.

## Modules

| File                          | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `semantic-snapshot-loader.ts` | Fetch and cache the generated snapshot via IndexedDB     |
| `semantic-snapshot-search.ts` | Heading-weighted lexical search over the loaded snapshot |
| `semantic-snapshot.types.ts`  | Shared TypeScript interfaces                             |

---

## Snapshot format

`docs/assets/semantic-snapshot.json` is a **generated artifact** produced by the docs
pipeline. Never edit it directly — it is regenerated on every `npm run docs` run.

**To update the corpus:**

```sh
# 1. Edit source documents (READMEs, plans, skills, etc.)
# 2. Rebuild the SQLite index:
node scripts/semantic-index/build-index.mjs
# 3. Publish the snapshot (or let npm run docs do it):
node scripts/semantic-index/build-browser-snapshot.mjs
```

### Top-level schema

```jsonc
{
  "schema_version": "1",
  "generated_at": "2026-05-23T14:32:00.725Z",
  "families": ["readme", "skill", "agent", "plan", "demo"],
  "documents": [/* SemanticSnapshotDocument[] */],
}
```

| Field            | Type                         | Notes                                                |
| ---------------- | ---------------------------- | ---------------------------------------------------- |
| `schema_version` | `"1"`                        | Loader rejects payloads with any other value         |
| `generated_at`   | ISO timestamp                | Cache freshness key — also the IndexedDB primary key |
| `families`       | `string[]`                   | Corpus families present in this build                |
| `documents`      | `SemanticSnapshotDocument[]` | Repository files with nested chunks                  |

### SemanticSnapshotDocument

| Field       | Type                      | Notes                                                        |
| ----------- | ------------------------- | ------------------------------------------------------------ |
| `doc_id`    | `number`                  | Stable SQLite primary key                                    |
| `file_path` | `string`                  | Repository-relative path (no absolute local paths)           |
| `family`    | `string`                  | Corpus family: `readme`, `plan`, `skill`, `agent`, or `demo` |
| `chunks`    | `SemanticSnapshotChunk[]` | Ordered by source position                                   |

### SemanticSnapshotChunk

| Field          | Type     | Notes                                                             |
| -------------- | -------- | ----------------------------------------------------------------- |
| `chunk_id`     | `number` | Stable SQLite primary key                                         |
| `heading_path` | `string` | Markdown heading trail giving the chunk its documentation context |
| `body_text`    | `string` | Plain text body for lexical search                                |
| `char_start`   | `number` | Inclusive character offset in the source document                 |
| `char_end`     | `number` | Exclusive character offset in the source document                 |

---

## Generated output contract

`docs/assets/semantic-snapshot.json` is published as part of Stage 1 of the docs pipeline
(see `scripts/run-docs.ts → runFullDocsWorkflow`). The snapshot generator runs before the
browser bundles and folder docs are copied, so the served file is always in sync with the
SQLite index at build time.

**Treat `docs/assets/semantic-snapshot.json` as read-only** — the same rule that applies to
all generated artifacts under `docs/` (see the `educational-docs` skill and the
`implementation-standards` skill for generated-artifact handling).

---

## Loader: IndexedDB cache behavior

`loadSemanticSnapshot(url, options?)` fetches the snapshot and caches it in IndexedDB.

```ts
import { loadSemanticSnapshot } from '../shared/semantic/semantic-snapshot-loader';

const snapshot = await loadSemanticSnapshot('/assets/semantic-snapshot.json');
```

### Cache lifecycle

1. **First load** — fetches from `url`, validates schema_version `"1"`, writes to IndexedDB
   keyed by `generated_at`.
2. **Subsequent loads** — reads all cached records and picks the entry with the latest
   `generated_at` (no network round-trip).
3. **Stale detection** — pass `forceRefresh: true` to skip the cache and always fetch the
   served snapshot.
4. **No IndexedDB** — falls back to a direct network fetch without caching (Node.js test
   runners, server-side environments, or browsers with storage disabled).

### Options (`LoadSemanticSnapshotOptions`)

| Option         | Type      | Default                         | Notes                                 |
| -------------- | --------- | ------------------------------- | ------------------------------------- |
| `databaseName` | `string`  | `"neataptic-semantic-snapshot"` | IndexedDB database name               |
| `storeName`    | `string`  | `"snapshots"`                   | Object store inside the database      |
| `forceRefresh` | `boolean` | `false`                         | Bypass cache; always fetch from `url` |

---

## Search: heading-path weighting

`searchSnapshot(snapshot, query, options?)` scores chunks by how many query tokens they
contain, giving heading matches five times the weight of body matches.

```ts
import { searchSnapshot } from '../shared/semantic/semantic-snapshot-search';

const results = searchSnapshot(snapshot, 'NEAT activation', { limit: 5 });
for (const { score, document, chunk } of results) {
  console.log(score, document.file_path, chunk.heading_path);
}
```

### Scoring algorithm

1. **Tokenize** — split `query` into unique lowercase Unicode tokens (`[\p{L}\p{N}_-]+`).
2. **Score each chunk** — for every token count occurrences in `heading_path` (×5) and
   `body_text` (×1); sum over all tokens.
3. **Filter** — discard chunks with score 0.
4. **Sort** — descending score, then stable source order (document index, then chunk index).
5. **Slice** — return the top `options.limit` results (default: 10).

The algorithm is intentionally simple — no stemming, synonym expansion, or TF-IDF. Heading
weighting is the primary differentiator: chunks that name a concept in their heading rank
above those that mention it only in the body.

### Result shape (`SemanticSnapshotSearchResult`)

| Field      | Type                       | Notes                                                    |
| ---------- | -------------------------- | -------------------------------------------------------- |
| `score`    | `number`                   | Heading-weighted lexical match count; higher ranks first |
| `document` | `SemanticSnapshotDocument` | Parent repository document                               |
| `chunk`    | `SemanticSnapshotChunk`    | Matched chunk inside the document                        |

---

## Quick integration example

```ts
import { loadSemanticSnapshot } from '../shared/semantic/semantic-snapshot-loader';
import { searchSnapshot } from '../shared/semantic/semantic-snapshot-search';

// Load (cached after first fetch)
const snapshot = await loadSemanticSnapshot('/assets/semantic-snapshot.json');

// Query
const results = searchSnapshot(snapshot, 'population speciation', { limit: 5 });
results.forEach(({ score, document, chunk }) => {
  console.log(`[${score}] ${document.file_path} — ${chunk.heading_path}`);
});
```
