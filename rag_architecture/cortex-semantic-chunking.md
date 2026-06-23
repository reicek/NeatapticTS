# Cortex Semantic Chunking Architecture

> Extracted from `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md` (Step 02) for permanent reference.

> **Backing database.** The Repo Cortex is backed by a single consolidated Turso (libSQL) database accessed via the fully async `@libsql/client` driver (default local embedded replica `data/turso-replica.sqlite`; cloud primary `libsql://<db>.turso.io`). Vectors use native Turso vectors with `F8_BLOB` 8-bit quantization, approximate nearest neighbor search runs server-side via DiskANN (`libsql_vector_idx`, `vector_top_k()`), and hybrid ranking is performed SQL-side via Reciprocal Rank Fusion (RRF, k=60). The historical design content below describes the pre-Turso architecture that was subsequently migrated to this stack.

Complete design for AST-aware TypeScript chunking and structure-aware markdown chunking with cross-chunk context headers.

### Semantic Chunking Architecture — Complete Design

#### Design A: TypeScript AST-Aware Chunking

**Problem statement.** The current `ts-chunker.mjs` produces one chunk per exported symbol with no size limit, creating chunks from 20 chars to 42,084 chars. The embedding model (`all-MiniLM-L6-v2`) truncates at 512 tokens (~2,048 chars), meaning 95% of a 42K-char chunk is invisible to the embedding. 2,604 ts-source chunks are under 500 chars — semantically empty stubs that pollute the index. Duplicate `heading_path` values (e.g., `default` appears 15 times) prevent disambiguation.

**A.1 Two-level chunking architecture.**

Level 1 — **AST boundary detection** (existing, enhanced):

- Use ts-morph `Project` → `getExportedDeclarations()` to traverse all exported symbols
- For each exported symbol, compute total source text length
- If ≤ 1,500 chars: emit a single chunk (depth=0, no sub-chunks)
- If > 1,500 chars: apply Level 2 sub-chunking at declaration-member boundaries

Level 2 — **Declaration-member sub-chunking** (new):

| Declaration type          | Sub-chunking strategy                                                                                                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Class                     | Class signature + JSDoc → parent chunk (depth=0). Each method/property with body → sub-chunk (depth=1). Methods ≤ 300 chars are grouped into a "small methods" sub-chunk per class. |
| Interface                 | Interface signature → parent chunk (depth=0). Property groups ≤ 1,500 chars → sub-chunk (depth=1). Properties grouped by logical section (if JSDoc sections exist) or batched.      |
| Type alias                | If union/intersection > 1,500 chars → parent chunk with signature, member groups as sub-chunks. Otherwise single chunk.                                                             |
| Function                  | If > 1,500 chars (rare, usually large switch statements) → split at statement-group boundaries (coherent blocks between blank lines).                                               |
| Variable (arrow function) | Treat arrow function body like a function declaration. Split at statement-group boundaries if > 1,500 chars.                                                                        |
| Variable (constant)       | Merge with parent module context if < 100 chars. Keep as single chunk if 100–1,500 chars.                                                                                           |

**A.2 Re-export and barrel-file merging.**

Current problem: 2,604 chunks under 500 chars, many of which are bare re-exports like `export { Network } from './src/architecture/network'`.

Merging rules:

- **Re-exports** (`export { X } from './path'`): Merge all re-exports from the same barrel file into one "module index" chunk. The chunk body contains the file path and all re-exported symbol names. Context header includes the barrel file path.
- **Type aliases < 100 chars** with no JSDoc: Merge into the parent module context chunk.
- **Small constants < 100 chars**: Merge into the parent module context chunk.
- **Legitimate small exports** (100–500 chars with meaningful JSDoc or signature): Keep as single depth=0 chunks with enriched context header.

This eliminates approximately 80% of the under-500-char stubs while preserving legitimate small symbols.

**A.3 Cross-chunk context headers.**

Every ts-source chunk receives a `context_header` string with the format:

```
[<file_path> > <parent_symbol> > <member_symbol>]
```

Examples:

- Parent: `[src/neat.ts > Neat]`
- Method sub-chunk: `[src/neat.ts > Neat > evolve]`
- Merged re-exports: `[src/architecture/index.ts > module-index]`

The `heading_path` column is enriched from bare `symbol_name` to `symbol_name > member_name` for sub-chunks:

- Parent chunk: `heading_path = "Network"`
- Method sub-chunk: `heading_path = "Network > activate"`

The `context_header` is stored as a separate column, NOT prepended to `body_text`. This keeps `body_text` clean for embedding computation while providing agents with provenance context.

**A.4 Overlap strategy for TypeScript sub-chunks.**

When a class method exceeds 1,500 chars and requires further splitting:

- Include the parent class signature (up to 256 chars) as the overlap prefix in `body_text`
- Split at statement-group boundaries (blank lines between logical blocks)
- Never split inside a control flow structure (if/else, switch case, try/catch, loop body)
- Sub-chunk overlap: the last 256 chars of the previous chunk are prepended as overlap context

For class-level chunks:

- Parent chunk contains: class signature + class-level JSDoc + constructor + property declarations
- Method sub-chunks contain: method signature + method JSDoc + method body
- Each method sub-chunk includes the class signature in `context_header` (not `body_text`)

**A.5 Chunk size targets.**

| Metric                             | Target                 | Rationale                                             |
| ---------------------------------- | ---------------------- | ----------------------------------------------------- |
| Body text target                   | 800–1,500 chars        | Optimal for 512-token embedding window (~2,048 chars) |
| Hard maximum                       | 2,048 chars            | Never exceed embedding model context window           |
| Minimum viable                     | 100 chars              | Below this, merge into parent or barrel chunk         |
| Overlap                            | 256 chars              | ~2–3 sentences of context, includes parent signature  |
| Sub-chunk grouping (small methods) | ≤ 1,500 chars combined | Group adjacent small methods into one sub-chunk       |

**A.6 TypeScript chunking algorithm (pseudocode).**

```
chunkTypeScriptSourcesV2():
  declarations = loadExportedTypeScriptDeclarations()

  // Phase 1: Collect all declarations grouped by file
  fileGroups = groupBy(declarations, d => d.file_path)

  for each (filePath, fileDeclarations) in fileGroups:
    chunks = []

    // Separate re-exports from substantive declarations
    (reexports, substantive) = partition(fileDeclarations, isReexport)

    // Merge re-exports into a module-index chunk
    if reexports.length > 0:
      chunks.push(createModuleIndexChunk(filePath, reexports))

    for each declaration in substantive:
      sourceText = declaration.getText()
      sourceLength = sourceText.length

      if sourceLength <= 1500:
        // Single symbol chunk
        chunks.push(createSymbolChunk(declaration, depth=0))
      else if isClassDeclaration(declaration):
        chunks.push(...createClassChunks(declaration))
      else if isInterfaceDeclaration(declaration):
        chunks.push(...createInterfaceChunks(declaration))
      else:
        chunks.push(...createLargeSymbolChunks(declaration))

    assignChunkIndices(chunks)

  return all chunks

createClassChunks(classDecl):
  parentChunk = createClassParentChunk(classDecl)
  methodChunks = []
  smallMethods = []

  for each method in classDecl.getMethods():
    methodText = method.getText()
    if methodText.length <= 300:
      smallMethods.push(method)
    else:
      methodChunks.push(createMethodSubChunk(
        parent=parentChunk,
        method=method,
        contextHeader=buildContextHeader(classDecl, method)
      ))

  if smallMethods.length > 0:
    methodChunks.push(createSmallMethodsChunk(
      parent=parentChunk,
      methods=smallMethods,
      contextHeader=buildContextHeader(classDecl, "small-methods")
    ))

  return [parentChunk, ...methodChunks]
```

---

#### Design B: Structure-Aware Markdown Chunking

**Problem statement.** The current `chunker.mjs` splits purely on heading boundaries + character count. 51.4% of chunks have no `heading_path`. Overlap is character-based, splitting mid-word or mid-sentence. Code blocks and tables can be split mid-fence. Sub-chunks beyond the first in a section lose their section context.

**B.1 Heading hierarchy preservation.**

Every markdown chunk receives a full heading path computed from the heading stack:

```
# Architecture > ## Network > ### activate
```

Sub-chunks that continue a section inherit the full heading path with `(continued)`:

```
# Architecture > ## Network > ### activate (continued)
```

The `context_header` for markdown chunks follows the pattern:

```
[<file_path> > <heading_path>]
```

Example: `[src/architecture/network/README.md > Network > activate]`

**B.2 Semantic boundary splitting.**

When a section exceeds `maxChars` (default 2,048), split at semantic boundaries in priority order:

1. **Heading boundary** — split at `##`, `###`, etc. (already handled by section collection)
2. **Fenced code block boundary** — never split inside ` ``` ` fences. If a code block exceeds `maxChars`, keep it as one chunk (override max).
3. **Table boundary** — never split inside a Markdown table (`| ... |`). If a table exceeds `maxChars`, keep it as one chunk.
4. **Paragraph boundary** — split at double-newline (`\n\n`).
5. **Sentence boundary** — split at sentence-ending punctuation followed by whitespace (`. `, `! `, `? `).
6. **Character boundary** — last resort: split at `maxChars` if no semantic boundary is found within a 20% tolerance window.

The splitting algorithm:

```
splitSectionAtSemanticBoundary(text, maxChars, overlapChars):
  if text.length <= maxChars:
    return [text]

  // Find the best split point within tolerance window
  toleranceStart = maxChars * 0.8  // 80% of maxChars
  toleranceEnd = maxChars * 1.2    // 120% of maxChars

  splitPoint = findBestSplitPoint(text, toleranceStart, toleranceEnd)

  if no splitPoint found:
    // Hard split at maxChars (never exceed hard max)
    splitPoint = maxChars

  firstPart = text.slice(0, splitPoint)
  secondPart = text.slice(splitPoint)

  // Add overlap context from end of first part
  if overlapChars > 0 and secondPart.length > 0:
    overlapText = firstPart.slice(-overlapChars)
    secondPart = overlapText + secondPart

  return [firstPart, ...splitSectionAtSemanticBoundary(secondPart, maxChars, overlapChars)]
```

**B.3 Code block awareness.**

The chunker must detect fenced code blocks (` ``` `) and treat them as atomic units:

- If a code block starts before the split point and ends after it, move the split point past the closing fence
- If a code block exceeds `maxChars` × 2, it becomes its own chunk regardless (unavoidable large code blocks)
- The `context_header` for code block chunks includes the language tag if present: `[path > heading > typescript]`

**B.4 Overlap strategy for markdown.**

Replace the current character-based 512-char overlap with **sentence-boundary overlap**:

- Overlap target: 256 chars (reduced from 512)
- Find the nearest sentence boundary within ±20% of the target overlap
- Include the section heading as overlap context (not just trailing text)
- Overlap text is prepended to the continuation chunk, NOT included in the embedding body (stored separately)

Implementation:

```
computeOverlap(firstChunkText, overlapTarget=256):
  // Find the last sentence boundary within tolerance of overlapTarget
  searchStart = firstChunkText.length - overlapTarget - 50
  searchEnd = firstChunkText.length - overlapTarget + 50

  // Look for sentence-ending punctuation
  overlapStart = findLastSentenceBoundary(firstChunkText, searchStart, searchEnd)

  return firstChunkText.slice(overlapStart)
```

**B.5 Markdown chunking algorithm (pseudocode).**

```
chunkMarkdownV2(markdownText, options = {}):
  maxChars = options.maxChars ?? 2048
  overlapChars = options.overlapChars ?? 256

  sections = collectSectionsWithHierarchy(markdownText)
  chunks = []

  for each section in sections:
    if section.bodyText.length <= maxChars:
      chunks.push({
        heading_path: section.fullHeadingPath,
        body_text: section.bodyText,
        context_header: buildContextHeader(section),
        char_start: section.charStart,
        char_end: section.charEnd,
        depth: section.headingLevel,
      })
    else:
      // Split at semantic boundaries
      subChunks = splitAtSemanticBoundaries(
        section.bodyText,
        maxChars,
        overlapChars,
        preserveCodeBlocks=true,
        preserveTables=true,
      )

      for each (index, subChunk) in subChunks:
        chunks.push({
          heading_path: index === 0
            ? section.fullHeadingPath
            : section.fullHeadingPath + " (continued)",
          body_text: subChunk.text,
          context_header: buildContextHeader(section),
          char_start: section.charStart + subChunk.offset,
          char_end: section.charStart + subChunk.offset + subChunk.text.length,
          depth: section.headingLevel,
        })

  return chunks

collectSectionsWithHierarchy(markdownText):
  headings = findHeadings(markdownText)  // regex: /^(#{1,6})\s+(.+)$/gm
  stack = []
  sections = []

  for each heading in headings:
    // Maintain heading hierarchy stack
    while stack.length > 0 and stack[stack.length-1].level >= heading.level:
      stack.pop()
    stack.push(heading)

    // Full heading path from root to current
    fullHeadingPath = stack.map(h => h.text).join(' > ')

    // Extract section body until next heading at same or higher level
    bodyText = extractSectionBody(markdownText, heading, nextHeading)

    sections.push({
      fullHeadingPath,
      bodyText,
      headingLevel: heading.level,
      charStart: heading.index,
      charEnd: nextHeading?.index ?? markdownText.length,
    })

  return sections
```

---

#### Design C: Schema Changes

**C.1 New columns for `chunks` table.**

```sql
-- Schema v2: Add semantic chunking columns
ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER;
ALTER TABLE chunks ADD COLUMN depth INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN context_header TEXT;
ALTER TABLE chunks ADD COLUMN symbol_name TEXT;
ALTER TABLE chunks ADD COLUMN signature_text TEXT;
ALTER TABLE chunks ADD COLUMN jsdoc_text TEXT;
ALTER TABLE chunks ADD COLUMN export_type TEXT;
ALTER TABLE chunks ADD COLUMN module_path TEXT;
```

| Column            | Type    | Default | Purpose                                                                     |
| ----------------- | ------- | ------- | --------------------------------------------------------------------------- |
| `parent_chunk_id` | INTEGER | NULL    | References the parent symbol chunk for sub-chunks (depth > 0)               |
| `depth`           | INTEGER | 0       | 0 = top-level symbol/section, 1 = method/property sub-chunk                 |
| `context_header`  | TEXT    | NULL    | Cross-chunk context: `[file_path > heading_path > signature]`               |
| `symbol_name`     | TEXT    | NULL    | Exported symbol name (ts-source only)                                       |
| `signature_text`  | TEXT    | NULL    | Function/class/interface signature (ts-source only)                         |
| `jsdoc_text`      | TEXT    | NULL    | JSDoc summary text (ts-source only)                                         |
| `export_type`     | TEXT    | NULL    | One of: `function`, `class`, `interface`, `type`, `variable`, `reexport`    |
| `module_path`     | TEXT    | NULL    | Folder-based module path, e.g., `src/architecture/network` (ts-source only) |

**C.2 Schema migration strategy.**

Migration is performed by `scripts/semantic-index/migrate-schema.mjs`:

1. Add new columns with DEFAULT values (NULL for text, 0 for depth) — backward compatible, existing queries continue to work
2. Run `build-index.mjs --force` to re-chunk the entire corpus with the new chunkers, populating all new columns
3. Run `embed-index.mjs` to re-embed all chunks (chunk SHA-256 changes because `context_header` and new metadata are included in the hash)
4. The existing `(doc_id, chunk_index)` UNIQUE constraint is preserved; `chunk_index` is now a sequential counter per document

**C.3 chunk_sha256 computation update.**

The embedding pipeline's `chunk_sha256` must include the new columns to detect content changes:

```javascript
createHash('sha256')
  .update(
    JSON.stringify({
      body_text,
      char_end,
      char_start,
      chunk_id,
      chunk_index,
      context_header, // NEW
      depth, // NEW
      doc_family,
      file_path,
      heading_path,
      symbol_name, // NEW
    }),
  )
  .digest('hex');
```

Note: `signature_text`, `jsdoc_text`, `export_type`, `module_path` are intentionally excluded from the SHA-256 hash because they are metadata that should not change the embedding — only `context_header`, `depth`, and `symbol_name` affect chunk identity.

**C.4 Backward compatibility.**

- Existing `search_corpus`, `load_chunk`, `load_document` tools continue to work because new columns have DEFAULT values
- New columns are returned in `readChunkRow()` automatically via `SELECT *`
- The `chunks_fts` FTS5 table continues to index `body_text` and `heading_path`; `context_header` is NOT indexed by FTS (it is agent-facing metadata, not search content)
- MCP tool `search_corpus` gains an optional `metadata` parameter in a later step (Step 09) for structured filtering

---

#### Design D: Re-Chunking Strategy

**D.1 Incremental rebuild mechanism.**

The existing freshness-proof mechanism (mtime + size + SHA-256) handles file-level incremental rebuild. When a file is unchanged, its chunks are skipped entirely. When a file changes, all its old chunks are deleted and new ones are inserted.

The key insight: **chunk IDs are not stable across re-chunking.** When a file changes, its old `doc_id` row is updated and all old chunks are deleted. New chunks get new `chunk_id` values. This is already the behavior of `build-index.mjs`.

For the embedding pipeline:

- `chunk_sha256` is computed from the new fields including `context_header`
- When the hash changes, the embedding is recomputed
- When the hash matches, the embedding is reused (incremental skip)
- This means the one-time schema migration triggers a full re-embed, which is acceptable

**D.2 Migration plan.**

| Step | Action                                          | Impact                                               |
| ---- | ----------------------------------------------- | ---------------------------------------------------- |
| 1    | Run `migrate-schema.mjs` to add new columns     | Columns added with defaults; no data loss            |
| 2    | Run `build-index.mjs --force` with new chunkers | All chunks re-created with enriched metadata         |
| 3    | Run `embed-index.mjs`                           | All embeddings recomputed (chunk SHA-256 changed)    |
| 4    | Verify with `eval-embeddings.mjs`               | Baseline MRR@5 should improve due to better chunking |
| 5    | Update `readChunkRow()` in `cortex-db.mjs`      | Return new columns in search results                 |
| 6    | Update MCP tool schemas                         | Add new fields to `search_corpus` response           |

**D.3 Parent-child chunk relationship.**

The `parent_chunk_id` column creates a tree structure:

- depth=0 chunks: `parent_chunk_id = NULL` (top-level symbols/sections)
- depth=1 chunks: `parent_chunk_id` references the depth=0 parent chunk

For `load_document`, the response now includes the hierarchy:

```json
{
  "file_path": "src/neat.ts",
  "chunks": [
    { "chunk_id": 42, "depth": 0, "heading_path": "Neat", "parent_chunk_id": null, ... },
    { "chunk_id": 43, "depth": 1, "heading_path": "Neat > evolve", "parent_chunk_id": 42, ... },
    { "chunk_id": 44, "depth": 1, "heading_path": "Neat > create", "parent_chunk_id": 42, ... }
  ]
}
```

For `search_corpus`, sub-chunks are returned independently (a query about `evolve` returns the method sub-chunk, not the entire class). The `parent_chunk_id` allows agents to request the parent chunk for additional context.

---

#### Design E: File Organization

**E.1 New files.**

| File                                                                  | Purpose                                                                                      |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `scripts/semantic-index/chunker-v2.mjs`                               | Structure-aware markdown chunker (replaces `chunker.mjs` in build pipeline)                  |
| `scripts/semantic-index/ts-chunker-v2.mjs`                            | AST-aware TypeScript chunker with sub-chunking (replaces `ts-chunker.mjs` in build pipeline) |
| `scripts/semantic-index/schema-v2.sql`                                | Schema migration DDL from v1 to v2                                                           |
| `scripts/semantic-index/migrate-schema.mjs`                           | Schema migration runner                                                                      |
| `scripts/semantic-index/ts-chunk/__tests__/ts-chunker-v2.red.test.ts` | Red tests for new TypeScript chunker                                                         |
| `scripts/semantic-index/__tests__/chunker-v2.red.test.ts`             | Red tests for new markdown chunker                                                           |

**E.2 Modified files.**

| File                                           | Change                                                                                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `scripts/semantic-index/build-index.mjs`       | Import `chunkMarkdownV2` from `chunker-v2.mjs` and `chunkTypeScriptSourcesV2` from `ts-chunker-v2.mjs`; update `chunkDocument()` to use v2 chunkers; update `insertChunk` to include new columns |
| `scripts/semantic-index/embed-index.mjs`       | Update `chunk_sha256` computation to include new fields                                                                                                                                          |
| `scripts/semantic-index/schema.sql`            | Replace with v2 schema (all new columns included)                                                                                                                                                |
| `scripts/semantic-index/init-schema.mjs`       | Update `initSemanticIndex()` to create v2 schema                                                                                                                                                 |
| `scripts/mcp-semantic/tools/cortex-db.mjs`     | Update `readChunkRow()` to return new columns                                                                                                                                                    |
| `scripts/mcp-semantic/tools/search-corpus.mjs` | Return new columns in search results                                                                                                                                                             |
| `scripts/mcp-semantic/tools/load-chunk.mjs`    | Return new columns; add optional `parent_chunk_id` query                                                                                                                                         |
| `scripts/mcp-semantic/tools/load-document.mjs` | Return new columns with hierarchy info                                                                                                                                                           |
| `scripts/mcp-semantic/repo-cortex-mcp.mjs`     | Update tool schemas for new fields                                                                                                                                                               |
| `scripts/semantic-index/chunker.d.mts`         | Update type definitions for v2 chunk types                                                                                                                                                       |

**E.3 Backward compatibility approach.**

- `chunker.mjs` and `ts-chunker.mjs` are preserved as-is (not deleted) for reference and rollback
- `build-index.mjs` switches to v2 chunkers via import change
- The v2 chunkers export the same interface shape (`chunkMarkdownV2` returns `MarkdownChunkV2[]` with additional fields; `chunkTypeScriptSourcesV2` returns similar shape with additional fields)
- The `MarkdownChunk` and `TypeScriptChunk` types are extended, not replaced
- Old chunker tests remain; new v2 tests are added alongside

---

#### Design F: MCP Tool Contract Changes

**F.1 `search_corpus` response changes.**

The `search_corpus` tool response gains these new fields:

```json
{
  "results": [
    {
      "chunk_id": 42,
      "file_path": "src/neat.ts",
      "family": "ts-source",
      "chunk_index": 0,
      "heading_path": "Neat > evolve",
      "text": "...",
      "char_start": 1234,
      "char_end": 2345,
      "score": 0.85,
      "depth": 1,
      "parent_chunk_id": 41,
      "context_header": "[src/neat.ts > Neat > evolve]",
      "symbol_name": "evolve",
      "signature_text": "evolve(inputs: number[][], fitnessFunction: FitnessFunction): Network[]",
      "jsdoc_text": "Evolves the population...",
      "export_type": "method",
      "module_path": "src/neat"
    }
  ]
}
```

New fields: `depth`, `parent_chunk_id`, `context_header`, `symbol_name`, `signature_text`, `jsdoc_text`, `export_type`, `module_path`.

**F.2 New tool: `load_parent_chunk`.**

```json
{
  "name": "load_parent_chunk",
  "description": "Load the parent chunk for a sub-chunk, providing class-level or section-level context.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "chunk_id": {
        "type": "number",
        "description": "The sub-chunk ID whose parent to load."
      }
    },
    "required": ["chunk_id"]
  }
}
```

This allows agents to request the parent class/section chunk when a method/property sub-chunk is returned, enabling "zoom out" context navigation.

**F.3 Updated `load_document` response.**

The `load_document` response now includes hierarchy information:

```json
{
  "file_path": "src/neat.ts",
  "chunks": [...],
  "hierarchy": {
    "depth_0_count": 15,
    "depth_1_count": 42,
    "has_sub_chunks": true
  }
}
```

---

#### Design G: Embedding Impact Analysis

**G.1 How chunking quality improves embedding relevance.**

Current system: A 42K-char `onnxImportOrchestratorsUtils` chunk gets embedded as a single vector from only the first ~2K chars. The remaining 40K chars are invisible to the embedding model. When an agent asks "how do I import ONNX models?", the chunk may not match because the relevant code is in the invisible 95%.

With AST-aware chunking: The same file produces ~30 method sub-chunks, each 500–1,500 chars. Each sub-chunk is fully embedded, making method-level queries highly relevant. The parent chunk provides class-level context via `context_header` (not embedded, but available to agents).

**G.2 Expected MRR@5 improvement.**

Based on the audit findings:

- Fixing ts-source chunking alone should improve MRR@5 by 0.05–0.10 (currently 8/20 hybrid queries score 0 because the relevant content is in an oversized chunk)
- Fixing markdown heading context should improve MRR@5 by 0.02–0.05 (eliminates the 51.4% heading_path gap)
- Combined chunking improvements expected: +0.07–0.15 MRR@5 absolute improvement
- This is a larger improvement than any single model upgrade would provide

**G.3 Embedding model compatibility.**

- `all-MiniLM-L6-v2`: 512 token max (~2,048 chars). Target chunk size of 800–1,500 chars fits well within this window with room for tokenization overhead.
- Future `bge-small-en-v1.5` or `all-MiniLM-L12-v2`: Same 512 token window. Chunk targets remain appropriate.
- The architecture supports model hot-swapping via `model_id` + `model_sha256` in `model-meta.json`.

---

#### Design H: Validation Criteria

**H.1 Chunking quality metrics (post-implementation).**

| Metric                                   | Current                           | Target                              |
| ---------------------------------------- | --------------------------------- | ----------------------------------- |
| ts-source chunks > 5K chars              | 47                                | 0                                   |
| ts-source chunks < 100 chars             | ~800 (re-exports)                 | 0 (merged into module-index chunks) |
| ts-source chunks with empty heading_path | ~2,604                            | 0                                   |
| Markdown chunks with empty heading_path  | ~16,133 (51.4%)                   | 0                                   |
| Average ts-source chunk size             | ~1,100 chars (skewed by outliers) | 800–1,200 chars                     |
| Median ts-source chunk size              | ~200 chars                        | 600–1,000 chars                     |
| MRR@5 (hybrid)                           | 0.308                             | ≥ 0.38                              |

**H.2 Acceptance tests.**

1. All ts-source chunks are between 100 and 2,048 chars (hard limits)
2. No ts-source chunk exceeds 2,048 chars body_text length
3. Every ts-source chunk has a non-empty `heading_path` (symbol_name for depth=0, symbol_name > member for depth=1)
4. Every markdown chunk has a non-empty `heading_path` (full heading hierarchy)
5. Every chunk (ts-source and markdown) has a non-empty `context_header`
6. Parent-child relationships are consistent: `parent_chunk_id` references a valid depth=0 chunk
7. Re-export stubs are merged into module-index chunks (no single-symbol re-export chunks)
8. The eval-queries.json baseline MRR@5 does not regress
9. `build-index.mjs --force` completes without error on the full corpus
10. `embed-index.mjs` completes without error on the re-chunked corpus

**Step completion validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`

**Stop conditions:** ✅ COMPLETE — semantic chunking architecture designed with AST-aware TypeScript sub-chunking, structure-aware markdown chunking, cross-chunk context headers, chunk size targets, overlap strategy, re-chunking strategy, schema changes, file organization, MCP contract changes, and validation criteria.

---

> **Source:** `plans/Repo_Cortex_Advanced_RAG_Architecture.plans.md`, lines 475–1063.
> **Extracted:** 2026-06-08. Content preserved verbatim for permanent reference.
