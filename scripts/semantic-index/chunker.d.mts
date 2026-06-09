export interface MarkdownChunk {
  heading_path: string;
  body_text: string;
  char_start: number;
  char_end: number;
}

export interface ChunkMarkdownOptions {
  maxChars?: number;
  overlapChars?: number;
}

export function chunkMarkdown(markdownText: string, options?: ChunkMarkdownOptions): MarkdownChunk[];

/** V2 markdown chunk with hierarchy metadata and context headers. */
export interface MarkdownChunkV2 {
  heading_path: string;
  body_text: string;
  char_start: number;
  char_end: number;
  depth: number;
  context_header: string;
  parent_chunk_id: number | null;
  symbol_name: string | null;
  signature_text: string | null;
  jsdoc_text: string | null;
  export_type: string | null;
  module_path: string | null;
}

/** V2 markdown chunking options with file path for context headers. */
export interface ChunkMarkdownV2Options {
  maxChars?: number;
  overlapChars?: number;
  filePath?: string;
}

export function chunkMarkdownV2(markdownText: string, options?: ChunkMarkdownV2Options): MarkdownChunkV2[];

/** V2 TypeScript chunk with sub-chunking metadata and context headers. */
export interface TypeScriptChunkV2 {
  body_text: string;
  char_end: number;
  char_start: number;
  chunk_index: number;
  context_header: string;
  depth: number;
  doc_family: string;
  export_type: string;
  file_path: string;
  heading_path: string;
  jsdoc_text: string;
  module_path: string;
  parent_chunk_id: number | null;
  signature_text: string;
  symbol_name: string;
}

export function chunkTypeScriptSourcesV2(options?: {
  sourcePaths?: string[];
  patterns?: string[];
  ignore?: string[];
  project?: unknown;
}): Promise<TypeScriptChunkV2[]>;