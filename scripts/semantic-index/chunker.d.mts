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