const DEFAULT_MAX_CHARS = 2048;
const DEFAULT_OVERLAP_CHARS = 512;
const HEADING_PATTERN = /^(#{1,6})\s+(.+)$/gm;

export function chunkMarkdown(markdownText, options = {}) {
  const maxChars = Math.max(1, Number(options.maxChars ?? DEFAULT_MAX_CHARS));
  const overlapChars = Math.max(
    0,
    Math.min(
      Number(options.overlapChars ?? DEFAULT_OVERLAP_CHARS),
      maxChars - 1,
    ),
  );
  const sections = collectSections(String(markdownText));

  return sections.flatMap((section) =>
    chunkSection(section, { maxChars, overlapChars }),
  );
}

function collectSections(markdownText) {
  const headings = [...markdownText.matchAll(HEADING_PATTERN)].map((match) => ({
    marker: match[1],
    text: match[2].trim(),
    index: match.index,
    lineEnd:
      markdownText.indexOf('\n', match.index) === -1
        ? markdownText.length
        : markdownText.indexOf('\n', match.index),
  }));

  if (headings.length === 0) {
    return [
      {
        heading_path: '',
        body_text: markdownText.trim(),
        char_start: 0,
        char_end: markdownText.length,
      },
    ];
  }

  const headingStack = [];
  return headings
    .map((heading, headingIndex) => {
      const headingLevel = heading.marker.length;
      while (headingStack.length >= headingLevel) headingStack.pop();
      headingStack.push(`${heading.marker} ${heading.text}`);

      const nextHeading = headings[headingIndex + 1];
      const charStart = heading.index;
      const charEnd = nextHeading?.index ?? markdownText.length;
      return {
        heading_path: headingStack.join(' > '),
        body_text: markdownText.slice(heading.lineEnd, charEnd).trim(),
        char_start: charStart,
        char_end: charEnd,
      };
    })
    .filter(
      (section) =>
        section.body_text.length > 0 || section.heading_path.length > 0,
    );
}

function chunkSection(section, options) {
  const sectionText = section.body_text || section.heading_path;
  const chunks = [];
  let offset = 0;

  while (offset < sectionText.length) {
    const endOffset = Math.min(sectionText.length, offset + options.maxChars);
    chunks.push({
      heading_path: section.heading_path,
      body_text: sectionText.slice(offset, endOffset),
      char_start: section.char_start + offset,
      char_end: section.char_start + endOffset,
    });

    if (endOffset === sectionText.length) break;
    offset = endOffset - options.overlapChars;
  }

  return chunks;
}
