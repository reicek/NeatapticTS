/**
 * browserLogger.ts
 *
 * Provides a `createBrowserLogger` factory that returns a function compatible with `forceLog`.
 * The logger converts common ANSI sequences into inline HTML styles and appends
 * formatted output to a <pre> element inside the provided container.
 */

/** Minimal mapping of 256-color palette indices used by the demo to CSS hex colors. */
import { MazeUtils } from './mazeUtils';

/**
 * Mapping from 256-color ANSI palette indices to CSS hex colors (subset used by demo).
 * @remarks Kept intentionally sparse – only indices actually produced by the maze demo
 * are included to avoid allocating a full 256‑entry table. Frozen to lock hidden class.
 */
const ANSI_256_MAP: { [code: number]: string } = Object.freeze({
  205: '#ff6ac1',
  93: '#b48bf2',
  154: '#a6d189',
  51: '#00bcd4',
  226: '#ffd166',
  214: '#ff9f43',
  196: '#ff3b30',
  46: '#00e676',
  123: '#6ec6ff',
  177: '#caa6ff',
  80: '#00bfa5',
  121: '#9bdc8a',
  203: '#ff6b9f',
  99: '#6b62d6',
  44: '#00a9e0',
  220: '#ffd54f',
  250: '#ececec',
  45: '#00aaff',
  201: '#ff4fc4',
  231: '#ffffff',
  218: '#ffc6d3',
  217: '#ffcdb5',
  117: '#6fb3ff',
  118: '#6ee07a',
  48: '#00a300',
  57: '#2f78ff',
  33: '#1e90ff',
  87: '#00d7ff',
  159: '#cfeeff',
  208: '#ff8a00',
  197: '#ff5ea6',
  234: '#0e1114',
  23: '#123044',
  17: '#000b16',
  16: '#000000',
  39: '#0078ff',
});

/** Bold font-weight applied for ANSI SGR code 1. */
const FONT_WEIGHT_BOLD = '700' as const;
/** SGR code representing a full reset of styles. */
const SGR_RESET = 0 as const;
/** SGR code enabling bold weight. */
const SGR_BOLD = 1 as const;
/** SGR code disabling bold (normal intensity). */
const SGR_BOLD_OFF = 22 as const;
/** SGR parameter introducing extended foreground color (expect `38;5;<idx>` sequence). */
const SGR_FG_EXTENDED = 38 as const;
/** SGR parameter introducing extended background color (expect `48;5;<idx>` sequence). */
const SGR_BG_EXTENDED = 48 as const;
/** SGR code clearing the current foreground color only. */
const SGR_FG_DEFAULT = 39 as const;
/** SGR code clearing the current background color only. */
const SGR_BG_DEFAULT = 49 as const;

/** Regex used to detect the presence of any HTML‑sensitive characters for fast bailout. */
const HTML_ESCAPE_PRESENCE = /[&<>]/;
/** Basic 8-color foreground palette for codes 30–37. */
const BASIC_FG_COLORS = Object.freeze([
  '#000000',
  '#800000',
  '#008000',
  '#808000',
  '#000080',
  '#800080',
  '#008080',
  '#c0c0c0',
]);
/** Bright foreground palette for codes 90–97. */
const BRIGHT_FG_COLORS = Object.freeze([
  '#808080',
  '#ff0000',
  '#00ff00',
  '#ffff00',
  '#0000ff',
  '#ff00ff',
  '#00ffff',
  '#ffffff',
]);

/**
 * Escape HTML special characters in a string.
 *
 * Keeps the implementation tiny and allocation-free; used before inserting
 * strings into innerHTML to avoid XSS and layout issues.
 *
 * @param s - input string possibly containing HTML-sensitive chars
 * @returns escaped string safe for insertion into innerHTML
 */
function escapeHtml(raw: string): string {
  // Fast path: no escaping needed.
  if (!HTML_ESCAPE_PRESENCE.test(raw)) return raw;
  // Order: & first to avoid double-escaping.
  return raw.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * Ensure there is a <pre> element in the provided container (or default host)
 * and return it. If the document is not present or host can't be found,
 * returns null.
 *
 * @param container - Optional host element to place the <pre> into.
 * @returns the <pre> element or null when unavailable
 */
function ensurePre(container?: HTMLElement): HTMLPreElement | null {
  const hostElement =
    container ??
    (typeof document !== 'undefined'
      ? document.getElementById('ascii-maze-output')
      : null);
  if (!hostElement) return null;
  let preElement = hostElement.querySelector('pre');
  if (!preElement) {
    preElement = document.createElement('pre');
    preElement.style.fontFamily = 'monospace';
    preElement.style.whiteSpace = 'pre';
    preElement.style.margin = '0';
    preElement.style.padding = '4px';
    preElement.style.fontSize = '10px';
    hostElement.appendChild(preElement);
  }
  return preElement as HTMLPreElement;
}

/**
 * Internal ANSI -> HTML converter with private helpers for a declarative main flow.
 * @remarks Designed for high-frequency short strings (logging lines). Avoids
 * per-call allocations of intermediate arrays by using manual parses and
 * mutable private fields. Not reentrant: each call uses a fresh instance via
 * `convert` so callers never share internal state.
 */
class AnsiHtmlConverter {
  /**
   * Global regex used to locate SGR parameter sequences. Reset before each parse.
   * `([0-9;]*)` captures the parameter list which may be empty (equivalent to reset).
   */
  static #SgrSequencePattern = /\x1b\[([0-9;]*)m/g;

  /** Marker inserted for newline during streaming conversion (literal `<br/>`). */
  static #HtmlNewline = '<br/>' as const;

  /** Small object pool for converter instances (avoid GC churn under heavy logging). */
  static #Pool: AnsiHtmlConverter[] = [];
  static #POOL_SIZE_LIMIT = 32; // Safety cap – logging lines are short, pool doesn't need to grow large.

  /** Cache mapping style signature -> opening span tag for reuse. */
  static #StyleCache = new Map<string, string>();

  // Per-instance mutable state (cleared between uses via #resetForInput).
  #input = '';
  #htmlOutput = '';
  #lastProcessedIndex = 0;

  // Current style fields.
  #currentColor: string | undefined;
  #currentBackground: string | undefined;
  #currentFontWeight: string | undefined;
  #hasActiveStyle = false;
  #currentStyleSpanStart = '';

  // Scratch array reused for parsed numeric codes (grown as needed, not shrunk).
  #parsedCodes: number[] = [];

  private constructor() {
    /* instances created via pool only */
  }

  /** Acquire a converter instance (from pool or new). */
  static #acquire(input: string): AnsiHtmlConverter {
    const instance = this.#Pool.pop() ?? new AnsiHtmlConverter();
    instance.#resetForInput(input);
    return instance;
  }

  /** Return a used instance back into the pool (bounded). */
  static #release(instance: AnsiHtmlConverter): void {
    if (this.#Pool.length < this.#POOL_SIZE_LIMIT) {
      this.#Pool.push(instance);
    }
  }

  /** Public entry point: convert ANSI encoded text into HTML (single pass, pooled). */
  static convert(input: string): string {
    const instance = this.#acquire(input);
    try {
      instance.#process();
      return instance.#htmlOutput;
    } finally {
      this.#release(instance);
    }
  }

  /** Prepare internal state for a fresh parse. */
  #resetForInput(input: string): void {
    this.#input = input;
    this.#htmlOutput = '';
    this.#lastProcessedIndex = 0;
    this.#resetStyles();
    AnsiHtmlConverter.#SgrSequencePattern.lastIndex = 0;
  }

  /** Main processing loop: walk all SGR sequences and emit transformed HTML. */
  #process(): void {
    let ansiMatch: RegExpExecArray | null;
    while (
      (ansiMatch = AnsiHtmlConverter.#SgrSequencePattern.exec(this.#input)) !==
      null
    ) {
      this.#emitPlainTextSegment(ansiMatch.index);
      this.#applyRawCodeSequence(ansiMatch[1]);
      this.#lastProcessedIndex =
        AnsiHtmlConverter.#SgrSequencePattern.lastIndex;
    }
    this.#emitPlainTextSegment(this.#input.length); // trailing segment
  }

  /** Emit plain (non-ANSI) text between the previous index and the supplied stop. */
  #emitPlainTextSegment(stopExclusive: number): void {
    if (this.#lastProcessedIndex >= stopExclusive) return;
    const rawChunk = this.#input.substring(
      this.#lastProcessedIndex,
      stopExclusive
    );
    if (!rawChunk) return;
    // Fast path: no newline present.
    if (!rawChunk.includes('\n')) {
      const escapedSingle = escapeHtml(rawChunk);
      this.#htmlOutput += this.#wrapIfStyled(escapedSingle);
      return;
    }
    // Slow path: split by newline *without* allocating array via manual scan.
    let segmentStart = 0;
    for (let scanIndex = 0; scanIndex <= rawChunk.length; scanIndex++) {
      const isEnd = scanIndex === rawChunk.length;
      const isNewline = !isEnd && rawChunk.charCodeAt(scanIndex) === 10; // '\n'
      if (isNewline || isEnd) {
        if (scanIndex > segmentStart) {
          const sub = rawChunk.substring(segmentStart, scanIndex);
          this.#htmlOutput += this.#wrapIfStyled(escapeHtml(sub));
        }
        if (isNewline) this.#htmlOutput += AnsiHtmlConverter.#HtmlNewline;
        segmentStart = scanIndex + 1;
      }
    }
  }

  /** Apply a raw parameter string (could be empty meaning reset) to update style state. */
  #applyRawCodeSequence(rawCodes: string): void {
    if (rawCodes === '') {
      this.#resetStyles();
      return;
    }
    // Manual parse (no split/map allocations): write into #parsedCodes.
    let accumulator = '';
    let parsedCount = 0;
    for (let charIndex = 0; charIndex < rawCodes.length; charIndex++) {
      const character = rawCodes[charIndex];
      if (character === ';') {
        if (accumulator) {
          this.#parsedCodes[parsedCount++] = parseInt(accumulator, 10);
          accumulator = '';
        }
      } else {
        accumulator += character;
      }
    }
    if (accumulator)
      this.#parsedCodes[parsedCount++] = parseInt(accumulator, 10);
    this.#applyParsedCodes(parsedCount);
    this.#rebuildStyleSpanStart();
  }

  /** Apply parsed numeric codes currently buffered in #parsedCodes (length = count). */
  #applyParsedCodes(parsedCount: number): void {
    for (let codeIndex = 0; codeIndex < parsedCount; codeIndex++) {
      const ansiCode = this.#parsedCodes[codeIndex];
      switch (
        true // using switch(true) for homogeneous structure & readability
      ) {
        case ansiCode === SGR_RESET: {
          this.#resetStyles();
          break;
        }
        case ansiCode === SGR_BOLD: {
          this.#currentFontWeight = FONT_WEIGHT_BOLD;
          this.#hasActiveStyle = true;
          break;
        }
        case ansiCode === SGR_BOLD_OFF: {
          this.#currentFontWeight = undefined;
          this.#hasActiveStyle = Boolean(
            this.#currentColor ||
              this.#currentBackground ||
              this.#currentFontWeight
          );
          break;
        }
        case ansiCode === SGR_FG_EXTENDED &&
          this.#parsedCodes[codeIndex + 1] === 5: {
          const paletteIndex = this.#parsedCodes[codeIndex + 2];
          if (paletteIndex != null) {
            const mapped = ANSI_256_MAP[paletteIndex];
            if (mapped) {
              this.#currentColor = mapped;
              this.#hasActiveStyle = true;
            }
          }
          codeIndex += 2; // skip '5;<idx>'
          break;
        }
        case ansiCode === SGR_BG_EXTENDED &&
          this.#parsedCodes[codeIndex + 1] === 5: {
          const paletteIndex = this.#parsedCodes[codeIndex + 2];
          if (paletteIndex != null) {
            const mapped = ANSI_256_MAP[paletteIndex];
            if (mapped) {
              this.#currentBackground = mapped;
              this.#hasActiveStyle = true;
            }
          }
          codeIndex += 2;
          break;
        }
        case ansiCode >= 30 && ansiCode <= 37: {
          this.#currentColor = BASIC_FG_COLORS[ansiCode - 30];
          this.#hasActiveStyle = true;
          break;
        }
        case ansiCode >= 90 && ansiCode <= 97: {
          this.#currentColor = BRIGHT_FG_COLORS[ansiCode - 90];
          this.#hasActiveStyle = true;
          break;
        }
        case ansiCode === SGR_FG_DEFAULT: {
          this.#currentColor = undefined;
          this.#hasActiveStyle = Boolean(
            this.#currentBackground || this.#currentFontWeight
          );
          break;
        }
        case ansiCode === SGR_BG_DEFAULT: {
          this.#currentBackground = undefined;
          this.#hasActiveStyle = Boolean(
            this.#currentColor || this.#currentFontWeight
          );
          break;
        }
        default: {
          // Unsupported / intentionally ignored SGR code.
        }
      }
    }
  }

  /** Reset style-related state to defaults (SGR 0 or empty parameter list). */
  #resetStyles(): void {
    this.#currentColor = this.#currentBackground = this.#currentFontWeight = undefined;
    this.#hasActiveStyle = false;
    this.#currentStyleSpanStart = '';
  }

  /** Rebuild the opening span tag (if any style active) using deterministic property ordering. */
  #rebuildStyleSpanStart(): void {
    if (!this.#hasActiveStyle) {
      this.#currentStyleSpanStart = '';
      return;
    }
    // Create a stable signature for cache key (null placeholders keep positional clarity).
    const signatureColor = this.#currentColor ?? '';
    const signatureBg = this.#currentBackground ?? '';
    const signatureWeight = this.#currentFontWeight ?? '';
    const signature = `${signatureColor}|${signatureBg}|${signatureWeight}`;
    const cached = AnsiHtmlConverter.#StyleCache.get(signature);
    if (cached) {
      this.#currentStyleSpanStart = cached;
      return;
    }
    const styleFragments: string[] = [];
    if (signatureColor) styleFragments.push(`color: ${signatureColor}`);
    if (signatureBg) styleFragments.push(`background: ${signatureBg}`);
    if (signatureWeight) styleFragments.push(`font-weight: ${signatureWeight}`);
    const built = styleFragments.length
      ? `<span style="${styleFragments.join(';')}">`
      : '';
    AnsiHtmlConverter.#StyleCache.set(signature, built);
    this.#currentStyleSpanStart = built;
  }

  /** Wrap a text segment with current style if active. */
  #wrapIfStyled(text: string): string {
    return this.#currentStyleSpanStart
      ? `${this.#currentStyleSpanStart}${text}</span>`
      : text;
  }

  /** Convert ANSI-coded text to HTML (newlines already streamed into `<br/>`). */
  static formatWithNewlines(input: string): string {
    return this.convert(input);
  }
}

/**
 * Create a browser logger function that appends formatted, ANSI->HTML
 * converted text to a <pre> element in `container` (or the default host).
 *
 * The returned logger intentionally mutates the provided `args` array to
 * avoid creating a temporary copy when an options object is passed as the
 * last argument (common pattern in the demo). This keeps short-lived
 * allocations low during intensive logging.
 *
 * @param container - Optional host element for log output
 * @returns logger function compatible with the demo's forceLog API
 */
export function createBrowserLogger(
  container?: HTMLElement
): (...args: any[]) => void {
  /**
   * Create a browser logger function that appends formatted, ANSI->HTML
   * converted text to a <pre> element in `container` (or the default host).
   *
   * The returned logger intentionally mutates the provided `args` array to
   * avoid creating a temporary copy when an options object is passed as the
   * last argument (common pattern in the demo). This keeps short-lived
   * allocations low during intensive logging.
   *
   * @example
   * const logger = createBrowserLogger(document.getElementById('out'));
   * logger('\x1b[38;5;205mHello\x1b[0m', { prepend: true });
   *
   * @remarks Not reentrant if the DOM node is externally replaced while a log
   * operation is in progress (single-threaded assumption). Designed for high
   * throughput incremental logging with minimal allocations.
   */
  return (...args: any[]) => {
    // Resolve (or recreate) the <pre> element each time because the clearer
    // may remove it (clearFunction sets container.innerHTML = ''), leaving
    // a stale reference otherwise.
    const logPreElement = ensurePre(container);

    // Detect an optional options object in the last argument. Consumers can
    // pass `{ prepend: true }` to indicate the text should be added at the
    // top of the log (useful for archive views where newest entries appear
    // above older ones).
    let logOptions: any = undefined;
    if (args.length) {
      const lastArgument = MazeUtils.safeLast(args as any);
      if (
        lastArgument &&
        typeof lastArgument === 'object' &&
        'prepend' in (lastArgument as any)
      ) {
        logOptions = lastArgument as any;
        // Remove the last arg in-place to avoid allocating a new args array.
        // This is a deliberate micro-optimization for hot logging paths.
        args.pop();
      }
    }

    // Build the combined text without allocating an intermediate mapped array.
    let combinedText = '';
    for (let argumentIndex = 0; argumentIndex < args.length; argumentIndex++) {
      if (argumentIndex) combinedText += ' ';
      const argumentValue = args[argumentIndex];
      combinedText +=
        typeof argumentValue === 'string'
          ? argumentValue
          : JSON.stringify(argumentValue);
    }
    // Convert ANSI -> HTML and preserve explicit newlines as <br/> so the
    // boxed ASCII layout remains intact inside the pre element.
    if (!logPreElement) return;

    const html = AnsiHtmlConverter.formatWithNewlines(combinedText) + '<br/>';

    if (logOptions && logOptions.prepend) {
      // Use insertAdjacentHTML to avoid reparsing entire existing content.
      logPreElement.insertAdjacentHTML('afterbegin', html);
      logPreElement.scrollTop = 0; // newest visible
    } else {
      logPreElement.insertAdjacentHTML('beforeend', html);
      logPreElement.scrollTop = logPreElement.scrollHeight;
    }
  };
}
