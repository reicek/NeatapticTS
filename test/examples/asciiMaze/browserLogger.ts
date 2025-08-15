/**
 * browserLogger.ts
 *
 * Provides a `createBrowserLogger` factory that returns a function compatible with `forceLog`.
 * The logger converts common ANSI sequences into inline HTML styles and appends
 * formatted output to a <pre> element inside the provided container.
 */

/** Minimal mapping of 256-color palette indices used by the demo to CSS hex colors. */
const ANSI_256_MAP: { [code: number]: string } = {
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
};

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function ensurePre(container?: HTMLElement) {
  const host =
    container ??
    (typeof document !== 'undefined'
      ? document.getElementById('ascii-maze-output')
      : null);
  if (!host) return null;
  let pre = host.querySelector('pre');
  if (!pre) {
    pre = document.createElement('pre');
    pre.style.fontFamily = 'monospace';
    pre.style.whiteSpace = 'pre';
    pre.style.margin = '0';
    pre.style.padding = '4px';
    pre.style.fontSize = '10px';
    host.appendChild(pre);
  }
  return pre as HTMLPreElement;
}

/**
 * Convert a string that may contain SGR ANSI sequences (\x1b[...m) into HTML.
 * Supports: 0 (reset), 1 (bold), 38;5;<n> (fg), 48;5;<n> (bg), and simple color codes.
 */
function ansiToHtml(input: string) {
  const re = /\x1b\[([0-9;]*)m/g;
  let out = '';
  let lastIndex = 0;
  let style: { color?: string; background?: string; fontWeight?: string } = {};

  let match: RegExpExecArray | null;
  while ((match = re.exec(input)) !== null) {
    const chunk = input.substring(lastIndex, match.index);
    if (chunk) {
      const text = escapeHtml(chunk);
      if (Object.keys(style).length) {
        const css: string[] = [];
        if (style.color) css.push(`color: ${style.color}`);
        if (style.background) css.push(`background: ${style.background}`);
        if (style.fontWeight) css.push(`font-weight: ${style.fontWeight}`);
        out += `<span style="${css.join(';')}">${text}</span>`;
      } else {
        out += text;
      }
    }

    const codes = match[1]
      .split(';')
      .filter((c) => c.length)
      .map((c) => parseInt(c, 10));
    if (codes.length === 0) {
      // CSI m with no codes = reset
      style = {};
    } else {
      // Process codes sequentially
      for (let i = 0; i < codes.length; i++) {
        const c = codes[i];
        if (c === 0) {
          style = {};
        } else if (c === 1) {
          style.fontWeight = '700';
        } else if (c === 22) {
          delete style.fontWeight;
        } else if (c === 38 && codes[i + 1] === 5) {
          const n = codes[i + 2];
          if (typeof n === 'number' && ANSI_256_MAP[n])
            style.color = ANSI_256_MAP[n];
          i += 2;
        } else if (c === 48 && codes[i + 1] === 5) {
          const n = codes[i + 2];
          if (typeof n === 'number' && ANSI_256_MAP[n])
            style.background = ANSI_256_MAP[n];
          i += 2;
        } else if (c >= 30 && c <= 37) {
          // basic colors, map a few
          const basic = [
            '#000000',
            '#800000',
            '#008000',
            '#808000',
            '#000080',
            '#800080',
            '#008080',
            '#c0c0c0',
          ];
          style.color = basic[c - 30];
        } else if (c >= 90 && c <= 97) {
          const bright = [
            '#808080',
            '#ff0000',
            '#00ff00',
            '#ffff00',
            '#0000ff',
            '#ff00ff',
            '#00ffff',
            '#ffffff',
          ];
          style.color = bright[c - 90];
        } else if (c === 39) {
          delete style.color;
        } else if (c === 49) {
          delete style.background;
        }
      }
    }

    lastIndex = re.lastIndex;
  }

  // Trailing text
  if (lastIndex < input.length) {
    const tail = escapeHtml(input.substring(lastIndex));
    if (Object.keys(style).length) {
      const css: string[] = [];
      if (style.color) css.push(`color: ${style.color}`);
      if (style.background) css.push(`background: ${style.background}`);
      if (style.fontWeight) css.push(`font-weight: ${style.fontWeight}`);
      out += `<span style="${css.join(';')}">${tail}</span>`;
    } else {
      out += tail;
    }
  }

  return out;
}

export function createBrowserLogger(container?: HTMLElement) {
  return (...args: any[]) => {
    // Resolve (or recreate) the <pre> element each time because the clearer
    // may remove it (clearFunction sets container.innerHTML = ''), leaving
    // a stale reference otherwise.
    const pre = ensurePre(container);

    // Detect an optional options object in the last argument. Consumers can
    // pass `{ prepend: true }` to indicate the text should be added at the
    // top of the log (useful for archive views where newest entries appear
    // above older ones).
    let opts: any = undefined;
    if (
      args.length &&
      typeof args[args.length - 1] === 'object' &&
      args[args.length - 1] &&
      'prepend' in args[args.length - 1]
    ) {
      opts = args[args.length - 1];
      args = args.slice(0, -1);
    }

    const text = args
      .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
      .join(' ');
    // Convert ANSI -> HTML and preserve explicit newlines as <br/> so the
    // boxed ASCII layout remains intact inside the pre element.
    const html = ansiToHtml(text).replace(/\n/g, '<br/>') + '<br/>';
    if (!pre) return;

    if (opts && opts.prepend) {
      // Put new content above existing content so newest entries appear first.
      pre.innerHTML = html + pre.innerHTML;
      // Scroll to top so the newly prepended item is visible
      pre.scrollTop = 0;
    } else {
      // Default behavior: append at the bottom
      pre.innerHTML += html;
      // Keep pre scrolled to bottom
      pre.scrollTop = pre.scrollHeight;
    }
  };
}
