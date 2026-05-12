import { resolveBrowserWorkerAssetUrl } from './network.worker-payload';

type BrowserDocumentLike = Document & {
  currentScript?: { src?: string | null } | null;
};

describe('resolveBrowserWorkerAssetUrl', () => {
  afterEach(() => {
    Reflect.deleteProperty(globalThis, 'document');
    Reflect.deleteProperty(globalThis, 'location');
  });

  it('prefers an explicit baseUrl over browser globals', () => {
    expect(
      resolveBrowserWorkerAssetUrl('worker.js', {
        baseUrl: 'https://example.test/assets/host.js',
      }),
    ).toBe('https://example.test/assets/worker.js');
  });

  it('falls back to document.currentScript when no explicit baseUrl exists', () => {
    const browserDocument = {} as BrowserDocumentLike;

    Object.defineProperty(browserDocument, 'currentScript', {
      configurable: true,
      value: {
        src: 'https://example.test/docs/runtime.bundle.js',
      },
    });
    Reflect.set(globalThis, 'document', browserDocument);

    expect(resolveBrowserWorkerAssetUrl('worker.js')).toBe(
      'https://example.test/docs/worker.js',
    );
  });

  it('falls back to location.href when no current script is available', () => {
    Reflect.set(globalThis, 'document', {} as BrowserDocumentLike);
    Reflect.set(globalThis, 'location', {
      href: 'https://example.test/docs/index.html',
    });

    expect(resolveBrowserWorkerAssetUrl('worker.js')).toBe(
      'https://example.test/docs/worker.js',
    );
  });

  it('returns undefined when no browser base URL is available', () => {
    Reflect.set(globalThis, 'document', {} as BrowserDocumentLike);

    expect(resolveBrowserWorkerAssetUrl('worker.js')).toBeUndefined();
  });
});
