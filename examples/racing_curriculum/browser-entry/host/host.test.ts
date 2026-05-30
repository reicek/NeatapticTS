import { createRacingHost, RACING_NARROW_VIEWPORT_THRESHOLD_PX } from './host';

describe('racing host boundary', () => {
  describe('createRacingHost', () => {
    it('switches the shell to the narrow layout below the viewport threshold', () => {
      const previousDocument = Reflect.get(globalThis, 'document');
      const previousWindow = Reflect.get(globalThis, 'window');
      const fakeDocument = createMockDocument();
      const containerElement = createMockElement('div') as HTMLDivElement &
        MockElement;

      Reflect.set(globalThis, 'document', fakeDocument);
      Reflect.set(globalThis, 'window', {
        innerWidth: RACING_NARROW_VIEWPORT_THRESHOLD_PX + 10,
      });

      try {
        const racingHost = createRacingHost(containerElement);

        racingHost.applyViewportLayout(RACING_NARROW_VIEWPORT_THRESHOLD_PX - 1);

        expect(racingHost.rootElement.dataset.racingLayout).toBe('narrow');
      } finally {
        if (previousDocument === undefined) {
          Reflect.deleteProperty(globalThis, 'document');
        } else {
          Reflect.set(globalThis, 'document', previousDocument);
        }

        if (previousWindow === undefined) {
          Reflect.deleteProperty(globalThis, 'window');
        } else {
          Reflect.set(globalThis, 'window', previousWindow);
        }
      }
    });
  });
});

function createMockDocument(): {
  createElement: (tagName: string) => MockElement;
} {
  return {
    createElement(tagName: string): MockElement {
      return createMockElement(tagName);
    },
  };
}

function createMockElement(tagName: string): MockElement {
  return {
    attributes: new Map<string, string>(),
    children: [],
    classList: {
      toggle: () => undefined,
    },
    className: '',
    dataset: {},
    height: tagName === 'canvas' ? 0 : undefined,
    setAttribute(name: string, value: string): void {
      this.attributes.set(name, value);
    },
    append(...childNodes: MockElement[]): void {
      this.children.push(...childNodes);
    },
    replaceChildren(...childNodes: MockElement[]): void {
      this.children = [...childNodes];
    },
    textContent: '',
    width: tagName === 'canvas' ? 0 : undefined,
  };
}

type MockElement = {
  attributes: Map<string, string>;
  children: MockElement[];
  classList: {
    toggle: (token: string, force?: boolean) => void;
  };
  className: string;
  dataset: Record<string, string>;
  height?: number;
  setAttribute: (name: string, value: string) => void;
  append: (...childNodes: MockElement[]) => void;
  replaceChildren: (...childNodes: MockElement[]) => void;
  textContent: string;
  width?: number;
};
