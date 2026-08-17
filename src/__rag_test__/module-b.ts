import { helperFunc, HelperInterface, HelperType } from './module-a';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { externalThing } from 'external-package';

/**
 * Consumer class implementing HelperInterface.
 * @description Uses helperFunc internally.
 */
export class Consumer implements HelperInterface {
  value: string = '';
  data: HelperType = '';

  /**
   * Use helper function with additional computation.
   * This method body is intentionally long enough to exceed the minimum
   * method entity character threshold so that a separate function entity
   * (with parent_class metadata) is produced for it during code entity
   * extraction.
   */
  useHelper(): number {
    const input = 42;
    const intermediate = helperFunc(input);
    const squared = intermediate * intermediate;
    const adjusted = squared > 0 ? squared - 1 : 0;
    return adjusted;
  }

  /** Get value. */
  getValue(): string {
    return this.value;
  }
}

/** Standalone function calling helper. */
export function standaloneFunc(): number {
  return helperFunc(0);
}
