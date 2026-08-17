/**
 * Helper function.
 * @param x - Input number.
 * @returns Incremented value.
 */
export function helperFunc(x: number): number {
  return x + 1;
}

/**
 * Helper interface.
 */
export interface HelperInterface {
  /** Value field. */
  value: string;
}

/** Type alias. */
export type HelperType = string | number;

/** Arrow function variable. */
export const helperArrow = (x: string): string => x.toUpperCase();

/** Non-function variable. */
export const helperConst: number = 42;

/** Error class. */
export class HelperError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'HelperError';
  }
}

/** Default export. */
export default function helperDefault(): void {
  // noop
}
