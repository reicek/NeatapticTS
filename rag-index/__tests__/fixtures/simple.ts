/**
 * A simple exported function with JSDoc.
 * @param x - Input number.
 * @returns Doubled value.
 */
export function double(x: number): number {
  return x * 2;
}

/** No-jsdoc function. */
export function noop(): void {
  // does nothing
}

/**
 * A class with large and small methods.
 * @description This class demonstrates method sub-chunking.
 */
export class Calculator {
  private value: number = 0;

  /** Small getter. */
  get current(): number {
    return this.value;
  }

  /** Small method. */
  reset(): void {
    this.value = 0;
  }

  /**
   * Large method that exceeds the small method threshold.
   * @param n - Number to add.
   * @returns The new value after adding n to the current value and performing some complex logic.
   */
  addLarge(n: number): number {
    this.value = this.value + n;
    // Add a lot of body text to make this method large enough
    const temp1 = n * 2;
    const temp2 = temp1 + this.value;
    const temp3 = temp2 * 3;
    const temp4 = temp3 + 100;
    const temp5 = temp4 * 5;
    const temp6 = temp5 + 200;
    const temp7 = temp6 * 7;
    const temp8 = temp7 + 300;
    const temp9 = temp8 * 9;
    const temp10 = temp9 + 400;
    const temp11 = temp10 * 11;
    const temp12 = temp11 + 500;
    const temp13 = temp12 * 13;
    const temp14 = temp13 + 600;
    const temp15 = temp14 * 15;
    const temp16 = temp15 + 700;
    const temp17 = temp16 * 17;
    const temp18 = temp17 + 800;
    const temp19 = temp18 * 19;
    const temp20 = temp19 + 900;
    return this.value + temp20;
  }

  /**
   * Another large method for testing multiple large method sub-chunks.
   * @param n - Number to multiply.
   * @returns The new value after multiplying.
   */
  multiplyLarge(n: number): number {
    this.value = this.value * n;
    const step1 = n + 10;
    const step2 = step1 * 20;
    const step3 = step2 + 30;
    const step4 = step3 * 40;
    const step5 = step4 + 50;
    const step6 = step5 * 60;
    const step7 = step6 + 70;
    const step8 = step7 * 80;
    const step9 = step8 + 90;
    const step10 = step9 * 100;
    const step11 = step10 + 110;
    const step12 = step11 * 120;
    const step13 = step12 + 130;
    const step14 = step13 * 140;
    const step15 = step14 + 150;
    const step16 = step15 * 160;
    const step17 = step16 + 170;
    const step18 = step17 * 180;
    const step19 = step18 + 190;
    const step20 = step19 * 200;
    return this.value + step20;
  }
}

/**
 * Interface with many properties for sub-chunk testing.
 */
export interface DataRecord {
  /** First field. */
  field1: string;
  /** Second field. */
  field2: number;
  /** Third field. */
  field3: boolean;
  /** Fourth field. */
  field4: string;
  /** Fifth field. */
  field5: number;
  /** Sixth field. */
  field6: boolean;
  /** Seventh field. */
  field7: string;
  /** Eighth field. */
  field8: number;
  /** Ninth field. */
  field9: boolean;
  /** Tenth field. */
  field10: string;
  /** Eleventh field. */
  field11: number;
  /** Twelfth field. */
  field12: boolean;
  /** Thirteenth field. */
  field13: string;
  /** Fourteenth field. */
  field14: number;
  /** Fifteenth field. */
  field15: boolean;
  /** Sixteenth field. */
  field16: string;
  /** Seventeenth field. */
  field17: number;
  /** Eighteenth field. */
  field18: boolean;
  /** Nineteenth field. */
  field19: string;
  /** Twentieth field. */
  field20: number;
  /** Twenty-first field with a longer JSDoc to increase property body length. */
  field21: boolean;
  /** Twenty-second field with another longer JSDoc description for testing. */
  field22: string;
  /** Twenty-third field. */
  field23: number;
  /** Twenty-fourth field. */
  field24: boolean;
  /** Twenty-fifth field. */
  field25: string;
  /** Twenty-sixth field. */
  field26: number;
  /** Twenty-seventh field. */
  field27: boolean;
  /** Twenty-eighth field. */
  field28: string;
  /** Twenty-ninth field. */
  field29: number;
  /** Thirtieth field. */
  field30: boolean;
}

/** A type alias. */
export type ID = string | number;

/** An arrow function variable. */
export const greet = (name: string): string => `Hello, ${name}`;

/** A non-function variable. */
export const MAX_VALUE: number = 100;

/** A class extending Error. */
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

/** Small interface with no properties. */
export interface Empty {}

/** Default export function. */
export default function defaultFn(): void {
  console.log('default');
}