import { double, Calculator, DataRecord, ID } from './simple';

/**
 * A class that implements an interface and imports from simple.
 */
export class DataManager implements DataRecord {
  field1: string = '';
  field2: number = 0;
  field3: boolean = false;
  field4: string = '';
  field5: number = 0;
  field6: boolean = false;
  field7: string = '';
  field8: number = 0;
  field9: boolean = false;
  field10: string = '';
  field11: number = 0;
  field12: boolean = false;
  field13: string = '';
  field14: number = 0;
  field15: boolean = false;
  field16: string = '';
  field17: number = 0;
  field18: boolean = false;
  field19: string = '';
  field20: number = 0;
  field21: boolean = false;
  field22: string = '';
  field23: number = 0;
  field24: boolean = false;
  field25: string = '';
  field26: number = 0;
  field27: boolean = false;
  field28: string = '';
  field29: number = 0;
  field30: boolean = false;

  /** Process data using imported function. */
  processData(input: number): number {
    const result = double(input);
    const calc = new Calculator();
    return calc.addLarge(result);
  }

  /** Get the ID. */
  getId(): ID {
    return 'dm-001';
  }
}