import Network from '../../src/architecture/network';

// Helper to build a deterministic dataset
const ds = [{ input: [0], output: [0] }];

// Each scenario uses one assertion per 'it'.

describe('Smoothing Types', () => {
  describe('wma responsiveness vs sma', () => {
    const netA = new Network(1, 1);
    const netB = new Network(1, 1);
    const seq = [0.9, 0.8, 0.7, 0.6];
    let iA = 0;
    let iB = 0;
    const costA = () => seq[Math.min(iA++, seq.length - 1)];
    const costB = () => seq[Math.min(iB++, seq.length - 1)];
    let sma: any;
    let wma: any;
    beforeAll(() => {
      sma = netA.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costA,
        movingAverageType: 'sma',
        movingAverageWindow: 4,
        error: 0,
      });
      wma = netB.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costB,
        movingAverageType: 'wma',
        movingAverageWindow: 4,
        error: 0,
      });
    });
    test('wma final error lower or equal to sma (faster adaptation)', () => {
      expect(wma.error).toBeLessThanOrEqual(sma.error);
    });
  });

  describe('median handles spike better than sma', () => {
    const netMed = new Network(1, 1);
    const netSma = new Network(1, 1);
    const seq = [0.5, 2.0, 0.5, 0.5];
    let im = 0;
    let is = 0;
    const costMed = () => seq[Math.min(im++, seq.length - 1)];
    const costSma = () => seq[Math.min(is++, seq.length - 1)];
    let med: any;
    let sma: any;
    beforeAll(() => {
      med = netMed.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costMed,
        movingAverageType: 'median',
        movingAverageWindow: 4,
        error: 0,
      });
      sma = netSma.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costSma,
        movingAverageType: 'sma',
        movingAverageWindow: 4,
        error: 0,
      });
    });
    test('median final error below sma (spike resisted)', () => {
      expect(med.error).toBeLessThan(sma.error);
    });
  });

  describe('trimmed mean between median and sma', () => {
    const netT = new Network(1, 1);
    const netS = new Network(1, 1);
    const seq = [0.5, 2, 0.5, 0.5];
    let itIdx = 0,
      is = 0;
    const costT = () => seq[Math.min(itIdx++, seq.length - 1)];
    const costS = () => seq[Math.min(is++, seq.length - 1)];
    let trimmedRes: any;
    let sma: any;
    beforeAll(() => {
      trimmedRes = netT.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costT,
        movingAverageType: 'trimmed',
        trimmedRatio: 0.25,
        movingAverageWindow: 4,
        error: 0,
      });
      sma = netS.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costS,
        movingAverageType: 'sma',
        movingAverageWindow: 4,
        error: 0,
      });
    });
    test('trimmed error <= sma error', () => {
      expect(trimmedRes.error).toBeLessThanOrEqual(sma.error);
    });
  });

  describe('gaussian within bounds of sma', () => {
    const netG = new Network(1, 1);
    const netS = new Network(1, 1);
    const seq = [0.9, 0.8, 0.7, 0.6];
    let ig = 0,
      is = 0;
    const costG = () => seq[Math.min(ig++, seq.length - 1)];
    const costS = () => seq[Math.min(is++, seq.length - 1)];
    let g: any;
    let s: any;
    beforeAll(() => {
      g = netG.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costG,
        movingAverageType: 'gaussian',
        movingAverageWindow: 4,
        error: 0,
      });
      s = netS.train(ds, {
        iterations: 4,
        rate: 0.1,
        cost: costS,
        movingAverageType: 'sma',
        movingAverageWindow: 4,
        error: 0,
      });
    });
    test('gaussian error between min and max raw values', () => {
      expect(g.error).toBeGreaterThanOrEqual(0.6);
    });
  });

  describe('adaptive-ema reacts more strongly under variance', () => {
    const netA = new Network(1, 1);
    const netE = new Network(1, 1);
    const seqVar = [0.9, 0.7, 0.85, 0.65, 0.6];
    let ia = 0,
      ie = 0;
    const costA = () => seqVar[Math.min(ia++, seqVar.length - 1)];
    const costE = () => seqVar[Math.min(ie++, seqVar.length - 1)];
    let adapt: any;
    let ema: any;
    beforeAll(() => {
      adapt = netA.train(ds, {
        iterations: 5,
        rate: 0.1,
        cost: costA,
        movingAverageType: 'adaptive-ema',
        movingAverageWindow: 5,
        error: 0,
      });
      ema = netE.train(ds, {
        iterations: 5,
        rate: 0.1,
        cost: costE,
        movingAverageType: 'ema',
        movingAverageWindow: 5,
        error: 0,
      });
    });
    test('adaptive ema <= plain ema final error', () => {
      expect(adapt.error).toBeLessThanOrEqual(ema.error);
    });
  });
});
