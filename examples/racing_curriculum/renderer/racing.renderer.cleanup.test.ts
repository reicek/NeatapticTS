import fs from 'fs';
import path from 'path';

const sourcePath = path.join(__dirname, 'racing.renderer.ts');
const sourceText = fs.readFileSync(sourcePath, 'utf8');

describe('renderer cleanup — no yellow optimal-line overlay', () => {
  it('does not define a drawOptimalLineGuidance function', () => {
    expect(sourceText.includes('function drawOptimalLineGuidance')).toBe(false);
  });

  it('does not call drawOptimalLineGuidance from the drawTrack pipeline', () => {
    expect(sourceText.includes('drawOptimalLineGuidance(')).toBe(false);
  });

  it('does not contain the yellow optimal-line RGB constant COLOR_GUIDANCE_LINE_RGB', () => {
    expect(sourceText.includes('COLOR_GUIDANCE_LINE_RGB')).toBe(false);
  });

  it('does not contain the yellow color value 255,209,102 used by the optimal-line overlay', () => {
    expect(sourceText.includes('255,209,102')).toBe(false);
  });

  it('does not reference optimalLinePoints in the track render geometry', () => {
    expect(sourceText.includes('optimalLinePoints')).toBe(false);
  });
});

describe('renderer cleanup — no cyan center divider', () => {
  it('does not define a drawTrackCenterline function', () => {
    expect(sourceText.includes('function drawTrackCenterline')).toBe(false);
  });

  it('does not call drawTrackCenterline from the drawTrack pipeline', () => {
    expect(sourceText.includes('drawTrackCenterline(')).toBe(false);
  });

  it('does not contain the cyan centerline constant COLOR_CENTERLINE', () => {
    expect(sourceText.includes('COLOR_CENTERLINE')).toBe(false);
  });

  it('does not contain the cyan centerline color value rgba(0,180,220,0.30)', () => {
    expect(sourceText.includes('rgba(0,180,220,0.30)')).toBe(false);
  });
});

describe('renderer cleanup — blue team guide lines retained', () => {
  it('still defines the COLOR_GUIDING_LINE_TEAM_A_RGB blue constant', () => {
    expect(sourceText.includes('COLOR_GUIDING_LINE_TEAM_A_RGB')).toBe(true);
  });

  it('still defines the drawTeamGuidingLines function', () => {
    expect(sourceText.includes('function drawTeamGuidingLines')).toBe(true);
  });
});

describe('renderer cleanup — red team guide lines retained', () => {
  it('still defines the COLOR_GUIDING_LINE_TEAM_B_RGB red constant', () => {
    expect(sourceText.includes('COLOR_GUIDING_LINE_TEAM_B_RGB')).toBe(true);
  });

  it('still exports the buildGuidingLineForTeam helper', () => {
    expect(sourceText.includes('buildGuidingLineForTeam')).toBe(true);
  });
});

describe('renderer cleanup — no orphaned constants or data for removed overlays', () => {
  it('does not retain the COLOR_GUIDANCE_LINE_RGB constant used only by the optimal-line', () => {
    expect(sourceText.includes('COLOR_GUIDANCE_LINE_RGB')).toBe(false);
  });

  it('does not retain the COLOR_CENTERLINE constant used only by the center divider', () => {
    expect(sourceText.includes('COLOR_CENTERLINE')).toBe(false);
  });

  it('does not retain the optimalLinePoints geometry field used only by the optimal-line', () => {
    expect(sourceText.includes('optimalLinePoints')).toBe(false);
  });
});