import assert from 'node:assert/strict';
import {
  POSTPROCESS_MODES,
  postprocessMotionForDisplay,
  resolvePostprocessMode,
} from '../../site/infinite_demo/postprocess.js';

const historyLength = 4;
const generationFrames = 40;
const motionDim = 21;
const rootDim = 5;
const rotationStart = 5;
const rotationEnd = 17;
const rootMean = new Float32Array(rootDim);
const rootStd = new Float32Array(rootDim).fill(1);
const fullMotion = new Float32Array((historyLength + generationFrames) * motionDim);
for (let frame = 0; frame < historyLength + generationFrames; frame += 1) {
  const base = frame * motionDim;
  const alternating = frame % 2 ? 0.09 : -0.09;
  fullMotion[base] = frame * 0.05 + alternating;
  fullMotion[base + 1] = 1;
  fullMotion[base + 2] = frame * 0.02 - alternating;
  fullMotion[base + 3] = Math.cos(frame * 0.02);
  fullMotion[base + 4] = Math.sin(frame * 0.02);
  for (let feature = rotationStart; feature < rotationEnd; feature += 1) {
    fullMotion[base + feature] = Math.sin(frame * 0.12 + feature) + alternating;
  }
  for (let feature = motionDim - 4; feature < motionDim; feature += 1) {
    fullMotion[base + feature] = (frame + feature) % 2;
  }
}
const original = fullMotion.slice();
const historyFrames = Array.from({ length: historyLength }, (_, frame) => (
  fullMotion.subarray(frame * motionDim, (frame + 1) * motionDim)
));

function run(mode) {
  return postprocessMotionForDisplay({
    fullMotion,
    historyFrames,
    historyLength,
    generationFrames,
    motionDim,
    rootDim,
    rotationStart,
    rotationEnd,
    excludedTailFeatures: 4,
    rootMean,
    rootStd,
    mode,
    protectedRootFrames: [2, 20],
  });
}

function highFrequencyEnergy(values, feature) {
  let total = 0;
  for (let frame = 2; frame < generationFrames; frame += 1) {
    const current = values[frame * motionDim + feature];
    const previous = values[(frame - 1) * motionDim + feature];
    const before = values[(frame - 2) * motionDim + feature];
    const secondDifference = current - 2 * previous + before;
    total += secondDifference * secondDifference;
  }
  return total;
}

assert.equal(resolvePostprocessMode('unknown'), POSTPROCESS_MODES.raw);
const raw = run('raw');
assert.deepEqual(raw.motion, fullMotion.slice(historyLength * motionDim));
assert.equal(raw.diagnostics.root_max_deviation_m, 0);

const balanced = run('balanced');
const strong = run('strong');
assert.ok(balanced.allFinite && strong.allFinite);
assert.ok(highFrequencyEnergy(balanced.motion, 0) < highFrequencyEnergy(raw.motion, 0));
assert.ok(highFrequencyEnergy(balanced.motion, rotationStart) < highFrequencyEnergy(raw.motion, rotationStart));
assert.ok(highFrequencyEnergy(strong.motion, 0) < highFrequencyEnergy(balanced.motion, 0));
assert.ok(highFrequencyEnergy(strong.motion, rotationStart) < highFrequencyEnergy(balanced.motion, rotationStart));
assert.ok(balanced.diagnostics.root_endpoint_deviation_m < 1e-7);
assert.ok(strong.diagnostics.root_endpoint_deviation_m < 1e-7);
for (const mode of ['seam', 'root', 'pose', 'balanced', 'strong']) {
  const result = run(mode);
  for (const protectedFrame of [2, 20]) {
    const protectedOutput = protectedFrame * motionDim;
    const protectedInput = (historyLength + protectedFrame) * motionDim;
    assert.equal(result.motion[protectedOutput], fullMotion[protectedInput]);
    assert.equal(result.motion[protectedOutput + 2], fullMotion[protectedInput + 2]);
  }
  for (let frame = 0; frame < generationFrames; frame += 1) {
    const base = frame * motionDim;
    const cosine = result.motion[base + 3] * rootStd[3] + rootMean[3];
    const sine = result.motion[base + 4] * rootStd[4] + rootMean[4];
    assert.ok(Math.abs(Math.hypot(cosine, sine) - 1) < 1e-6, `${mode} heading ${frame}`);
  }
}
for (let frame = 0; frame < generationFrames; frame += 1) {
  for (let feature = motionDim - 4; feature < motionDim; feature += 1) {
    const output = frame * motionDim + feature;
    const input = (historyLength + frame) * motionDim + feature;
    assert.equal(balanced.motion[output], fullMotion[input]);
  }
}
assert.deepEqual(fullMotion, original, 'postprocessing must not mutate model output');

const begin = performance.now();
for (let index = 0; index < 1000; index += 1) run('balanced');
const meanMs = (performance.now() - begin) / 1000;
console.log(`POSTPROCESS_TEST_PASS balanced_mean_ms=${meanMs.toFixed(4)}`);
