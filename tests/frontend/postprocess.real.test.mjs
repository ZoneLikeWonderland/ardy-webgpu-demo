import assert from 'node:assert/strict';
import fs from 'node:fs';

import { postprocessMotionForDisplay } from '../../site/infinite_demo/postprocess.js';

const manifest = JSON.parse(fs.readFileSync(
  new URL('../../site/infinite_demo/manifest.json', import.meta.url),
  'utf8',
));
const golden = JSON.parse(fs.readFileSync(
  new URL('../../site/infinite_demo/browser_validation_case.json', import.meta.url),
  'utf8',
));
const specification = golden.continuation;
const historyLength = manifest.history_frames;
const generationFrames = manifest.generation_frames;
const motionDim = manifest.motion_dim;
const jointCount = manifest.skeleton.parents.length;
const rotationStart = manifest.root_dim + (jointCount - 1) * 3;
const rotationEnd = rotationStart + jointCount * 6;
const history = Float32Array.from(specification.history);
const raw = Float32Array.from(specification.expected.motion);
const fullMotion = new Float32Array((historyLength + generationFrames) * motionDim);
fullMotion.set(history);
fullMotion.set(raw, history.length);
const original = fullMotion.slice();
const historyFrames = Array.from({ length: historyLength }, (_, frame) => (
  history.subarray(frame * motionDim, (frame + 1) * motionDim)
));

function run(mode) {
  return postprocessMotionForDisplay({
    fullMotion,
    historyFrames,
    historyLength,
    generationFrames,
    motionDim,
    rootDim: manifest.root_dim,
    rotationStart,
    rotationEnd,
    excludedTailFeatures: manifest.inertialization.excluded_tail_features,
    rootMean: manifest.root_stats.mean,
    rootStd: manifest.root_stats.std_eps,
    mode,
    protectedRootFrames: [2],
  });
}

function secondDifferenceRms(values, featureStart, featureEnd) {
  let squared = 0;
  let count = 0;
  for (let frame = 2; frame < generationFrames; frame += 1) {
    for (let feature = featureStart; feature < featureEnd; feature += 1) {
      const delta = (
        values[frame * motionDim + feature]
        - 2 * values[(frame - 1) * motionDim + feature]
        + values[(frame - 2) * motionDim + feature]
      );
      squared += delta * delta;
      count += 1;
    }
  }
  return Math.sqrt(squared / count);
}

function seamVelocityMismatch(values, featureStart, featureEnd) {
  const previousBase = (historyLength - 2) * motionDim;
  const currentBase = (historyLength - 1) * motionDim;
  let squared = 0;
  let count = 0;
  for (let feature = featureStart; feature < featureEnd; feature += 1) {
    const expected = 2 * history[currentBase + feature] - history[previousBase + feature];
    const delta = values[feature] - expected;
    squared += delta * delta;
    count += 1;
  }
  return Math.sqrt(squared / count);
}

const rawResult = run('raw');
const seam = run('seam');
const balanced = run('balanced');
const strong = run('strong');
assert.deepEqual(rawResult.motion, raw);
assert.deepEqual(fullMotion, original);
assert.ok(seamVelocityMismatch(seam.motion, 0, manifest.root_dim)
  < seamVelocityMismatch(raw, 0, manifest.root_dim));
assert.ok(seamVelocityMismatch(seam.motion, rotationStart, rotationEnd)
  < seamVelocityMismatch(raw, rotationStart, rotationEnd));
assert.ok(secondDifferenceRms(balanced.motion, 0, 3)
  < secondDifferenceRms(raw, 0, 3));
assert.ok(secondDifferenceRms(balanced.motion, rotationStart, rotationEnd)
  < secondDifferenceRms(raw, rotationStart, rotationEnd));
assert.ok(secondDifferenceRms(strong.motion, 0, 3)
  < secondDifferenceRms(balanced.motion, 0, 3));
assert.ok(secondDifferenceRms(strong.motion, rotationStart, rotationEnd)
  < secondDifferenceRms(balanced.motion, rotationStart, rotationEnd));

for (const result of [seam, balanced, strong]) {
  assert.equal(result.motion[2 * motionDim], raw[2 * motionDim]);
  assert.equal(result.motion[2 * motionDim + 2], raw[2 * motionDim + 2]);
  for (let frame = 0; frame < generationFrames; frame += 1) {
    const base = frame * motionDim;
    const cosine = (
      result.motion[base + 3] * manifest.root_stats.std_eps[3]
      + manifest.root_stats.mean[3]
    );
    const sine = (
      result.motion[base + 4] * manifest.root_stats.std_eps[4]
      + manifest.root_stats.mean[4]
    );
    assert.ok(Math.abs(Math.hypot(cosine, sine) - 1) < 1e-6);
  }
}

for (let frame = 0; frame < generationFrames; frame += 1) {
  for (let feature = motionDim - 4; feature < motionDim; feature += 1) {
    assert.equal(balanced.motion[frame * motionDim + feature], raw[frame * motionDim + feature]);
  }
}

for (const mode of ['raw', 'balanced', 'strong']) {
  for (let warmup = 0; warmup < 50; warmup += 1) run(mode);
  const start = performance.now();
  for (let iteration = 0; iteration < 1000; iteration += 1) run(mode);
  const meanMs = (performance.now() - start) / 1000;
  console.log(`${mode}_mean_ms=${meanMs.toFixed(4)}`);
}
console.log('POSTPROCESS_REAL_TEST_PASS');
