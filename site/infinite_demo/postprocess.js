export const POSTPROCESS_MODES = Object.freeze({
  raw: Object.freeze({
    id: 'raw',
    label: '原始（对照）',
    description: '不增加任何前端滤波，完整保留模型输出。',
    seamFrames: 0,
    seamStrength: 0,
    rootStrength: 0,
    rootPasses: 0,
    poseStrength: 0,
    posePasses: 0,
    displayInterpolation: false,
  }),
  interp: Object.freeze({
    id: 'interp',
    label: '仅显示插帧',
    description: '在 20 FPS 相邻动作帧间插值，单独排查屏幕采样顿挫。',
    seamFrames: 0,
    seamStrength: 0,
    rootStrength: 0,
    rootPasses: 0,
    poseStrength: 0,
    posePasses: 0,
    displayInterpolation: true,
  }),
  seam: Object.freeze({
    id: 'seam',
    label: '仅接缝',
    description: '8 帧速度惯性化，只压低续写窗口边界的闪跳。',
    seamFrames: 8,
    seamStrength: 0.75,
    rootStrength: 0,
    rootPasses: 0,
    poseStrength: 0,
    posePasses: 0,
    displayInterpolation: false,
  }),
  root: Object.freeze({
    id: 'root',
    label: '路径稳化',
    description: '接缝惯性化 + 保端点根轨迹低通；动作细节不额外滤波。',
    seamFrames: 8,
    seamStrength: 0.75,
    rootStrength: 0.55,
    rootPasses: 1,
    poseStrength: 0,
    posePasses: 0,
    displayInterpolation: false,
  }),
  pose: Object.freeze({
    id: 'pose',
    label: '动作降噪',
    description: '接缝惯性化 + 6D 关节旋转低通；根路径不额外滤波。',
    seamFrames: 8,
    seamStrength: 0.75,
    rootStrength: 0,
    rootPasses: 0,
    poseStrength: 0.45,
    posePasses: 1,
    displayInterpolation: false,
  }),
  balanced: Object.freeze({
    id: 'balanced',
    label: '均衡（推荐）',
    description: '温和稳定路径、关节旋转和窗口接缝，优先保留响应速度。',
    seamFrames: 8,
    seamStrength: 0.75,
    rootStrength: 0.45,
    rootPasses: 1,
    poseStrength: 0.35,
    posePasses: 1,
    displayInterpolation: true,
  }),
  strong: Object.freeze({
    id: 'strong',
    label: '强力平滑',
    description: '两遍路径/动作低通；最稳，但快速转向和动作细节可能变软。',
    seamFrames: 10,
    seamStrength: 0.85,
    rootStrength: 0.70,
    rootPasses: 2,
    poseStrength: 0.55,
    posePasses: 2,
    displayInterpolation: true,
  }),
});

export function resolvePostprocessMode(value) {
  return POSTPROCESS_MODES[String(value || '').trim()] || POSTPROCESS_MODES.raw;
}

function nowMs() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function assertLayout({
  fullMotion,
  historyLength,
  generationFrames,
  motionDim,
  rootDim,
  rotationStart,
  rotationEnd,
  excludedTailFeatures,
}) {
  const fullFrames = historyLength + generationFrames;
  if (!(fullMotion instanceof Float32Array) || fullMotion.length !== fullFrames * motionDim) {
    throw new Error('postprocess fullMotion shape mismatch');
  }
  if (!Number.isInteger(historyLength) || historyLength < 0) throw new Error('invalid historyLength');
  if (!Number.isInteger(generationFrames) || generationFrames < 1) throw new Error('invalid generationFrames');
  if (!Number.isInteger(motionDim) || motionDim < 1) throw new Error('invalid motionDim');
  if (rootDim < 5 || rootDim > motionDim) throw new Error('invalid rootDim');
  if (rotationStart < rootDim || rotationEnd < rotationStart || rotationEnd > motionDim) {
    throw new Error('invalid rotation feature range');
  }
  if (excludedTailFeatures < 0 || excludedTailFeatures > motionDim) {
    throw new Error('invalid excludedTailFeatures');
  }
}

function applySeamInertialization({
  values,
  historyFrames,
  historyLength,
  generationFrames,
  motionDim,
  excludedTailFeatures,
  frames,
  strength,
}) {
  if (!historyFrames || historyFrames.length < 2 || frames <= 0 || strength <= 0) return;
  const count = Math.min(frames, generationFrames);
  const featureEnd = motionDim - excludedTailFeatures;
  const previous = historyFrames.at(-2);
  const current = historyFrames.at(-1);
  if (!previous || !current || previous.length !== motionDim || current.length !== motionDim) {
    throw new Error('postprocess history frame shape mismatch');
  }
  const generatedBase = historyLength * motionDim;
  const offset = new Float32Array(featureEnd);
  for (let feature = 0; feature < featureEnd; feature += 1) {
    offset[feature] = 2 * current[feature] - previous[feature] - values[generatedBase + feature];
  }
  for (let frame = 0; frame < count; frame += 1) {
    const phase = count === 1 ? 0 : frame / (count - 1);
    const decay = 1 - (3 * phase * phase - 2 * phase * phase * phase);
    const base = (historyLength + frame) * motionDim;
    for (let feature = 0; feature < featureEnd; feature += 1) {
      values[base + feature] += strength * decay * offset[feature];
    }
  }
}

function flattenContext(historyFrames, motion, motionDim, generationFrames) {
  const historyCount = historyFrames?.length ?? 0;
  const values = new Float32Array((historyCount + generationFrames) * motionDim);
  for (let frame = 0; frame < historyCount; frame += 1) {
    const source = historyFrames[frame];
    if (!source || source.length !== motionDim) throw new Error('postprocess context shape mismatch');
    values.set(source, frame * motionDim);
  }
  values.set(motion, historyCount * motionDim);
  return { values, historyCount };
}

function lowPassRange({
  motion,
  historyFrames,
  motionDim,
  generationFrames,
  featureStart,
  featureEnd,
  strength,
  passes,
}) {
  if (strength <= 0 || passes <= 0 || featureStart >= featureEnd) return;
  const { values, historyCount } = flattenContext(
    historyFrames,
    motion,
    motionDim,
    generationFrames,
  );
  const totalFrames = historyCount + generationFrames;
  for (let pass = 0; pass < passes; pass += 1) {
    const source = values.slice();
    for (let frame = historyCount; frame < totalFrames; frame += 1) {
      const previous = Math.max(0, frame - 1);
      const next = Math.min(totalFrames - 1, frame + 1);
      const base = frame * motionDim;
      const previousBase = previous * motionDim;
      const nextBase = next * motionDim;
      for (let feature = featureStart; feature < featureEnd; feature += 1) {
        const filtered = (
          source[previousBase + feature]
          + 2 * source[base + feature]
          + source[nextBase + feature]
        ) * 0.25;
        values[base + feature] = source[base + feature]
          + strength * (filtered - source[base + feature]);
      }
    }
  }
  motion.set(values.subarray(historyCount * motionDim));
}

function normalizeHeading(motion, generationFrames, motionDim, rootMean, rootStd) {
  for (let frame = 0; frame < generationFrames; frame += 1) {
    const base = frame * motionDim;
    const cosine = motion[base + 3] * rootStd[3] + rootMean[3];
    const sine = motion[base + 4] * rootStd[4] + rootMean[4];
    const inverse = 1 / Math.max(Math.hypot(cosine, sine), 1e-8);
    motion[base + 3] = (cosine * inverse - rootMean[3]) / rootStd[3];
    motion[base + 4] = (sine * inverse - rootMean[4]) / rootStd[4];
  }
}

function preserveRootAnchors({
  motion,
  anchors,
  protectedRootFrames,
  generationFrames,
  motionDim,
  rootDim,
}) {
  const lastFrame = generationFrames - 1;
  const anchorFrames = [...new Set([0, lastFrame, ...(protectedRootFrames || [])])]
    .filter((frame) => Number.isInteger(frame) && frame >= 0 && frame <= lastFrame)
    .sort((left, right) => left - right);
  for (let feature = 0; feature < Math.min(3, rootDim); feature += 1) {
    for (let anchor = 0; anchor < anchorFrames.length - 1; anchor += 1) {
      const beginFrame = anchorFrames[anchor];
      const endFrame = anchorFrames[anchor + 1];
      const beginBase = beginFrame * motionDim;
      const endBase = endFrame * motionDim;
      const beginCorrection = anchors[beginBase + feature] - motion[beginBase + feature];
      const endCorrection = anchors[endBase + feature] - motion[endBase + feature];
      for (let frame = beginFrame; frame <= endFrame; frame += 1) {
        const phase = endFrame === beginFrame ? 0 : (frame - beginFrame) / (endFrame - beginFrame);
        motion[frame * motionDim + feature] += (
          (1 - phase) * beginCorrection + phase * endCorrection
        );
      }
    }
  }
  const lastBase = lastFrame * motionDim;
  for (const feature of [3, 4]) {
    motion[feature] = anchors[feature];
    motion[lastBase + feature] = anchors[lastBase + feature];
  }
}

function diagnostics(raw, display, generationFrames, motionDim, rootStd, rotationStart, rotationEnd) {
  let rootSquared = 0;
  let rootMax = 0;
  let poseAbsolute = 0;
  let poseCount = 0;
  for (let frame = 0; frame < generationFrames; frame += 1) {
    const base = frame * motionDim;
    const dx = (display[base] - raw[base]) * rootStd[0];
    const dz = (display[base + 2] - raw[base + 2]) * rootStd[2];
    const distance = Math.hypot(dx, dz);
    rootSquared += distance * distance;
    rootMax = Math.max(rootMax, distance);
    for (let feature = rotationStart; feature < rotationEnd; feature += 1) {
      poseAbsolute += Math.abs(display[base + feature] - raw[base + feature]);
      poseCount += 1;
    }
  }
  const lastBase = (generationFrames - 1) * motionDim;
  const endpointDx = (display[lastBase] - raw[lastBase]) * rootStd[0];
  const endpointDz = (display[lastBase + 2] - raw[lastBase + 2]) * rootStd[2];
  return {
    root_rms_deviation_m: Math.sqrt(rootSquared / generationFrames),
    root_max_deviation_m: rootMax,
    root_endpoint_deviation_m: Math.hypot(endpointDx, endpointDz),
    pose_mean_abs_normalized: poseCount ? poseAbsolute / poseCount : 0,
  };
}

export function postprocessMotionForDisplay({
  fullMotion,
  historyFrames = null,
  historyLength,
  generationFrames,
  motionDim,
  rootDim,
  rotationStart,
  rotationEnd,
  excludedTailFeatures = 4,
  rootMean,
  rootStd,
  mode: modeValue,
  protectedRootFrames = [],
}) {
  const start = nowMs();
  const mode = resolvePostprocessMode(modeValue?.id ?? modeValue);
  assertLayout({
    fullMotion,
    historyLength,
    generationFrames,
    motionDim,
    rootDim,
    rotationStart,
    rotationEnd,
    excludedTailFeatures,
  });
  if (!rootMean || !rootStd || rootMean.length < rootDim || rootStd.length < rootDim) {
    throw new Error('postprocess root statistics shape mismatch');
  }

  const working = fullMotion.slice();
  applySeamInertialization({
    values: working,
    historyFrames,
    historyLength,
    generationFrames,
    motionDim,
    excludedTailFeatures,
    frames: mode.seamFrames,
    strength: mode.seamStrength,
  });
  const begin = historyLength * motionDim;
  const end = (historyLength + generationFrames) * motionDim;
  const raw = fullMotion.slice(begin, end);
  const motion = working.slice(begin, end);
  const rootAnchors = motion.slice();
  for (const frame of protectedRootFrames || []) {
    if (!Number.isInteger(frame) || frame < 0 || frame >= generationFrames) continue;
    const base = frame * motionDim;
    // Sparse mouse constraints are x/z conditions.  They take precedence over
    // seam correction even when the constrained frame is inside the seam span.
    rootAnchors[base] = raw[base];
    rootAnchors[base + 2] = raw[base + 2];
  }

  lowPassRange({
    motion,
    historyFrames,
    motionDim,
    generationFrames,
    featureStart: 0,
    featureEnd: rootDim,
    strength: mode.rootStrength,
    passes: mode.rootPasses,
  });
  if (
    (mode.rootStrength > 0 && mode.rootPasses > 0)
    || (mode.seamStrength > 0 && protectedRootFrames.length > 0)
  ) {
    preserveRootAnchors({
      motion,
      anchors: rootAnchors,
      protectedRootFrames,
      generationFrames,
      motionDim,
      rootDim,
    });
  }
  if (mode.seamStrength > 0 || mode.rootStrength > 0) {
    // Seam extrapolation and component-wise low-pass do not preserve the
    // physical cos/sin unit circle.  Normalize once after all root edits;
    // this preserves heading angle while avoiding malformed rotations.
    normalizeHeading(motion, generationFrames, motionDim, rootMean, rootStd);
  }
  lowPassRange({
    motion,
    historyFrames,
    motionDim,
    generationFrames,
    featureStart: rotationStart,
    featureEnd: rotationEnd,
    strength: mode.poseStrength,
    passes: mode.posePasses,
  });

  const allFinite = motion.every(Number.isFinite);
  if (!allFinite) throw new Error(`postprocess ${mode.id} produced NaN/Inf`);
  return {
    motion,
    mode,
    elapsedMs: nowMs() - start,
    allFinite,
    diagnostics: diagnostics(
      raw,
      motion,
      generationFrames,
      motionDim,
      rootStd,
      rotationStart,
      rotationEnd,
    ),
  };
}
