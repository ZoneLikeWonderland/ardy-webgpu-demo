import * as ort from '../vendor/onnxruntime-web-1.27.0/ort.webgpu.bundle.min.mjs';
import {
  POSTPROCESS_MODES,
  postprocessMotionForDisplay,
  resolvePostprocessMode,
} from './postprocess.js';

const runtimeBase = new URL('../vendor/onnxruntime-web-1.27.0/', import.meta.url).href;
ort.env.wasm.wasmPaths = runtimeBase;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';
ort.env.webgpu.powerPreference = 'high-performance';

const urlOptions = new URLSearchParams(window.location.search);
const benchmarkMode = urlOptions.get('benchmark') === '1';
const gpuProfilingEnabled = urlOptions.get('profile') === '1';
// GitHub Pages is a static deployment. The optional server bridge is useful
// only for the local comparison server and must be explicitly enabled.
const serverBridgeEnabled = urlOptions.get('bridge') === '1';
const candidateId = (urlOptions.get('candidate') || '').trim();
const frontendProtocol = 4;
const initialPostprocessMode = resolvePostprocessMode(
  urlOptions.get('post')
  || localStorage.getItem('ardy-infinite-demo-postprocess')
  || 'balanced',
).id;
let activeProfileContext = null;
let generationProfileRows = [];
let demoCommandBusy = false;

function demoAssetUrl(filename) {
  return candidateId
    ? `./infinite_demo/candidates/${candidateId}/${filename}`
    : `./infinite_demo/${filename}`;
}

if (gpuProfilingEnabled) {
  ort.env.webgpu.profiling = {
    mode: 'default',
    ondata: (data) => {
      const startTime = Number(data?.startTime ?? 0);
      const endTime = Number(data?.endTime ?? startTime);
      generationProfileRows.push({
        context: activeProfileContext ? { ...activeProfileContext } : null,
        version: Number(data?.version ?? 1),
        start_time_ms: startTime,
        end_time_ms: endTime,
        duration_ms: Math.max(0, endTime - startTime),
        kernel_id: Number(data?.kernelId ?? -1),
        kernel_name: String(data?.kernelName ?? ''),
        kernel_type: String(data?.kernelType ?? ''),
        program_name: String(data?.programName ?? ''),
      });
    },
  };
}

const ui = {
  canvas: document.querySelector('#viewport'),
  play: document.querySelector('#play-button'),
  restart: document.querySelector('#restart-button'),
  clearWaypoints: document.querySelector('#clear-waypoints-button'),
  precision: document.querySelector('#precision-select'),
  prompt: document.querySelector('#prompt-select'),
  postprocess: document.querySelector('#postprocess-select'),
  postprocessDescription: document.querySelector('#postprocess-description'),
  gpuBadge: document.querySelector('#gpu-badge'),
  precisionBadge: document.querySelector('#precision-badge'),
  state: document.querySelector('#state-value'),
  frame: document.querySelector('#frame-value'),
  target: document.querySelector('#target-value'),
  timing: document.querySelector('#timing-value'),
  postprocessValue: document.querySelector('#postprocess-value'),
  log: document.querySelector('#log'),
  loading: document.querySelector('#loading'),
  loadingTitle: document.querySelector('#loading-title'),
  loadingDetail: document.querySelector('#loading-detail'),
  timeline: document.querySelector('#timeline-canvas'),
  timelineTime: document.querySelector('#timeline-time'),
  timelineFrame: document.querySelector('#timeline-frame'),
};

const ctx = ui.canvas.getContext('2d');
const timelineCtx = ui.timeline.getContext('2d');
const sessions = {
  encoder: null,
  flow: null,
  decoder: null,
  finalizeInitial: null,
  finalizeContinuation: null,
};
const state = {
  manifest: null,
  precision: 'fp16',
  ready: false,
  generating: false,
  replanRequested: false,
  playing: false,
  frameIndex: 0,
  motionFrames: [],
  displayMotionFrames: [],
  jointFrames: [],
  segments: [],
  waypoints: [],
  motionStats: null,
  generationCount: 0,
  epoch: 0,
  lastTick: performance.now(),
  frameAccumulator: 0,
  cameraX: 0,
  cameraZ: 0,
  width: 1,
  height: 1,
  timelineWidth: 1,
  timelineHeight: 1,
  dpr: 1,
  adapter: {},
  suppressNextTick: false,
  benchmarking: false,
  promptMetadata: null,
  promptFeatureBuffer: null,
  selectedPromptIndex: 0,
  postprocessMode: initialPostprocessMode,
  lastPostprocessMs: 0,
  lastPostprocessFrames: 0,
};
const clientId = localStorage.getItem('ardy-infinite-demo-client-id') || crypto.randomUUID();
localStorage.setItem('ardy-infinite-demo-client-id', clientId);

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);
const CORE_NEUTRAL_JOINTS = [
  [0, 0, 0], [0, .0709891, -.0473261], [0, .1642033182, -.0637622892],
  [0, .2584953309, -.0720118225], [0, .3531475266, -.0720119171], [0, .6016095506, -.036517567],
  [0, .7297793389, -.0139178969], [-.0319949, .5259195719, -.0186872922],
  [-.1909029, .5259195169, -.0186872922], [-.4863389, .5259194145, -.0186872922],
  [-.7189909, .525919334, -.0186872922], [-.7886024, .5259193098, -.0186872922],
  [-.7468354936, .5073562715, .0277204242], [.0319949, .5259195719, -.0186872922],
  [.1909029, .5259195719, -.0186872922], [.4863389, .5259195719, -.0186872922],
  [.7189909, .5259195719, -.0186872922], [.7886024, .5259195719, -.0186872922],
  [.7468355, .5073565192, .0277204242], [-.0949182, -.0277289, 0], [-.0949182, -.4398469, 0],
  [-.0949182, -.8959379, 0], [-.0949182, -.9544128252, .1606582662],
  [.0949182, -.0277289, 0], [.0949182, -.4398469, 0], [.0949182, -.8959379, 0],
  [.0949182, -.9544128252, .1606582662],
];
const interpolatedJointScratch = new Float32Array(CORE_NEUTRAL_JOINTS.length * 3);
const interpolatedMatrixScratch = new Float32Array(CORE_NEUTRAL_JOINTS.length * 9);

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  ui.log.textContent += `[${stamp}] ${message}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

function setLoading(visible, title = '', detail = '') {
  ui.loading.classList.toggle('hidden', !visible);
  if (title) ui.loadingTitle.textContent = title;
  if (detail) ui.loadingDetail.textContent = detail;
}

function setStatus(text) {
  ui.state.textContent = text;
}

function currentPostprocessMode() {
  return resolvePostprocessMode(state.postprocessMode);
}

function updatePostprocessDescription() {
  const mode = currentPostprocessMode();
  ui.postprocess.value = mode.id;
  ui.postprocessDescription.textContent = `${mode.description} 仅作用于显示，不回灌模型 history。`;
}

function configurePostprocessUi() {
  ui.postprocess.replaceChildren();
  for (const mode of Object.values(POSTPROCESS_MODES)) {
    const option = document.createElement('option');
    option.value = mode.id;
    option.textContent = mode.label;
    ui.postprocess.append(option);
  }
  updatePostprocessDescription();
}

async function telemetry(kind, payload = {}) {
  if (!serverBridgeEnabled) return;
  try {
    await fetch(`/api/demo/${kind}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        client_id: clientId,
        model: state.manifest?.model ?? null,
        model_release: state.manifest?.model_release ?? null,
        precision: state.precision,
        ready: state.ready,
        generating: state.generating,
        playing: state.playing,
        frame_index: state.frameIndex,
        generated_frames: state.motionFrames.length,
        user_agent: navigator.userAgent,
        adapter: state.adapter,
        frontend_protocol: frontendProtocol,
        postprocess_mode: state.postprocessMode,
        prompt_id: selectedPromptEntry()?.prompt_id ?? null,
        prompt_text: selectedPromptEntry()?.text ?? null,
        ...payload,
      }),
    });
  } catch {
    // Telemetry is only for remote debugging and must never stop inference.
  }
}

function float32ToFloat16Bits(value) {
  f32Scratch[0] = value;
  const bits = u32Scratch[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) {
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  }
  mantissa += 0x1000;
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
    if (exponent >= 31) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

function float16BitsToFloat32(bits) {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let out;
  if (exponent === 0) {
    if (mantissa === 0) {
      out = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      out = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 31) {
    out = sign | 0x7f800000 | (mantissa << 13);
  } else {
    out = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  u32Scratch[0] = out;
  return f32Scratch[0];
}

function toHalfArray(values) {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) result[i] = float32ToFloat16Bits(values[i]);
  return result;
}

function selectedPromptEntry() {
  return state.promptMetadata?.entries?.[state.selectedPromptIndex] ?? null;
}

function selectedPromptFeatureBits() {
  const metadata = state.promptMetadata;
  if (!metadata || !state.promptFeatureBuffer) throw new Error('Qwen prompt 特征尚未加载');
  const begin = state.selectedPromptIndex * metadata.feature_dim;
  return new Uint16Array(
    state.promptFeatureBuffer,
    begin * Uint16Array.BYTES_PER_ELEMENT,
    metadata.feature_dim,
  );
}

function selectedPromptTensor() {
  const bits = selectedPromptFeatureBits();
  const dims = [1, state.promptMetadata.feature_dim];
  if (state.precision === 'fp16') return new ort.Tensor('float16', bits, dims);
  const values = new Float32Array(bits.length);
  for (let index = 0; index < bits.length; index += 1) {
    values[index] = float16BitsToFloat32(bits[index]);
  }
  return new ort.Tensor('float32', values, dims);
}

async function loadPromptFeatures() {
  if (!state.manifest.text_conditioning_enabled) return;
  const metadataUrl = state.manifest.text_conditioning?.prompt_bundle_url;
  if (!metadataUrl) throw new Error('manifest 缺少 prompt feature bundle');
  const metadataResponse = await fetch(`./${metadataUrl}`, { cache: 'no-cache' });
  if (!metadataResponse.ok) throw new Error(`prompt metadata HTTP ${metadataResponse.status}`);
  const metadata = await metadataResponse.json();
  const expectedDim = state.manifest.text_conditioning.feature_dim;
  if (
    metadata.schema !== 'ardy_browser_prompt_features_v1'
    || metadata.storage_dtype !== 'float16_le'
    || metadata.compression !== 'none'
    || metadata.feature_dim !== expectedDim
    || metadata.entries.length !== metadata.count
  ) {
    throw new Error('prompt feature metadata 与当前模型不匹配');
  }
  const featureResponse = await fetch(`./${metadata.feature_url}`, { cache: 'no-cache' });
  if (!featureResponse.ok) throw new Error(`prompt features HTTP ${featureResponse.status}`);
  const featureBuffer = await featureResponse.arrayBuffer();
  if (featureBuffer.byteLength !== metadata.size_bytes || featureBuffer.byteLength !== metadata.count * metadata.feature_dim * 2) {
    throw new Error('prompt feature binary 大小错误');
  }
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', featureBuffer));
  const digestHex = [...digest].map((value) => value.toString(16).padStart(2, '0')).join('');
  if (digestHex !== metadata.sha256) throw new Error('prompt feature binary SHA256 不匹配');

  state.promptMetadata = metadata;
  state.promptFeatureBuffer = featureBuffer;
  const defaultIndex = metadata.entries.findIndex(
    (entry) => entry.prompt_id === metadata.default_prompt_id,
  );
  state.selectedPromptIndex = defaultIndex >= 0 ? defaultIndex : 0;
  ui.prompt.replaceChildren(...metadata.entries.map((entry, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = entry.text || '(无文本 / unconditional)';
    option.title = `${entry.family} · prompt ${entry.prompt_id}`;
    return option;
  }));
  ui.prompt.value = String(state.selectedPromptIndex);
  log(`Qwen prompt 特征：${metadata.count} 条 × ${metadata.feature_dim}，FP16 无压缩，${(metadata.size_bytes / 1048576).toFixed(2)} MiB`);
}

function roundArrayToHalf(values, start = 0, end = values.length) {
  for (let i = start; i < end; i += 1) values[i] = float16BitsToFloat32(float32ToFloat16Bits(values[i]));
}

function floatTensor(values, dims, precision = state.precision) {
  if (precision === 'fp16') return new ort.Tensor('float16', toHalfArray(values), dims);
  const data = values instanceof Float32Array ? values : Float32Array.from(values);
  return new ort.Tensor('float32', data, dims);
}

function int64Tensor(values, dims) {
  return new ort.Tensor('int64', BigInt64Array.from(values, (value) => BigInt(value)), dims);
}

async function tensorToFloat32(tensor) {
  const data = await tensor.getData();
  if (tensor.type === 'float32') return new Float32Array(data);
  if (tensor.type !== 'float16') throw new Error(`不支持的输出 dtype: ${tensor.type}`);
  if (globalThis.Float16Array && data instanceof globalThis.Float16Array) return Float32Array.from(data);
  const bits = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
  const result = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) result[i] = float16BitsToFloat32(bits[i]);
  return result;
}

function disposeMap(values) {
  for (const tensor of Object.values(values)) tensor?.dispose?.();
}

async function runProfiled(sessionName, detail, session, feeds, fetches = undefined) {
  const previous = activeProfileContext;
  activeProfileContext = { session: sessionName, ...detail };
  try {
    return fetches === undefined ? await session.run(feeds) : await session.run(feeds, fetches);
  } finally {
    // ORT emits the WebGPU profiling callback while completing a run. Keep the
    // phase label alive through the following microtask as a defensive margin.
    await Promise.resolve();
    activeProfileContext = previous;
  }
}

function profileGroup(row) {
  const label = `${row.kernel_name} ${row.program_name}`.toLowerCase();
  if (row.context?.session === 'encoder') return 'history_encoder';
  if (row.context?.session === 'decoder') return 'motion_decoder';
  if (row.context?.session === 'flow') return 'nfe1_flow';
  return row.context?.session || 'unattributed';
}

function summarizeGpuProfile(rows) {
  if (!gpuProfilingEnabled) return { enabled: false };
  const groups = new Map();
  const programs = new Map();
  let total = 0;
  for (const row of rows) {
    const duration = Number.isFinite(row.duration_ms) ? row.duration_ms : 0;
    total += duration;
    const group = profileGroup(row);
    const groupValue = groups.get(group) || { count: 0, duration_ms: 0 };
    groupValue.count += 1;
    groupValue.duration_ms += duration;
    groups.set(group, groupValue);
    const name = row.kernel_name || row.program_name || row.kernel_type || 'unknown';
    const programValue = programs.get(name) || { count: 0, duration_ms: 0 };
    programValue.count += 1;
    programValue.duration_ms += duration;
    programs.set(name, programValue);
  }
  const groupRows = Object.fromEntries(
    [...groups.entries()].map(([name, value]) => [name, {
      count: value.count,
      duration_ms: value.duration_ms,
      share: total ? value.duration_ms / total : 0,
    }]),
  );
  const topKernels = [...programs.entries()]
    .map(([name, value]) => ({ name, ...value }))
    .sort((left, right) => right.duration_ms - left.duration_ms)
    .slice(0, 20);
  return {
    enabled: true,
    record_count: rows.length,
    total_kernel_ms: total,
    groups: groupRows,
    top_kernels: topKernels,
  };
}

function mulberry32(seed) {
  return () => {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianNoise(count, seed) {
  const random = mulberry32(seed);
  const values = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const u1 = Math.max(random(), 1e-7);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    values[i] = radius * Math.cos(2 * Math.PI * u2);
    if (i + 1 < count) values[i + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return values;
}

function yieldToAnimationFrame() {
  // requestAnimationFrame can be suspended indefinitely in a hidden tab.  The
  // demo also runs server-driven validation, so always provide a timer fallback
  // instead of making unattended diagnostics depend on tab visibility.
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const timeout = setTimeout(finish, 100);
    requestAnimationFrame(() => {
      clearTimeout(timeout);
      finish();
    });
  });
}

async function releasePrecisionSessions() {
  for (const key of ['encoder', 'flow', 'decoder']) {
    if (sessions[key]) await sessions[key].release();
    sessions[key] = null;
  }
}

async function createWebGpuSession(url) {
  return ort.InferenceSession.create(url, {
    executionProviders: [{ name: 'webgpu', validationMode: 'full' }],
    graphOptimizationLevel: 'all',
    enableProfiling: false,
  });
}

async function loadSharedSessions() {
  const utility = state.manifest.utilities;
  setLoading(true, '加载固定状态图', 'ARDY 根坐标、FSQ 与首段/续写 token 排布使用浏览器 WASM');
  const wasmOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  [sessions.finalizeInitial, sessions.finalizeContinuation] = await Promise.all([
    ort.InferenceSession.create(utility.finalize_initial.url, wasmOptions),
    ort.InferenceSession.create(utility.finalize_continuation.url, wasmOptions),
  ]);
}

async function loadPrecision() {
  state.ready = false;
  ui.play.disabled = true;
  ui.restart.disabled = true;
  ui.clearWaypoints.disabled = true;
  ui.precision.disabled = true;
  ui.prompt.disabled = true;
  ui.postprocess.disabled = true;
  await releasePrecisionSessions();
  const spec = state.manifest.models;
  state.precision = state.manifest.precision;
  const precisionLabel = state.precision.toUpperCase();
  setLoading(true, `加载 ${precisionLabel} history encoder`, `${(spec.encoder.size_bytes / 1048576).toFixed(1)} MiB`);
  const start = performance.now();
  sessions.encoder = await createWebGpuSession(spec.encoder.url);
  setLoading(true, `加载 ${precisionLabel} NFE=1 flow`, `${(spec.flow.size_bytes / 1048576).toFixed(1)} MiB，首次编译会较慢`);
  sessions.flow = await createWebGpuSession(spec.flow.url);
  setLoading(true, `加载 ${precisionLabel} motion decoder`, `${(spec.decoder.size_bytes / 1048576).toFixed(1)} MiB`);
  sessions.decoder = await createWebGpuSession(spec.decoder.url);
  state.ready = true;
  ui.precisionBadge.textContent = `${precisionLabel} · NFE=1`;
  ui.play.disabled = false;
  ui.restart.disabled = false;
  ui.clearWaypoints.disabled = false;
  ui.precision.disabled = true;
  ui.prompt.disabled = !state.promptMetadata;
  ui.postprocess.disabled = false;
  setLoading(false);
  setStatus('就绪');
  log(`${precisionLabel} 全 student 模型加载完成：${(performance.now() - start).toFixed(0)} ms`);
  telemetry('report', { event: 'models_ready', load_ms: performance.now() - start });
}

function roundTiesToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return Math.abs(lower % 2) === 0 ? lower : lower + 1;
}

function requantizeLatent(values) {
  const stats = state.manifest.post_quantization_stats;
  const halfWidth = stats.levels / 2;
  const result = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const raw = Math.max(-1, Math.min(1, values[index] * stats.std_eps[index] + stats.mean[index]));
    const quantized = roundTiesToEven(raw * halfWidth) / halfWidth;
    result[index] = (quantized - stats.mean[index]) / stats.std_eps[index];
  }
  return result;
}

async function encodeHistory(historyFrames) {
  const totalStart = performance.now();
  const feedStart = performance.now();
  const m = state.manifest;
  const values = new Float32Array(m.history_frames * m.body_dim);
  for (let frame = 0; frame < m.history_frames; frame += 1) {
    values.set(historyFrames[frame].subarray(m.root_dim), frame * m.body_dim);
  }
  const feeds = { normalized_body: floatTensor(values, [1, m.history_frames, m.body_dim]) };
  const feedBuildMs = performance.now() - feedStart;
  let outputs;
  let runMs = 0;
  let outputCopyMs = 0;
  try {
    const runStart = performance.now();
    outputs = await runProfiled('encoder', {}, sessions.encoder, feeds);
    runMs = performance.now() - runStart;
    const copyStart = performance.now();
    const encodedRaw = await tensorToFloat32(outputs.output);
    outputCopyMs = performance.now() - copyStart;
    const encoded = requantizeLatent(encodedRaw);
    const mean = m.root_stats.mean;
    const std = m.root_stats.std_eps;
    const translation = new Float32Array(3);
    translation[0] = historyFrames[3][0] * std[0] + mean[0];
    translation[1] = 0;
    translation[2] = historyFrames[3][2] * std[2] + mean[2];
    const historyHybrid = new Float32Array(m.hybrid_dim);
    for (let frame = 0; frame < m.history_frames; frame += 1) {
      for (let feature = 0; feature < m.root_dim; feature += 1) {
        let world = historyFrames[frame][feature] * std[feature] + mean[feature];
        if (feature === 0) world -= translation[0];
        if (feature === 2) world -= translation[2];
        historyHybrid[frame * m.root_dim + feature] = (world - mean[feature]) / std[feature];
      }
    }
    historyHybrid.set(encoded, 20);
    const headingCos = historyFrames[0][3] * std[3] + mean[3];
    const headingSin = historyFrames[0][4] * std[4] + mean[4];
    const headingAngle = Math.atan2(headingSin, headingCos);
    const firstHeading = new Float32Array([Math.cos(headingAngle), Math.sin(headingAngle)]);
    return {
      historyHybrid,
      translation,
      firstHeading,
      timing: {
        feed_build_ms: feedBuildMs,
        session_run_ms: runMs,
        output_copy_ms: outputCopyMs,
        total_ms: performance.now() - totalStart,
        note: 'session_run includes default CPU input upload, WebGPU compute, synchronization, and CPU output download',
      },
    };
  } finally {
    disposeMap(feeds);
    if (outputs) disposeMap(outputs);
  }
}

function rootPositionFromFrames(frames, frameIndex) {
  if (!frames.length) return { x: 0, y: 0, z: 0 };
  const frame = frames[Math.min(frameIndex, frames.length - 1)];
  const { mean, std_eps: std } = state.manifest.root_stats;
  return {
    x: frame[0] * std[0] + mean[0],
    y: frame[1] * std[1] + mean[1],
    z: frame[2] * std[2] + mean[2],
  };
}

function currentRootPosition(frameIndex = state.frameIndex) {
  return rootPositionFromFrames(state.motionFrames, frameIndex);
}

function displayRootPosition(frameIndex = state.frameIndex) {
  return rootPositionFromFrames(
    state.displayMotionFrames.length ? state.displayMotionFrames : state.motionFrames,
    frameIndex,
  );
}

function displayInterpolationAlpha() {
  if (!currentPostprocessMode().displayInterpolation || !state.playing || state.generating) return 0;
  if (state.frameIndex >= state.displayMotionFrames.length - 1) return 0;
  return Math.max(0, Math.min(1, state.frameAccumulator));
}

function currentDisplayRootPosition() {
  const current = displayRootPosition(state.frameIndex);
  const alpha = displayInterpolationAlpha();
  if (alpha <= 0) return current;
  const next = displayRootPosition(state.frameIndex + 1);
  return {
    x: current.x + (next.x - current.x) * alpha,
    y: current.y + (next.y - current.y) * alpha,
    z: current.z + (next.z - current.z) * alpha,
  };
}

function waypointDiagnostics() {
  return state.waypoints
    .filter((waypoint) => waypoint.frame < state.motionFrames.length)
    .map((waypoint) => {
      const actual = currentRootPosition(waypoint.frame);
      const displayed = displayRootPosition(waypoint.frame);
      return {
        frame: waypoint.frame,
        target_x: waypoint.x,
        target_z: waypoint.z,
        actual_x: actual.x,
        actual_z: actual.z,
        error_m: Math.hypot(actual.x - waypoint.x, actual.z - waypoint.z),
        display_x: displayed.x,
        display_z: displayed.z,
        display_error_m: Math.hypot(displayed.x - waypoint.x, displayed.z - waypoint.z),
      };
    });
}

function buildPathCondition({ historyLength, historyStart, translation }) {
  const manifest = state.manifest;
  const path = new Float32Array(manifest.path_frames * 3);

  if (state.waypoints.length) {
    const { mean, std_eps: std } = manifest.root_stats;
    const historyEnd = historyStart + historyLength - 1;
    const windowEnd = historyStart + manifest.path_frames - 1;
    // RootKeyframe2DSet defaults to dense_path=false.  A normal mouse click
    // therefore contributes only one sparse x/z constraint at current+60.
    for (const waypoint of state.waypoints) {
      if (waypoint.frame <= historyEnd || waypoint.frame > windowEnd) continue;
      const localFrame = waypoint.frame - historyStart;
      const base = localFrame * 3;
      path[base] = ((waypoint.x - translation[0]) - mean[0]) / std[0];
      path[base + 1] = ((waypoint.z - translation[2]) - mean[2]) / std[2];
      path[base + 2] = 1;
    }
  }
  return path;
}

async function runFlow({ initial, historyHybrid, translation, firstHeading, historyStart }) {
  const totalStart = performance.now();
  const setupStart = performance.now();
  const m = state.manifest;
  const historyLength = initial ? 0 : m.history_frames;
  const noise = gaussianNoise(10 * m.hybrid_dim, 20260713 + state.generationCount * 7919);
  const pathCondition = buildPathCondition({ historyLength, historyStart, translation });
  const setupMs = performance.now() - setupStart;
  const feedStart = performance.now();
  const feeds = {
    noise: floatTensor(noise, [1, 10, m.hybrid_dim]),
    history_hybrid: floatTensor(historyHybrid, [1, 1, m.hybrid_dim]),
    path_condition: floatTensor(pathCondition, [1, m.path_frames, 3]),
    first_heading: floatTensor(firstHeading, [1, 2]),
    has_history: floatTensor(new Float32Array([initial ? 0 : 1]), [1, 1]),
  };
  if (m.text_conditioning_enabled) {
    feeds.text_feature = selectedPromptTensor();
  }
  if (m.heading_conditioning_enabled) {
    // Mouse waypoints use the released sparse root x/z constraint.  A zero
    // validity channel explicitly means no separate keyboard-heading command.
    feeds.heading_condition = floatTensor(
      new Float32Array(m.path_frames * 3),
      [1, m.path_frames, 3],
    );
  }
  const feedBuildMs = performance.now() - feedStart;
  let outputs;
  let runMs = 0;
  let outputCopyMs = 0;
  let cleanGeneration;
  try {
    setStatus('NFE=1 flow');
    const runStart = performance.now();
    outputs = await runProfiled('flow', {}, sessions.flow, feeds);
    runMs = performance.now() - runStart;
    const copyStart = performance.now();
    cleanGeneration = await tensorToFloat32(outputs.output);
    outputCopyMs = performance.now() - copyStart;
  } finally {
    disposeMap(feeds);
    if (outputs) disposeMap(outputs);
  }
  return {
    cleanGeneration,
    pathCondition,
    timing: {
      setup_ms: setupMs,
      feed_build_ms: feedBuildMs,
      session_run_ms: runMs,
      output_copy_ms: outputCopyMs,
      total_ms: performance.now() - totalStart,
      note: 'one session_run includes CPU upload, WebGPU NFE=1 compute, synchronization, and CPU output download',
    },
  };
}

function interpolatedMotionValue(motion, motionBase, nextMotion, nextMotionBase, feature, alpha) {
  const current = motion[motionBase + feature];
  return nextMotion && alpha > 0
    ? current + (nextMotion[nextMotionBase + feature] - current) * alpha
    : current;
}

function motionFrameToJointsFromRotations({
  motion,
  motionBase,
  nextMotion = null,
  nextMotionBase = 0,
  alpha = 0,
  joints,
  jointBase,
  matrices,
}) {
  const m = state.manifest;
  const parents = m.skeleton.parents;
  const jointCount = parents.length;
  const { mean, std_eps: std } = state.motionStats;
  const rotationStart = m.root_dim + (jointCount - 1) * 3;
  const physical = (feature) => (
    interpolatedMotionValue(
      motion,
      motionBase,
      nextMotion,
      nextMotionBase,
      feature,
      alpha,
    ) * std[feature] + mean[feature]
  );
  for (let joint = 0; joint < jointCount; joint += 1) {
    const feature = rotationStart + joint * 6;
    let xx = physical(feature);
    let xy = physical(feature + 1);
    let xz = physical(feature + 2);
    let inverse = 1 / Math.max(Math.hypot(xx, xy, xz), 1e-8);
    xx *= inverse;
    xy *= inverse;
    xz *= inverse;
    const yRawX = physical(feature + 3);
    const yRawY = physical(feature + 4);
    const yRawZ = physical(feature + 5);
    let zx = xy * yRawZ - xz * yRawY;
    let zy = xz * yRawX - xx * yRawZ;
    let zz = xx * yRawY - xy * yRawX;
    inverse = 1 / Math.max(Math.hypot(zx, zy, zz), 1e-8);
    zx *= inverse;
    zy *= inverse;
    zz *= inverse;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    const matrix = joint * 9;
    matrices[matrix] = xx;
    matrices[matrix + 1] = yx;
    matrices[matrix + 2] = zx;
    matrices[matrix + 3] = xy;
    matrices[matrix + 4] = yy;
    matrices[matrix + 5] = zy;
    matrices[matrix + 6] = xz;
    matrices[matrix + 7] = yz;
    matrices[matrix + 8] = zz;
  }

  joints[jointBase] = physical(0);
  joints[jointBase + 1] = physical(1);
  joints[jointBase + 2] = physical(2);
  for (let joint = 1; joint < jointCount; joint += 1) {
    const parent = parents[joint];
    const rest = CORE_NEUTRAL_JOINTS[joint];
    const parentRest = CORE_NEUTRAL_JOINTS[parent];
    const rx = rest[0] - parentRest[0];
    const ry = rest[1] - parentRest[1];
    const rz = rest[2] - parentRest[2];
    const matrix = parent * 9;
    const parentPosition = jointBase + parent * 3;
    const position = jointBase + joint * 3;
    joints[position] = joints[parentPosition] + matrices[matrix] * rx + matrices[matrix + 1] * ry + matrices[matrix + 2] * rz;
    joints[position + 1] = joints[parentPosition + 1] + matrices[matrix + 3] * rx + matrices[matrix + 4] * ry + matrices[matrix + 5] * rz;
    joints[position + 2] = joints[parentPosition + 2] + matrices[matrix + 6] * rx + matrices[matrix + 7] * ry + matrices[matrix + 8] * rz;
  }
}

function motionToJointsFromRotations(motion, frameCount = state.manifest.generation_frames) {
  const m = state.manifest;
  const jointCount = m.skeleton.parents.length;
  const joints = new Float32Array(frameCount * jointCount * 3);
  const matrices = new Float32Array(jointCount * 9);
  for (let frame = 0; frame < frameCount; frame += 1) {
    motionFrameToJointsFromRotations({
      motion,
      motionBase: frame * m.motion_dim,
      joints,
      jointBase: frame * jointCount * 3,
      matrices,
    });
  }
  return joints;
}

async function finalizeAndDecode({
  cleanGeneration,
  historyHybrid,
  translation,
  initial,
  historyFrames,
  historyDisplayFrames = null,
  protectedRootFrames = [],
}) {
  const totalStart = performance.now();
  const m = state.manifest;
  const finalizeFeedStart = performance.now();
  let finalizer;
  let finalizeFeeds;
  if (initial) {
    finalizer = sessions.finalizeInitial;
    finalizeFeeds = {
      clean_generation: floatTensor(cleanGeneration, [1, 10, m.hybrid_dim], 'fp32'),
      global_translation: floatTensor(translation, [1, 3], 'fp32'),
    };
  } else {
    finalizer = sessions.finalizeContinuation;
    const localHybrid = new Float32Array(11 * m.hybrid_dim);
    localHybrid.set(historyHybrid, 0);
    localHybrid.set(cleanGeneration, m.hybrid_dim);
    finalizeFeeds = {
      local_hybrid: floatTensor(localHybrid, [1, 11, m.hybrid_dim], 'fp32'),
      global_translation: floatTensor(translation, [1, 3], 'fp32'),
    };
  }
  const finalizeFeedBuildMs = performance.now() - finalizeFeedStart;
  let finalized;
  let globalRoot;
  let latent;
  let localRoot;
  let tokenValid;
  let finalizeRunMs = 0;
  let finalizeOutputCopyMs = 0;
  let finalizeDisposeMs = 0;
  try {
    const finalizeRunStart = performance.now();
    finalized = await finalizer.run(finalizeFeeds);
    finalizeRunMs = performance.now() - finalizeRunStart;
    const finalizeCopyStart = performance.now();
    [globalRoot, latent, localRoot, tokenValid] = await Promise.all([
      tensorToFloat32(finalized.global_root),
      tensorToFloat32(finalized.decoder_latent),
      tensorToFloat32(finalized.decoder_local_root),
      tensorToFloat32(finalized.token_valid),
    ]);
    finalizeOutputCopyMs = performance.now() - finalizeCopyStart;
  } finally {
    const finalizeDisposeStart = performance.now();
    disposeMap(finalizeFeeds);
    if (finalized) disposeMap(finalized);
    finalizeDisposeMs = performance.now() - finalizeDisposeStart;
  }

  const decoderFeedStart = performance.now();
  const decoderFeeds = {
    latent_tokens: floatTensor(latent, [1, 11, m.latent_dim]),
    local_root: floatTensor(localRoot, [1, 44, 4]),
    token_valid: floatTensor(tokenValid, [1, 11]),
  };
  const decoderFeedBuildMs = performance.now() - decoderFeedStart;
  let decoded;
  let body;
  let decoderRunMs = 0;
  let decoderOutputCopyMs = 0;
  let decoderDisposeMs = 0;
  try {
    const decoderRunStart = performance.now();
    decoded = await runProfiled('decoder', {}, sessions.decoder, decoderFeeds);
    decoderRunMs = performance.now() - decoderRunStart;
    const decoderCopyStart = performance.now();
    body = await tensorToFloat32(decoded.output);
    decoderOutputCopyMs = performance.now() - decoderCopyStart;
  } finally {
    const decoderDisposeStart = performance.now();
    disposeMap(decoderFeeds);
    if (decoded) disposeMap(decoded);
    decoderDisposeMs = performance.now() - decoderDisposeStart;
  }

  const mergeStart = performance.now();
  const historyLength = initial ? 0 : m.history_frames;
  const fullFrames = initial ? m.generation_frames : m.decode_frames;
  const bodyFrameOffset = initial ? m.history_frames : 0;
  const fullMotion = new Float32Array(fullFrames * m.motion_dim);
  for (let frame = 0; frame < fullFrames; frame += 1) {
    fullMotion.set(globalRoot.subarray(frame * m.root_dim, (frame + 1) * m.root_dim), frame * m.motion_dim);
    const bodyFrame = frame + bodyFrameOffset;
    fullMotion.set(
      body.subarray(bodyFrame * m.body_dim, (bodyFrame + 1) * m.body_dim),
      frame * m.motion_dim + m.root_dim,
    );
  }
  const mergeMs = performance.now() - mergeStart;
  const rawMotion = fullMotion.slice(
    historyLength * m.motion_dim,
    (historyLength + m.generation_frames) * m.motion_dim,
  );
  const jointCount = m.skeleton.parents.length;
  const rotationStart = m.root_dim + (jointCount - 1) * 3;
  const rotationEnd = rotationStart + jointCount * 6;
  const display = postprocessMotionForDisplay({
    fullMotion,
    historyFrames: historyDisplayFrames || historyFrames,
    historyLength,
    generationFrames: m.generation_frames,
    motionDim: m.motion_dim,
    rootDim: m.root_dim,
    rotationStart,
    rotationEnd,
    excludedTailFeatures: m.inertialization?.excluded_tail_features ?? 4,
    rootMean: m.root_stats.mean,
    rootStd: m.root_stats.std_eps,
    mode: state.postprocessMode,
    protectedRootFrames,
  });
  const fkStart = performance.now();
  const joints = motionToJointsFromRotations(display.motion);
  const fkMs = performance.now() - fkStart;

  return {
    motion: rawMotion,
    displayMotion: display.motion,
    joints,
    timing: {
      finalize: {
        feed_build_ms: finalizeFeedBuildMs,
        session_run_ms: finalizeRunMs,
        output_copy_ms: finalizeOutputCopyMs,
        dispose_ms: finalizeDisposeMs,
        total_ms: finalizeFeedBuildMs + finalizeRunMs + finalizeOutputCopyMs + finalizeDisposeMs,
      },
      decoder: {
        feed_build_ms: decoderFeedBuildMs,
        session_run_ms: decoderRunMs,
        output_copy_ms: decoderOutputCopyMs,
        dispose_ms: decoderDisposeMs,
        total_ms: decoderFeedBuildMs + decoderRunMs + decoderOutputCopyMs + decoderDisposeMs,
        note: 'session_run includes default CPU input upload, WebGPU compute, synchronization, and CPU output download',
      },
      postprocess: {
        merge_ms: mergeMs,
        mode: display.mode.id,
        display_filter_ms: display.elapsedMs,
        display_diagnostics: display.diagnostics,
        fk_ms: fkMs,
        total_ms: mergeMs + display.elapsedMs + fkMs,
      },
      total_ms: performance.now() - totalStart,
    },
  };
}

async function generateNext(reason = 'auto') {
  if (!state.ready) return;
  if (state.generating) {
    state.replanRequested = true;
    return;
  }
  state.generating = true;
  generationProfileRows = [];
  state.replanRequested = false;
  state.frameAccumulator = 0;
  state.suppressNextTick = true;
  ui.precision.disabled = true;
  ui.prompt.disabled = true;
  ui.postprocess.disabled = true;
  const epoch = state.epoch;
  const initial = state.motionFrames.length === 0;
  let historyEnd = -1;
  let historyStart = 0;
  let historyFrames = null;
  let historyDisplayFrames = null;
  let historyHybrid = new Float32Array(148);
  let translation = new Float32Array([0, 0, 0]);
  let firstHeading = new Float32Array([1, 0]);
  const timing = {
    history: 0,
    flow: 0,
    decode: 0,
    total: 0,
    history_detail: null,
    flow_detail: null,
    decode_detail: null,
  };
  let completedReport = null;
  const totalStart = performance.now();
  try {
    setStatus(initial ? '生成首段' : `重新规划（${reason}）`);
    if (initial) setLoading(true, '生成首段动作', 'NFE=1 ARDY student 正在浏览器 WebGPU 上运行');
    if (!initial) {
      historyEnd = Math.max(3, Math.min(state.motionFrames.length - 1, state.frameIndex + 1));
      historyStart = historyEnd - 3;
      historyFrames = state.motionFrames.slice(historyStart, historyEnd + 1);
      historyDisplayFrames = state.displayMotionFrames.slice(historyStart, historyEnd + 1);
      const start = performance.now();
      const encoded = await encodeHistory(historyFrames);
      historyHybrid = encoded.historyHybrid;
      translation = encoded.translation;
      firstHeading = encoded.firstHeading;
      timing.history = performance.now() - start;
      timing.history_detail = { ...encoded.timing, wall_total_ms: timing.history };
    }

    const flowStart = performance.now();
    const flowed = await runFlow({ initial, historyHybrid, translation, firstHeading, historyStart });
    timing.flow = performance.now() - flowStart;
    timing.flow_detail = { ...flowed.timing, wall_total_ms: timing.flow };
    const historyLength = initial ? 0 : state.manifest.history_frames;
    const protectedRootFrames = [];
    for (
      let pathFrame = historyLength;
      pathFrame < historyLength + state.manifest.generation_frames;
      pathFrame += 1
    ) {
      if (flowed.pathCondition[pathFrame * 3 + 2] > 0.5) {
        protectedRootFrames.push(pathFrame - historyLength);
      }
    }
    const decodeStart = performance.now();
    const segment = await finalizeAndDecode({
      cleanGeneration: flowed.cleanGeneration,
      historyHybrid,
      translation,
      initial,
      historyFrames,
      historyDisplayFrames,
      protectedRootFrames,
    });
    timing.decode = performance.now() - decodeStart;
    timing.decode_detail = { ...segment.timing, wall_total_ms: timing.decode };
    timing.total = performance.now() - totalStart;
    if (epoch !== state.epoch) return;
    if (![segment.motion, segment.displayMotion, segment.joints]
      .every((array) => array.every(Number.isFinite))) {
      throw new Error('生成结果包含 NaN/Inf');
    }

    if (!initial) {
      state.motionFrames.length = historyEnd + 1;
      state.displayMotionFrames.length = historyEnd + 1;
      state.jointFrames.length = historyEnd + 1;
      state.segments = state.segments
        .filter((item) => item.start <= historyEnd)
        .map((item) => {
          const end = Math.min(item.end, historyEnd);
          return {
            ...item,
            end,
            protectedRootFrames: item.protectedRootFrames.filter(
              (frame) => item.start + frame <= end,
            ),
          };
        });
    }
    const segmentStart = state.motionFrames.length;
    for (let frame = 0; frame < state.manifest.generation_frames; frame += 1) {
      state.motionFrames.push(segment.motion.slice(frame * state.manifest.motion_dim, (frame + 1) * state.manifest.motion_dim));
      state.displayMotionFrames.push(segment.displayMotion.slice(
        frame * state.manifest.motion_dim,
        (frame + 1) * state.manifest.motion_dim,
      ));
      state.jointFrames.push(segment.joints.slice(frame * state.manifest.skeleton.joint_names.length * 3, (frame + 1) * state.manifest.skeleton.joint_names.length * 3));
    }
    state.segments.push({
      start: segmentStart,
      end: segmentStart + state.manifest.generation_frames - 1,
      protectedRootFrames: [...protectedRootFrames],
    });
    state.lastPostprocessMs = timing.decode_detail.postprocess.display_filter_ms;
    state.lastPostprocessFrames = state.manifest.generation_frames;
    state.generationCount += 1;
    const waypointErrors = waypointDiagnostics();
    const gpuProfile = summarizeGpuProfile(generationProfileRows);
    ui.timing.textContent = `${timing.total.toFixed(0)} ms（flow ${timing.flow.toFixed(0)}）`;
    log(`#${state.generationCount} ${reason}: history_end=${historyEnd}, output=${state.motionFrames.length} 帧, total=${timing.total.toFixed(1)} ms, flow=${timing.flow.toFixed(1)} ms, decode=${timing.decode.toFixed(1)} ms`);
    const prompt = selectedPromptEntry();
    if (prompt) log(`  text prompt ${prompt.prompt_id} [${prompt.family}]: ${prompt.text || '(unconditional)'}`);
    if (timing.history_detail) {
      log(`  encoder run=${timing.history_detail.session_run_ms.toFixed(2)} ms, copy=${timing.history_detail.output_copy_ms.toFixed(2)} ms`);
    }
    log(`  NFE=1 flow run=${timing.flow_detail.session_run_ms.toFixed(2)} ms, feeds=${timing.flow_detail.feed_build_ms.toFixed(2)} ms`);
    const displayDiagnostics = timing.decode_detail.postprocess.display_diagnostics;
    log(`  decoder run=${timing.decode_detail.decoder.session_run_ms.toFixed(2)} ms, finalize=${timing.decode_detail.finalize.total_ms.toFixed(2)} ms, display-post=${timing.decode_detail.postprocess.mode}/${timing.decode_detail.postprocess.display_filter_ms.toFixed(2)} ms, FK=${timing.decode_detail.postprocess.fk_ms.toFixed(2)} ms`);
    log(`  display delta: root RMS=${displayDiagnostics.root_rms_deviation_m.toFixed(4)} m, max=${displayDiagnostics.root_max_deviation_m.toFixed(4)} m, endpoint=${displayDiagnostics.root_endpoint_deviation_m.toFixed(6)} m, pose mean=${displayDiagnostics.pose_mean_abs_normalized.toExponential(2)}`);
    if (gpuProfile.enabled) {
      log(`  GPU profiler: ${gpuProfile.record_count} kernels, timestamp sum=${gpuProfile.total_kernel_ms.toFixed(2)} ms`);
    }
    if (waypointErrors.length) {
      const latestError = waypointErrors.at(-1);
      log(`waypoint f${latestError.frame}: raw=(${latestError.actual_x.toFixed(2)}, ${latestError.actual_z.toFixed(2)})/${latestError.error_m.toFixed(3)} m, display=(${latestError.display_x.toFixed(2)}, ${latestError.display_z.toFixed(2)})/${latestError.display_error_m.toFixed(3)} m`);
    }
    setStatus(state.playing ? '播放中' : '已暂停');
    completedReport = {
      event: 'generation_complete',
      reason,
      generation_count: state.generationCount,
      history_end: historyEnd,
      all_finite: true,
      postprocess_mode: state.postprocessMode,
      timing_ms: timing,
      gpu_profile: gpuProfile,
      waypoint_errors: waypointErrors,
    };
    telemetry('report', completedReport);
  } catch (error) {
    console.error(error);
    setStatus('生成失败');
    log(`生成失败: ${error.stack || error.message || error}`);
    telemetry('report', { event: 'generation_error', reason, error: String(error.stack || error.message || error) });
    state.playing = false;
    updatePlayLabel();
  } finally {
    setLoading(false);
    state.generating = false;
    state.frameAccumulator = 0;
    state.lastTick = performance.now();
    state.suppressNextTick = true;
    ui.precision.disabled = true;
    ui.prompt.disabled = !state.ready || !state.promptMetadata;
    ui.postprocess.disabled = !state.ready || state.benchmarking;
    updateUi();
    if (state.replanRequested && state.ready) {
      state.replanRequested = false;
      queueMicrotask(() => generateNext('queued'));
    }
  }
  return completedReport;
}

function updatePlayLabel() {
  ui.play.textContent = state.playing ? '暂停' : '开始';
}

function updateUi() {
  ui.frame.textContent = `${state.motionFrames.length ? state.frameIndex + 1 : 0} / ${state.motionFrames.length}`;
  const latest = state.waypoints.at(-1);
  if (latest) ui.target.textContent = `(${latest.x.toFixed(1)}, ${latest.z.toFixed(1)}) @ ${latest.frame}`;
  else ui.target.textContent = '未设置';
  const mode = currentPostprocessMode();
  ui.postprocessValue.textContent = state.lastPostprocessMs > 0
    ? `${mode.label} · ${state.lastPostprocessMs.toFixed(2)} ms/${state.lastPostprocessFrames}帧`
    : mode.label;
}

function restart() {
  state.epoch += 1;
  state.playing = false;
  state.frameIndex = 0;
  state.frameAccumulator = 0;
  state.motionFrames = [];
  state.displayMotionFrames = [];
  state.jointFrames = [];
  state.segments = [];
  state.generationCount = 0;
  state.lastPostprocessMs = 0;
  state.lastPostprocessFrames = 0;
  state.replanRequested = false;
  setStatus('已清空');
  updatePlayLabel();
  updateUi();
}

function rebuildDisplayMotion() {
  const totalStart = performance.now();
  const m = state.manifest;
  if (!state.motionFrames.length) {
    state.displayMotionFrames = [];
    state.jointFrames = [];
    return { frames: 0, filter_ms: 0, fk_ms: 0, total_ms: 0 };
  }
  const jointCount = m.skeleton.parents.length;
  const rotationStart = m.root_dim + (jointCount - 1) * 3;
  const rotationEnd = rotationStart + jointCount * 6;
  const displayFrames = new Array(state.motionFrames.length);
  const jointFrames = new Array(state.motionFrames.length);
  const segments = state.segments.length
    ? state.segments
    : [{ start: 0, end: state.motionFrames.length - 1, protectedRootFrames: [] }];
  let processedFrames = 0;
  let filterMs = 0;
  let fkMs = 0;
  for (const item of segments) {
    const start = Math.max(0, item.start);
    const end = Math.min(item.end, state.motionFrames.length - 1);
    if (end < start) continue;
    if (start !== processedFrames) throw new Error(`显示段不连续: ${processedFrames} -> ${start}`);
    const generationFrames = end - start + 1;
    const historyLength = Math.min(m.history_frames, start);
    const historyStart = start - historyLength;
    const fullMotion = new Float32Array((historyLength + generationFrames) * m.motion_dim);
    for (let frame = historyStart; frame <= end; frame += 1) {
      fullMotion.set(state.motionFrames[frame], (frame - historyStart) * m.motion_dim);
    }
    const historyDisplayFrames = historyLength
      ? displayFrames.slice(historyStart, start)
      : null;
    const display = postprocessMotionForDisplay({
      fullMotion,
      historyFrames: historyDisplayFrames,
      historyLength,
      generationFrames,
      motionDim: m.motion_dim,
      rootDim: m.root_dim,
      rotationStart,
      rotationEnd,
      excludedTailFeatures: m.inertialization?.excluded_tail_features ?? 4,
      rootMean: m.root_stats.mean,
      rootStd: m.root_stats.std_eps,
      mode: state.postprocessMode,
      protectedRootFrames: item.protectedRootFrames.filter((frame) => frame < generationFrames),
    });
    filterMs += display.elapsedMs;
    const fkStart = performance.now();
    const joints = motionToJointsFromRotations(display.motion, generationFrames);
    fkMs += performance.now() - fkStart;
    for (let frame = 0; frame < generationFrames; frame += 1) {
      displayFrames[start + frame] = display.motion.slice(
        frame * m.motion_dim,
        (frame + 1) * m.motion_dim,
      );
      jointFrames[start + frame] = joints.slice(
        frame * jointCount * 3,
        (frame + 1) * jointCount * 3,
      );
    }
    processedFrames = end + 1;
  }
  if (processedFrames !== state.motionFrames.length) {
    throw new Error(`显示段只覆盖 ${processedFrames}/${state.motionFrames.length} 帧`);
  }
  state.displayMotionFrames = displayFrames;
  state.jointFrames = jointFrames;
  return {
    frames: processedFrames,
    filter_ms: filterMs,
    fk_ms: fkMs,
    total_ms: performance.now() - totalStart,
  };
}

async function selectPostprocessMode(modeValue, { rebuild = true } = {}) {
  const next = resolvePostprocessMode(modeValue);
  const previous = currentPostprocessMode();
  if (next.id === previous.id) {
    updatePostprocessDescription();
    return;
  }
  const hadMotion = state.motionFrames.length > 0;
  state.postprocessMode = next.id;
  let summary = null;
  ui.postprocess.disabled = true;
  try {
    if (hadMotion && rebuild) summary = rebuildDisplayMotion();
  } catch (error) {
    state.postprocessMode = previous.id;
    rebuildDisplayMotion();
    updatePostprocessDescription();
    throw error;
  } finally {
    ui.postprocess.disabled = !state.ready || state.generating || state.benchmarking;
  }
  localStorage.setItem('ardy-infinite-demo-postprocess', next.id);
  const url = new URL(window.location.href);
  if (next.id === 'raw') url.searchParams.delete('post');
  else url.searchParams.set('post', next.id);
  window.history.replaceState(null, '', url);
  updatePostprocessDescription();
  if (summary) {
    state.lastPostprocessMs = summary.filter_ms;
    state.lastPostprocessFrames = summary.frames;
  } else {
    state.lastPostprocessMs = 0;
    state.lastPostprocessFrames = 0;
  }
  updateUi();
  log(`显示后处理：${previous.label} -> ${next.label}。${summary ? `同一原始序列即时重算 ${summary.frames} 帧：filter=${summary.filter_ms.toFixed(2)} ms, FK=${summary.fk_ms.toFixed(2)} ms, total=${summary.total_ms.toFixed(2)} ms。` : ''}`);
  telemetry('report', {
    event: 'postprocess_changed',
    previous_postprocess_mode: previous.id,
    rebuild: Boolean(summary),
    rebuild_timing_ms: summary,
  });
}

function clearWaypoints() {
  const count = state.waypoints.length;
  state.waypoints = [];
  log(`已清空 ${count} 个 root waypoint；已生成动作保留。`);
  updateUi();
  telemetry('report', { event: 'waypoints_cleared', cleared_count: count });
}

function togglePlay() {
  state.playing = !state.playing;
  state.lastTick = performance.now();
  state.frameAccumulator = 0;
  updatePlayLabel();
  if (state.playing && !state.motionFrames.length) generateNext('start');
  else setStatus(state.playing ? '播放中' : '已暂停');
}

function maybeAutoReplan() {
  if (!state.playing || state.generating || !state.motionFrames.length) return;
  const remaining = state.motionFrames.length - 1 - state.frameIndex;
  if (remaining <= 4) generateNext('auto');
}

function resizeCanvas() {
  const rect = ui.canvas.getBoundingClientRect();
  state.dpr = Math.min(devicePixelRatio || 1, 2);
  state.width = rect.width;
  state.height = rect.height;
  ui.canvas.width = Math.round(rect.width * state.dpr);
  ui.canvas.height = Math.round(rect.height * state.dpr);
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  const timelineRect = ui.timeline.getBoundingClientRect();
  state.timelineWidth = timelineRect.width;
  state.timelineHeight = timelineRect.height;
  ui.timeline.width = Math.round(timelineRect.width * state.dpr);
  ui.timeline.height = Math.round(timelineRect.height * state.dpr);
  timelineCtx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
}

function formatTimelineTime(frame) {
  const seconds = frame / state.manifest.fps;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`;
}

function timelineRange() {
  return { start: Math.max(0, state.frameIndex - 20), end: state.frameIndex + 200 };
}

function drawTimeline() {
  if (!state.manifest) return;
  const width = state.timelineWidth;
  const height = state.timelineHeight;
  const range = timelineRange();
  const span = range.end - range.start;
  const xForFrame = (frame) => ((frame - range.start) / span) * width;
  timelineCtx.clearRect(0, 0, width, height);

  timelineCtx.fillStyle = 'rgba(27, 39, 52, .9)';
  timelineCtx.fillRect(0, 27, width, 14);
  if (state.motionFrames.length) {
    const generatedStart = Math.max(range.start, 0);
    const generatedEnd = Math.min(range.end, state.motionFrames.length - 1);
    if (generatedEnd >= generatedStart) {
      timelineCtx.fillStyle = 'rgba(57, 190, 156, .55)';
      timelineCtx.fillRect(xForFrame(generatedStart), 28, Math.max(2, xForFrame(generatedEnd) - xForFrame(generatedStart)), 12);
    }
  }

  const firstTick = Math.ceil(range.start / 10) * 10;
  timelineCtx.font = '9px ui-monospace, monospace';
  timelineCtx.textAlign = 'center';
  for (let frame = firstTick; frame <= range.end; frame += 10) {
    const x = xForFrame(frame);
    const major = frame % 20 === 0;
    timelineCtx.strokeStyle = major ? 'rgba(173, 192, 211, .45)' : 'rgba(126, 145, 165, .22)';
    timelineCtx.beginPath(); timelineCtx.moveTo(x, major ? 7 : 15); timelineCtx.lineTo(x, 45); timelineCtx.stroke();
    if (major) {
      timelineCtx.fillStyle = '#8192a6';
      timelineCtx.fillText(`${(frame / state.manifest.fps).toFixed(0)}s`, x, 8);
      timelineCtx.fillStyle = '#5f7185';
      timelineCtx.fillText(`${frame}`, x, 55);
    }
  }

  for (const waypoint of state.waypoints) {
    if (waypoint.frame < range.start || waypoint.frame > range.end) continue;
    const x = xForFrame(waypoint.frame);
    timelineCtx.fillStyle = '#ffad55';
    timelineCtx.beginPath();
    timelineCtx.moveTo(x, 19); timelineCtx.lineTo(x + 6, 25); timelineCtx.lineTo(x, 31); timelineCtx.lineTo(x - 6, 25);
    timelineCtx.closePath(); timelineCtx.fill();
  }

  const currentX = xForFrame(state.frameIndex);
  timelineCtx.strokeStyle = '#f0f6ff';
  timelineCtx.lineWidth = 2;
  timelineCtx.beginPath(); timelineCtx.moveTo(currentX, 4); timelineCtx.lineTo(currentX, height); timelineCtx.stroke();
  timelineCtx.fillStyle = '#f0f6ff';
  timelineCtx.beginPath(); timelineCtx.moveTo(currentX - 5, 2); timelineCtx.lineTo(currentX + 5, 2); timelineCtx.lineTo(currentX, 8); timelineCtx.closePath(); timelineCtx.fill();
  timelineCtx.lineWidth = 1;

  ui.timelineTime.textContent = formatTimelineTime(state.frameIndex);
  ui.timelineFrame.textContent = `frame ${state.frameIndex} / ${Math.max(0, state.motionFrames.length - 1)}`;
}

function onTimelineClick(event) {
  if (!state.manifest || !state.motionFrames.length || state.generating || state.benchmarking) return;
  const rect = ui.timeline.getBoundingClientRect();
  const range = timelineRange();
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1)));
  const requested = Math.round(range.start + fraction * (range.end - range.start));
  state.frameIndex = Math.max(0, Math.min(state.motionFrames.length - 1, requested));
  state.playing = false;
  state.frameAccumulator = 0;
  state.lastTick = performance.now();
  state.suppressNextTick = true;
  updatePlayLabel();
  setStatus('时间轴定位');
  updateUi();
}

function viewScale() {
  return Math.max(34, Math.min(64, state.width / 22));
}

function project(x, y, z) {
  const scale = viewScale();
  return {
    x: state.width * 0.54 + ((x - state.cameraX) - (z - state.cameraZ)) * scale,
    y: state.height * 0.64 + ((x - state.cameraX) + (z - state.cameraZ)) * scale * 0.34 - y * scale,
  };
}

function unprojectGround(screenX, screenY) {
  const scale = viewScale();
  const a = (screenX - state.width * 0.54) / scale;
  const b = (screenY - state.height * 0.64) / (scale * 0.34);
  return { x: state.cameraX + (a + b) / 2, z: state.cameraZ + (b - a) / 2 };
}

function drawGrid() {
  const centerX = Math.round(state.cameraX);
  const centerZ = Math.round(state.cameraZ);
  ctx.lineWidth = 1;
  for (let offset = -14; offset <= 14; offset += 1) {
    const alpha = offset % 5 === 0 ? 0.18 : 0.075;
    ctx.strokeStyle = `rgba(130, 158, 181, ${alpha})`;
    let a = project(centerX - 14, 0, centerZ + offset);
    let b = project(centerX + 14, 0, centerZ + offset);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    a = project(centerX + offset, 0, centerZ - 14);
    b = project(centerX + offset, 0, centerZ + 14);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
}

function drawTrajectory() {
  const frames = state.displayMotionFrames.length
    ? state.displayMotionFrames
    : state.motionFrames;
  if (frames.length < 2) return;
  const start = Math.max(0, state.frameIndex - 100);
  ctx.beginPath();
  for (let index = start; index < frames.length; index += 1) {
    const frame = frames[index];
    const mean = state.manifest.root_stats.mean;
    const std = state.manifest.root_stats.std_eps;
    const point = project(frame[0] * std[0] + mean[0], 0.012, frame[2] * std[2] + mean[2]);
    if (index === start) ctx.moveTo(point.x, point.y); else ctx.lineTo(point.x, point.y);
  }
  ctx.strokeStyle = 'rgba(61, 200, 165, .65)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTarget() {
  if (!state.waypoints.length) return;
  const latest = state.waypoints.at(-1);
  for (const waypoint of state.waypoints) {
    const p = project(waypoint.x, 0.02, waypoint.z);
    const isLatest = waypoint === latest;
    const radius = isLatest ? 7 + Math.sin(performance.now() * 0.006) * 2 : 6;
    const reached = waypoint.frame < state.frameIndex;
    ctx.strokeStyle = reached ? 'rgba(255, 173, 85, .35)' : '#ffad55';
    ctx.fillStyle = reached ? 'rgba(255, 173, 85, .06)' : 'rgba(255, 173, 85, .18)';
    ctx.lineWidth = isLatest ? 2 : 1.4;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p.x - 10, p.y); ctx.lineTo(p.x + 10, p.y); ctx.moveTo(p.x, p.y - 10); ctx.lineTo(p.x, p.y + 10); ctx.stroke();
    ctx.fillStyle = reached ? 'rgba(255, 190, 120, .45)' : '#ffc17c';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`f${waypoint.frame}`, p.x + 10, p.y - 10);
  }
}

function drawSkeleton() {
  let joints = state.jointFrames[state.frameIndex];
  if (!joints) return;
  const alpha = displayInterpolationAlpha();
  if (alpha > 0) {
    const currentMotion = state.displayMotionFrames[state.frameIndex];
    const nextMotion = state.displayMotionFrames[state.frameIndex + 1];
    motionFrameToJointsFromRotations({
      motion: currentMotion,
      motionBase: 0,
      nextMotion,
      nextMotionBase: 0,
      alpha,
      joints: interpolatedJointScratch,
      jointBase: 0,
      matrices: interpolatedMatrixScratch,
    });
    joints = interpolatedJointScratch;
  }
  const parents = state.manifest.skeleton.parents;
  const points = new Array(parents.length);
  for (let joint = 0; joint < parents.length; joint += 1) {
    const offset = joint * 3;
    points[joint] = project(joints[offset], joints[offset + 1], joints[offset + 2]);
  }
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(211, 232, 250, .92)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let joint = 0; joint < parents.length; joint += 1) {
    const parent = parents[joint];
    if (parent < 0) continue;
    ctx.moveTo(points[parent].x, points[parent].y);
    ctx.lineTo(points[joint].x, points[joint].y);
  }
  ctx.stroke();
  ctx.fillStyle = '#73e3c4';
  for (const point of points) {
    ctx.beginPath(); ctx.arc(point.x, point.y, 2.4, 0, Math.PI * 2); ctx.fill();
  }
  const root = points[state.manifest.skeleton.root_index];
  ctx.fillStyle = '#eef7ff';
  ctx.beginPath(); ctx.arc(root.x, root.y, 4, 0, Math.PI * 2); ctx.fill();
}

function render(now) {
  let dt = Math.min(0.1, (now - state.lastTick) / 1000);
  state.lastTick = now;
  if (state.suppressNextTick) {
    dt = 0;
    state.suppressNextTick = false;
  }
  if (state.playing && !state.generating && state.motionFrames.length) {
    state.frameAccumulator += dt * state.manifest.fps;
    while (state.frameAccumulator >= 1 && !state.generating && state.frameIndex < state.motionFrames.length - 1) {
      state.frameIndex += 1;
      state.frameAccumulator -= 1;
      maybeAutoReplan();
    }
  }
  const root = currentDisplayRootPosition();
  const cameraBlend = 1 - Math.exp(-dt * 4);
  state.cameraX += (root.x - state.cameraX) * cameraBlend;
  state.cameraZ += (root.z - state.cameraZ) * cameraBlend;

  ctx.clearRect(0, 0, state.width, state.height);
  drawGrid();
  drawTrajectory();
  drawTarget();
  drawSkeleton();
  drawTimeline();
  updateUi();
  requestAnimationFrame(render);
}

function onCanvasClick(event) {
  if (!state.ready || state.benchmarking) return;
  const rect = ui.canvas.getBoundingClientRect();
  const target = unprojectGround(event.clientX - rect.left, event.clientY - rect.top);
  const currentFrame = state.motionFrames.length ? state.frameIndex : 0;
  const waypoint = {
    x: target.x,
    z: target.z,
    frame: currentFrame + state.manifest.waypoint_interval_frames,
  };
  // RootKeyframe2DSet updates the existing keyframe when two clicks resolve to
  // the same absolute frame; otherwise constraints accumulate on the track.
  state.waypoints = state.waypoints.filter((item) => item.frame !== waypoint.frame);
  state.waypoints.push(waypoint);
  state.waypoints.sort((left, right) => left.frame - right.frame);
  log(`sparse waypoint: (${target.x.toFixed(2)}, ${target.z.toFixed(2)}) @ frame ${waypoint.frame}`);
  telemetry('report', { event: 'waypoint', target: waypoint, dense_path: false });
  updateUi();
  if (state.motionFrames.length) generateNext('waypoint');
}

function percentile(sorted, probability) {
  if (!sorted.length) return null;
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function summarizeSeries(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return { count: 0 };
  return {
    count: sorted.length,
    min: sorted[0],
    p50: percentile(sorted, 0.50),
    p90: percentile(sorted, 0.90),
    p95: percentile(sorted, 0.95),
    max: sorted.at(-1),
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  };
}

function validationMetrics(actual, reference) {
  if (actual.length !== reference.length) {
    throw new Error(`validation shape mismatch: ${actual.length} != ${reference.length}`);
  }
  const errors = new Array(actual.length);
  let allFinite = true;
  let maxAbs = 0;
  let sumAbs = 0;
  let dot = 0;
  let actualNorm = 0;
  let referenceNorm = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const left = Number(actual[index]);
    const right = Number(reference[index]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) allFinite = false;
    const error = Math.abs(left - right);
    errors[index] = error;
    maxAbs = Math.max(maxAbs, error);
    sumAbs += error;
    dot += left * right;
    actualNorm += left * left;
    referenceNorm += right * right;
  }
  errors.sort((left, right) => left - right);
  const denominator = Math.sqrt(actualNorm * referenceNorm);
  return {
    all_finite: allFinite,
    max_abs_error: maxAbs,
    mean_abs_error: errors.length ? sumAbs / errors.length : 0,
    p95_abs_error: percentile(errors, 0.95) ?? 0,
    p99_abs_error: percentile(errors, 0.99) ?? 0,
    cosine_similarity: denominator ? dot / denominator : 1,
  };
}

function validationFsqBinMatch(actual, reference) {
  if (actual.length !== reference.length || actual.length !== 10 * state.manifest.hybrid_dim) {
    throw new Error('validation clean endpoint shape mismatch');
  }
  const stats = state.manifest.post_quantization_stats;
  const halfWidth = stats.levels / 2;
  let matches = 0;
  let count = 0;
  for (let token = 0; token < 10; token += 1) {
    for (let feature = 0; feature < state.manifest.latent_dim; feature += 1) {
      const index = token * state.manifest.hybrid_dim + 20 + feature;
      const toBin = (value) => roundTiesToEven(
        Math.max(-1, Math.min(1, value * stats.std_eps[feature] + stats.mean[feature])) * halfWidth,
      );
      if (toBin(actual[index]) === toBin(reference[index])) matches += 1;
      count += 1;
    }
  }
  return { matches, count, bin_match_fraction: matches / count };
}

async function runValidationStage(spec, initial, conditions) {
  const m = state.manifest;
  let historyFrames = null;
  let historyHybrid = new Float32Array(m.hybrid_dim);
  let translation = new Float32Array([0, 0, 0]);
  let firstHeading = new Float32Array([1, 0]);
  let historyMetrics = null;
  if (!initial) {
    const flatHistory = Float32Array.from(spec.history);
    historyFrames = Array.from({ length: m.history_frames }, (_, frame) => (
      flatHistory.subarray(frame * m.motion_dim, (frame + 1) * m.motion_dim)
    ));
    const encoded = await encodeHistory(historyFrames);
    historyHybrid = encoded.historyHybrid;
    translation = encoded.translation;
    firstHeading = encoded.firstHeading;
    historyMetrics = validationMetrics(
      historyHybrid,
      Float32Array.from(spec.expected.history_hybrid),
    );
  }
  const feeds = {
    noise: floatTensor(Float32Array.from(spec.noise), [1, 10, m.hybrid_dim]),
    history_hybrid: floatTensor(historyHybrid, [1, 1, m.hybrid_dim]),
    path_condition: floatTensor(Float32Array.from(spec.path_condition), [1, m.path_frames, 3]),
    first_heading: floatTensor(firstHeading, [1, 2]),
    has_history: floatTensor(new Float32Array([initial ? 0 : 1]), [1, 1]),
  };
  if (m.text_conditioning_enabled) {
    feeds.text_feature = floatTensor(
      Float32Array.from(conditions.text_feature),
      [1, m.text_conditioning.feature_dim],
    );
  }
  if (m.heading_conditioning_enabled) {
    feeds.heading_condition = floatTensor(
      Float32Array.from(conditions.heading_condition),
      [1, m.path_frames, 3],
    );
  }
  let outputs;
  let clean;
  try {
    outputs = await runProfiled('flow', { validation: true }, sessions.flow, feeds);
    clean = await tensorToFloat32(outputs.output);
  } finally {
    disposeMap(feeds);
    if (outputs) disposeMap(outputs);
  }
  const decoded = await finalizeAndDecode({
    cleanGeneration: clean,
    historyHybrid,
    translation,
    initial,
    historyFrames,
  });
  const expectedClean = Float32Array.from(spec.expected.clean);
  const expectedMotion = Float32Array.from(spec.expected.motion);
  return {
    history_hybrid: historyMetrics,
    endpoint: validationMetrics(clean, expectedClean),
    fsq: validationFsqBinMatch(clean, expectedClean),
    explicit: validationMetrics(decoded.motion, expectedMotion),
  };
}

async function runBrowserValidation() {
  state.benchmarking = true;
  ui.play.disabled = true;
  ui.restart.disabled = true;
  ui.clearWaypoints.disabled = true;
  ui.postprocess.disabled = true;
  setStatus('WebGPU 数值对拍');
  try {
    const response = await fetch(demoAssetUrl('browser_validation_case.json'), { cache: 'no-cache' });
    if (!response.ok) throw new Error(`browser validation case HTTP ${response.status}`);
    const golden = await response.json();
    if (golden.schema !== 'ardy_webgpu_e2e_golden_v1') throw new Error('browser validation schema mismatch');
    if (golden.model_release !== state.manifest.model_release) throw new Error('browser validation release mismatch');
    if (golden.precision !== state.precision || golden.nfe !== state.manifest.nfe) {
      throw new Error('browser validation precision/NFE mismatch');
    }
    const initial = await runValidationStage(golden.initial, true, golden.conditions);
    const continuation = await runValidationStage(golden.continuation, false, golden.conditions);
    const thresholds = golden.thresholds;
    const stagePassed = (row, requireHistory) => (
      row.endpoint.all_finite
      && row.explicit.all_finite
      && row.endpoint.max_abs_error <= thresholds.endpoint_max_abs
      && row.explicit.mean_abs_error <= thresholds.explicit_mean_abs
      && row.explicit.p99_abs_error <= thresholds.explicit_p99_abs
      && row.explicit.cosine_similarity >= thresholds.explicit_cosine
      && row.fsq.bin_match_fraction >= thresholds.fsq_bin_match_fraction
      && (!requireHistory || (
        row.history_hybrid.all_finite
        && row.history_hybrid.mean_abs_error <= thresholds.history_mean_abs
        && row.history_hybrid.cosine_similarity >= thresholds.history_cosine
      ))
    );
    const result = {
      event: 'validation_complete',
      passed: stagePassed(initial, false) && stagePassed(continuation, true),
      model_release: golden.model_release,
      precision: golden.precision,
      nfe: golden.nfe,
      thresholds,
      initial,
      continuation,
    };
    window.__ardyValidation = result;
    log(`WebGPU 整链路数值对拍：${result.passed ? '通过' : '失败'}；initial/continuation endpoint max=${initial.endpoint.max_abs_error.toExponential(3)}/${continuation.endpoint.max_abs_error.toExponential(3)}`);
    log(`  motion mean=${initial.explicit.mean_abs_error.toExponential(3)}/${continuation.explicit.mean_abs_error.toExponential(3)}，FSQ bin=${(initial.fsq.bin_match_fraction * 100).toFixed(2)}%/${(continuation.fsq.bin_match_fraction * 100).toFixed(2)}%`);
    telemetry('report', result);
    setStatus(result.passed ? '数值对拍通过' : '数值对拍失败');
    return result;
  } finally {
    state.benchmarking = false;
    ui.play.disabled = !state.ready;
    ui.restart.disabled = !state.ready;
    ui.clearWaypoints.disabled = !state.ready;
    ui.postprocess.disabled = !state.ready;
  }
}

async function runPrecisionProbe() {
  state.benchmarking = true;
  ui.play.disabled = true;
  ui.restart.disabled = true;
  ui.clearWaypoints.disabled = true;
  ui.postprocess.disabled = true;
  setStatus('WebGPU 逐层精度探针');
  const cases = {};
  try {
    const response = await fetch('./infinite_demo/webgpu_precision_probe_case.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`precision probe case HTTP ${response.status}`);
    const golden = await response.json();
    if (golden.schema !== 'ardy_webgpu_precision_probe_v1') throw new Error('precision probe schema mismatch');
    if (golden.model_release !== state.manifest.model_release || golden.precision !== state.precision) {
      throw new Error('precision probe release/precision mismatch');
    }
    for (const moduleName of ['encoder', 'flow', 'decoder']) {
      const spec = golden.cases[moduleName];
      setLoading(true, `加载 ${moduleName} 逐层探针`, `${(spec.size_bytes / 1048576).toFixed(1)} MiB；仅本次诊断`);
      const session = await createWebGpuSession(spec.model_url);
      const feeds = {};
      let outputs;
      try {
        for (const [name, input] of Object.entries(spec.inputs)) {
          feeds[name] = floatTensor(Float32Array.from(input.values), input.shape, 'fp16');
        }
        setLoading(true, `运行 ${moduleName} 逐层探针`, `${spec.outputs.length} 个观测点`);
        outputs = await session.run(feeds);
        const rows = [];
        for (const output of spec.outputs) {
          const actual = await tensorToFloat32(outputs[output.tensor_name]);
          const reference = Float32Array.from(output.reference.values);
          rows.push({
            label: output.label,
            tensor_name: output.tensor_name,
            shape: output.reference.shape,
            ...validationMetrics(actual, reference),
          });
        }
        cases[moduleName] = rows;
        const first = rows[0];
        const last = rows.at(-1);
        log(`precision probe ${moduleName}: ${first.label} mean=${first.mean_abs_error.toExponential(3)} -> ${last.label} mean=${last.mean_abs_error.toExponential(3)}, max=${last.max_abs_error.toExponential(3)}`);
      } finally {
        disposeMap(feeds);
        if (outputs) disposeMap(outputs);
        await session.release();
      }
      await yieldToAnimationFrame();
    }
    const result = {
      event: 'precision_probe_complete',
      passed: Object.values(cases).every((rows) => rows.every((row) => row.all_finite)),
      model_release: golden.model_release,
      precision: golden.precision,
      cases,
    };
    window.__ardyPrecisionProbe = result;
    telemetry('report', result);
    setStatus(result.passed ? '逐层精度探针完成' : '逐层精度探针出现非有限值');
    return result;
  } finally {
    setLoading(false);
    state.benchmarking = false;
    ui.play.disabled = !state.ready;
    ui.restart.disabled = !state.ready;
    ui.clearWaypoints.disabled = !state.ready;
    ui.postprocess.disabled = !state.ready;
  }
}

function summarizeBenchmark(reports) {
  const series = (getter) => summarizeSeries(reports.map(getter));
  const summary = {
    total_ms: series((row) => row.timing_ms.total),
    history_encoder_wall_ms: series((row) => row.timing_ms.history),
    history_encoder_session_run_ms: series((row) => row.timing_ms.history_detail?.session_run_ms),
    flow_wall_ms: series((row) => row.timing_ms.flow),
    flow_session_run_ms: series((row) => row.timing_ms.flow_detail?.session_run_ms),
    finalize_ms: series((row) => row.timing_ms.decode_detail?.finalize.total_ms),
    motion_decoder_session_run_ms: series((row) => row.timing_ms.decode_detail?.decoder.session_run_ms),
    decoder_postprocess_merge_ms: series((row) => row.timing_ms.decode_detail?.postprocess.merge_ms),
    display_postprocess_ms: series((row) => row.timing_ms.decode_detail?.postprocess.display_filter_ms),
    decoder_postprocess_fk_ms: series((row) => row.timing_ms.decode_detail?.postprocess.fk_ms),
    decode_pipeline_wall_ms: series((row) => row.timing_ms.decode),
  };
  return summary;
}

async function runAutoBenchmark() {
  restart();
  state.benchmarking = true;
  state.playing = false;
  ui.play.disabled = true;
  ui.restart.disabled = true;
  ui.clearWaypoints.disabled = true;
  ui.precision.disabled = true;
  ui.postprocess.disabled = true;
  const warmupRuns = gpuProfilingEnabled ? 0 : 3;
  const timedRuns = gpuProfilingEnabled ? 1 : 20;
  const reports = [];
  log(`详细测速开始：profile=${gpuProfilingEnabled}, warmup=${warmupRuns}, timed=${timedRuns}`);
  telemetry('report', {
    event: 'benchmark_start',
    profiling: gpuProfilingEnabled,
    warmup_runs: warmupRuns,
    timed_runs: timedRuns,
  });
  try {
    const initial = await generateNext('benchmark_initial_warmup');
    if (!initial) throw new Error('初始段测速生成失败');
    for (let index = 0; index < warmupRuns + timedRuns; index += 1) {
      // Always regenerate from the same first four frames so sequence length and
      // encoder/decoder shapes remain identical across repetitions.
      state.frameIndex = 2;
      const warmup = index < warmupRuns;
      setStatus(warmup ? `预热 ${index + 1} / ${warmupRuns}` : `测速 ${index - warmupRuns + 1} / ${timedRuns}`);
      const report = await generateNext(warmup ? `benchmark_warmup_${index + 1}` : `benchmark_timed_${index - warmupRuns + 1}`);
      if (!report) throw new Error(`测速轮次 ${index + 1} 失败`);
      if (!warmup) reports.push(report);
    }
    const summary = summarizeBenchmark(reports);
    const result = {
      event: 'benchmark_complete',
      profiling: gpuProfilingEnabled,
      warmup_runs: warmupRuns,
      timed_runs: timedRuns,
      postprocess_mode: state.postprocessMode,
      summary_ms: summary,
      gpu_profile: reports.length === 1 ? reports[0].gpu_profile : undefined,
    };
    window.__ardyBenchmark = { reports, result };
    log(`测速完成：total p50=${summary.total_ms.p50.toFixed(2)} ms, p95=${summary.total_ms.p95.toFixed(2)} ms`);
    log(`  encoder run p50=${summary.history_encoder_session_run_ms.p50.toFixed(2)} ms, decoder run p50=${summary.motion_decoder_session_run_ms.p50.toFixed(2)} ms`);
    log(`  NFE=1 flow p50=${summary.flow_session_run_ms.p50.toFixed(2)} ms, FK p50=${summary.decoder_postprocess_fk_ms.p50.toFixed(2)} ms`);
    log(`  display-post ${state.postprocessMode} p50=${summary.display_postprocess_ms.p50.toFixed(3)} ms / 40 帧`);
    telemetry('report', result);
    setStatus('测速完成');
  } finally {
    state.benchmarking = false;
    ui.play.disabled = !state.ready;
    ui.restart.disabled = !state.ready;
    ui.clearWaypoints.disabled = !state.ready;
    ui.precision.disabled = true;
    ui.postprocess.disabled = !state.ready;
    updatePlayLabel();
  }
}

async function executeDemoCommand(job) {
  const action = job.action;
  const command = job.command || {};
  let validation;
  let precisionProbe;
  log(`收到服务器命令 ${job.job_id}: ${action}`);
  if (action === 'restart') {
    restart();
  } else if (action === 'pause') {
    state.playing = false;
    state.frameAccumulator = 0;
    state.lastTick = performance.now();
    updatePlayLabel();
    setStatus('已暂停');
  } else if (action === 'start') {
    state.playing = true;
    state.frameAccumulator = 0;
    state.lastTick = performance.now();
    updatePlayLabel();
    if (!state.motionFrames.length) await generateNext('remote_start');
    else setStatus('播放中');
  } else if (action === 'waypoint') {
    const currentFrame = state.motionFrames.length ? state.frameIndex : 0;
    const waypoint = {
      x: Number(command.x),
      z: Number(command.z),
      frame: currentFrame + Number(command.frame_offset || state.manifest.waypoint_interval_frames),
    };
    if (![waypoint.x, waypoint.z, waypoint.frame].every(Number.isFinite)) {
      throw new Error('服务器 waypoint 非 finite');
    }
    state.waypoints = state.waypoints.filter((item) => item.frame !== waypoint.frame);
    state.waypoints.push(waypoint);
    state.waypoints.sort((left, right) => left.frame - right.frame);
    log(`remote sparse waypoint: (${waypoint.x.toFixed(2)}, ${waypoint.z.toFixed(2)}) @ frame ${waypoint.frame}`);
    updateUi();
    if (state.motionFrames.length) await generateNext('remote_waypoint');
  } else if (action === 'clear_waypoints') {
    clearWaypoints();
  } else if (action === 'benchmark') {
    await runAutoBenchmark();
  } else if (action === 'validate') {
    validation = await runBrowserValidation();
  } else if (action === 'precision_probe') {
    precisionProbe = await runPrecisionProbe();
  } else if (action === 'prompt') {
    const index = state.promptMetadata?.entries?.findIndex(
      (entry) => entry.prompt_id === Number(command.prompt_id),
    ) ?? -1;
    if (index < 0) throw new Error(`prompt id 不在浏览器 bundle: ${command.prompt_id}`);
    state.selectedPromptIndex = index;
    ui.prompt.value = String(index);
    const prompt = selectedPromptEntry();
    log(`remote prompt ${prompt.prompt_id}: ${prompt.text || '(unconditional)'}`);
    if (state.motionFrames.length) await generateNext('remote_prompt');
  } else if (action === 'postprocess') {
    const mode = resolvePostprocessMode(command.mode);
    if (mode.id !== command.mode) throw new Error(`未知显示后处理模式: ${command.mode}`);
    await selectPostprocessMode(mode.id);
  } else if (action === 'reload') {
    // Give the command completion report time to reach the server first.
    setTimeout(() => window.location.reload(), 500);
  } else {
    throw new Error(`未知服务器命令: ${action}`);
  }
  return {
    passed: action === 'validate'
      ? validation.passed
      : (action === 'precision_probe' ? precisionProbe.passed : true),
    action,
    state: {
      ready: state.ready,
      playing: state.playing,
      frame_index: state.frameIndex,
      generated_frames: state.motionFrames.length,
      generation_count: state.generationCount,
      waypoint_count: state.waypoints.length,
      postprocess_mode: state.postprocessMode,
    },
    benchmark: action === 'benchmark' ? window.__ardyBenchmark?.result : undefined,
    validation: action === 'validate' ? validation : undefined,
    precision_probe: action === 'precision_probe' ? precisionProbe : undefined,
  };
}

async function pollDemoCommand() {
  if (!serverBridgeEnabled) return;
  if (demoCommandBusy || !state.ready || state.generating || state.benchmarking) return;
  demoCommandBusy = true;
  let job = null;
  let result;
  try {
    const response = await fetch(`/api/demo/next?client_id=${encodeURIComponent(clientId)}&protocol=${frontendProtocol}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`demo command poll HTTP ${response.status}`);
    ({ job } = await response.json());
    if (!job) return;
    result = await executeDemoCommand(job);
  } catch (error) {
    console.error(error);
    log(`服务器命令失败: ${error.stack || error}`);
    result = { passed: false, error: String(error.stack || error) };
  } finally {
    if (job) {
      try {
        const complete = await fetch(`/api/demo/complete/${job.job_id}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(result),
        });
        if (!complete.ok) throw new Error(`demo command complete HTTP ${complete.status}`);
      } catch (error) {
        log(`服务器命令回报失败: ${error.stack || error}`);
      }
    }
    demoCommandBusy = false;
  }
}

async function init() {
  resizeCanvas();
  requestAnimationFrame(render);
  try {
    if (!window.isSecureContext || !navigator.gpu) throw new Error('WebGPU 需要安全上下文和支持的浏览器');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('无法获取 WebGPU adapter');
    const info = adapter.info ?? {};
    const hasF16 = adapter.features.has('shader-f16');
    ui.gpuBadge.textContent = `${info.vendor || 'WebGPU'}${hasF16 ? ' · f16' : ''}`;
    state.adapter = {
      vendor: info.vendor || '',
      architecture: info.architecture || '',
      device: info.device || '',
      shader_f16: hasF16,
    };
    ui.gpuBadge.className = 'badge ok';
    if (candidateId && !/^[a-z0-9_-]+$/.test(candidateId)) {
      throw new Error(`candidate id 非法: ${candidateId}`);
    }
    const manifestUrl = demoAssetUrl('manifest.json');
    const response = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`manifest HTTP ${response.status}`);
    state.manifest = await response.json();
    if (state.manifest.schema_version !== 2 || !state.manifest.all_student || state.manifest.nfe !== 1) {
      throw new Error('manifest 不是当前 NFE=1 全 student 版本');
    }
    if (!['fp16', 'fp32'].includes(state.manifest.precision)) {
      throw new Error(`manifest precision 不支持: ${state.manifest.precision}`);
    }
    if (state.manifest.precision === 'fp16' && !hasF16) {
      throw new Error('当前正式模型是 FP16，但浏览器 WebGPU adapter 不支持 shader-f16');
    }
    await loadPromptFeatures();
    const statsResponse = await fetch(demoAssetUrl('motion_stats.json'), { cache: 'no-cache' });
    if (!statsResponse.ok) throw new Error(`motion stats HTTP ${statsResponse.status}`);
    state.motionStats = await statsResponse.json();
    if (state.motionStats.mean.length !== state.manifest.motion_dim || state.motionStats.std_eps.length !== state.manifest.motion_dim) {
      throw new Error('motion stats 维度与模型不一致');
    }
    const projectionPasses = state.manifest.distillation?.root_projection?.passes ?? 0;
    if (candidateId) log(`A/B 候选: ${candidateId}（默认 release 未被替换）`);
    log(`模型 ${state.manifest.model}; ${state.manifest.fps} FPS; history=4; generation=40; waypoint=60; graph root projection=${projectionPasses}`);
    log(`ONNX 总下载 ${(state.manifest.download.total_onnx_bytes / 1048576).toFixed(2)} MiB；显示后处理=${currentPostprocessMode().label}（不回灌模型）。`);
    if (gpuProfilingEnabled) log('ONNX Runtime WebGPU kernel profiler 已开启；这轮只用于 GPU 占比归因。');
    await loadSharedSessions();
    await loadPrecision();
    if (benchmarkMode) await runAutoBenchmark();
  } catch (error) {
    console.error(error);
    ui.gpuBadge.textContent = '初始化失败';
    ui.gpuBadge.className = 'badge bad';
    setLoading(true, '初始化失败', error.message || String(error));
    setStatus('不可用');
    log(error.stack || error.message || String(error));
    telemetry('report', { event: 'init_error', error: String(error.stack || error.message || error) });
  }
}

ui.play.addEventListener('click', togglePlay);
ui.restart.addEventListener('click', restart);
ui.clearWaypoints.addEventListener('click', clearWaypoints);
ui.postprocess.addEventListener('change', () => {
  selectPostprocessMode(ui.postprocess.value).catch((error) => {
    console.error(error);
    log(`切换显示后处理失败: ${error.stack || error}`);
  });
});
ui.prompt.addEventListener('change', () => {
  state.selectedPromptIndex = Number(ui.prompt.value);
  const prompt = selectedPromptEntry();
  log(`切换文本 prompt ${prompt?.prompt_id ?? '?'}: ${prompt?.text || '(unconditional)'}`);
  telemetry('report', { event: 'prompt_changed' });
  if (state.motionFrames.length) generateNext('prompt');
});
ui.canvas.addEventListener('click', onCanvasClick);
ui.timeline.addEventListener('click', onTimelineClick);
window.addEventListener('resize', resizeCanvas);

if (serverBridgeEnabled) {
  setInterval(() => telemetry('heartbeat'), 2000);
  setInterval(pollDemoCommand, 1000);
}

configurePostprocessUi();
init();
