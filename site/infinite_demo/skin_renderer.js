// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const ASSET_SCHEMA = 'ardy_core_skin_web_v1';
const FLOAT_BYTES = Float32Array.BYTES_PER_ELEMENT;

function requireCondition(condition, message) {
  if (!condition) throw new Error(`CoreSkin asset: ${message}`);
}

export function sectionView(metadata, binary, name) {
  const section = metadata.sections?.[name];
  requireCondition(section, `missing section ${name}`);
  const constructors = {
    float32: Float32Array,
    uint16: Uint16Array,
    uint8: Uint8Array,
  };
  const TypedArray = constructors[section.dtype];
  requireCondition(TypedArray, `unsupported dtype ${section.dtype}`);
  const byteLength = section.length * TypedArray.BYTES_PER_ELEMENT;
  requireCondition(section.offset_bytes % TypedArray.BYTES_PER_ELEMENT === 0, `${name} is misaligned`);
  requireCondition(section.offset_bytes >= 0 && section.offset_bytes + byteLength <= binary.byteLength, `${name} is out of range`);
  return new TypedArray(binary, section.offset_bytes, section.length);
}

export function validateCoreSkinMetadata(metadata, binaryByteLength) {
  requireCondition(metadata.schema === ASSET_SCHEMA, `unexpected schema ${metadata.schema}`);
  requireCondition(metadata.joint_count === 27, `expected 27 joints, got ${metadata.joint_count}`);
  requireCondition(metadata.influences_per_vertex === 5, 'expected five LBS influences');
  requireCondition(metadata.vertex_count > 0 && metadata.vertex_count <= 65535, 'invalid vertex count');
  requireCondition(metadata.triangle_count > 0, 'invalid triangle count');
  requireCondition(metadata.binary_size_bytes === binaryByteLength, 'binary size mismatch');
  const expectedLengths = {
    positions: metadata.vertex_count * 3,
    normals: metadata.vertex_count * 3,
    indices: metadata.triangle_count * 3,
    joint_indices_0: metadata.vertex_count * 4,
    joint_indices_1: metadata.vertex_count,
    weights_0: metadata.vertex_count * 4,
    weights_1: metadata.vertex_count,
    bind_transforms: metadata.joint_count * 16,
    inverse_bind_transforms: metadata.joint_count * 16,
  };
  for (const [name, length] of Object.entries(expectedLengths)) {
    requireCondition(metadata.sections?.[name]?.length === length, `${name} length mismatch`);
  }
}

function sha256Hex(buffer) {
  return crypto.subtle.digest('SHA-256', buffer).then((digest) => (
    [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('')
  ));
}

/**
 * Build the same posed_transform @ inverse_bind transform used by CoreSkin.lbs.
 * Input rotations and inverse binds are row-major; WebGL matrices are emitted column-major.
 */
export function buildBoneMatrices(globalRotations, jointPositions, inverseBind, output = null) {
  const jointCount = jointPositions.length / 3;
  requireCondition(Number.isInteger(jointCount), 'joint position length is invalid');
  requireCondition(globalRotations.length === jointCount * 9, 'rotation length mismatch');
  requireCondition(inverseBind.length === jointCount * 16, 'inverse bind length mismatch');
  const result = output ?? new Float32Array(jointCount * 16);
  requireCondition(result.length === jointCount * 16, 'bone output length mismatch');

  for (let joint = 0; joint < jointCount; joint += 1) {
    const rotationBase = joint * 9;
    const positionBase = joint * 3;
    const bindBase = joint * 16;
    const outputBase = joint * 16;
    for (let row = 0; row < 3; row += 1) {
      const r0 = globalRotations[rotationBase + row * 3];
      const r1 = globalRotations[rotationBase + row * 3 + 1];
      const r2 = globalRotations[rotationBase + row * 3 + 2];
      const translation = jointPositions[positionBase + row];
      for (let column = 0; column < 4; column += 1) {
        result[outputBase + column * 4 + row] = (
          r0 * inverseBind[bindBase + column]
          + r1 * inverseBind[bindBase + 4 + column]
          + r2 * inverseBind[bindBase + 8 + column]
          + translation * inverseBind[bindBase + 12 + column]
        );
      }
    }
    for (let column = 0; column < 4; column += 1) {
      result[outputBase + column * 4 + 3] = inverseBind[bindBase + 12 + column];
    }
  }
  return result;
}

export function skinVertexCpu(position, jointIndices, weights, boneMatrices, output = new Float32Array(3)) {
  requireCondition(position.length === 3, 'CPU skin position must have three values');
  requireCondition(jointIndices.length === weights.length, 'CPU skin influence length mismatch');
  let outX = 0;
  let outY = 0;
  let outZ = 0;
  for (let influence = 0; influence < weights.length; influence += 1) {
    const weight = weights[influence];
    const base = jointIndices[influence] * 16;
    outX += weight * (
      boneMatrices[base] * position[0]
      + boneMatrices[base + 4] * position[1]
      + boneMatrices[base + 8] * position[2]
      + boneMatrices[base + 12]
    );
    outY += weight * (
      boneMatrices[base + 1] * position[0]
      + boneMatrices[base + 5] * position[1]
      + boneMatrices[base + 9] * position[2]
      + boneMatrices[base + 13]
    );
    outZ += weight * (
      boneMatrices[base + 2] * position[0]
      + boneMatrices[base + 6] * position[1]
      + boneMatrices[base + 10] * position[2]
      + boneMatrices[base + 14]
    );
  }
  output[0] = outX;
  output[1] = outY;
  output[2] = outZ;
  return output;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader) || 'unknown shader error';
    gl.deleteShader(shader);
    throw new Error(`CoreSkin WebGL shader compile failed: ${detail}`);
  }
  return shader;
}

function createProgram(gl) {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, `#version 300 es
    precision highp float;
    precision highp int;
    layout(location = 0) in vec3 aPosition;
    layout(location = 1) in vec3 aNormal;
    layout(location = 2) in uvec4 aJoints0;
    layout(location = 3) in uint aJoint1;
    layout(location = 4) in vec4 aWeights0;
    layout(location = 5) in float aWeight1;
    uniform mat4 uBones[27];
    uniform vec2 uViewport;
    uniform vec2 uCameraXZ;
    uniform float uScale;
    out vec3 vNormal;

    void main() {
      mat4 skin = (
        aWeights0.x * uBones[aJoints0.x]
        + aWeights0.y * uBones[aJoints0.y]
        + aWeights0.z * uBones[aJoints0.z]
        + aWeights0.w * uBones[aJoints0.w]
        + aWeight1 * uBones[aJoint1]
      );
      vec4 world = skin * vec4(aPosition, 1.0);
      float screenX = uViewport.x * 0.54
        + ((world.x - uCameraXZ.x) - (world.z - uCameraXZ.y)) * uScale;
      float screenY = uViewport.y * 0.64
        + ((world.x - uCameraXZ.x) + (world.z - uCameraXZ.y)) * uScale * 0.34
        - world.y * uScale;
      vec2 clip = vec2(
        screenX * 2.0 / uViewport.x - 1.0,
        1.0 - screenY * 2.0 / uViewport.y
      );
      float viewDepth = (world.x - uCameraXZ.x) + (world.z - uCameraXZ.y) + 0.68 * world.y;
      gl_Position = vec4(clip, clamp(-viewDepth / 32.0, -0.95, 0.95), 1.0);
      vNormal = normalize(mat3(skin) * aNormal);
    }
  `);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    in vec3 vNormal;
    out vec4 outColor;

    void main() {
      vec3 normal = gl_FrontFacing ? normalize(vNormal) : -normalize(vNormal);
      vec3 lightDirection = normalize(vec3(-0.35, 0.86, 0.38));
      float diffuse = max(dot(normal, lightDirection), 0.0);
      float fill = 0.78 + diffuse * 0.22;
      vec3 officialLightColor = vec3(152.0, 189.0, 255.0) / 255.0;
      outColor = vec4(officialLightColor * fill, 1.0);
    }
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const detail = gl.getProgramInfoLog(program) || 'unknown link error';
    gl.deleteProgram(program);
    throw new Error(`CoreSkin WebGL program link failed: ${detail}`);
  }
  return program;
}

function uploadAttribute(gl, location, values, size, type, integer = false) {
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(location);
  if (integer) gl.vertexAttribIPointer(location, size, type, 0, 0);
  else gl.vertexAttribPointer(location, size, type, false, 0, 0);
  return buffer;
}

export class CoreSkinRenderer {
  constructor(canvas, metadata, binary) {
    validateCoreSkinMetadata(metadata, binary.byteLength);
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      depth: true,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('浏览器未提供 WebGL2，无法显示官方 CoreSkin 人体模型');

    this.canvas = canvas;
    this.gl = gl;
    this.metadata = metadata;
    this.program = createProgram(gl);
    this.vao = gl.createVertexArray();
    this.buffers = [];
    this.boneMatrices = new Float32Array(metadata.joint_count * 16);
    this.inverseBind = new Float32Array(sectionView(metadata, binary, 'inverse_bind_transforms'));
    this.indexCount = metadata.triangle_count * 3;
    this.lost = false;
    this.checkedFirstDraw = false;

    if (gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) < metadata.joint_count * 4) {
      throw new Error('WebGL2 vertex uniform 容量不足，无法上传 27 个蒙皮矩阵');
    }

    canvas.addEventListener('webglcontextlost', (event) => {
      event.preventDefault();
      this.lost = true;
    });

    gl.bindVertexArray(this.vao);
    this.buffers.push(uploadAttribute(gl, 0, sectionView(metadata, binary, 'positions'), 3, gl.FLOAT));
    this.buffers.push(uploadAttribute(gl, 1, sectionView(metadata, binary, 'normals'), 3, gl.FLOAT));
    this.buffers.push(uploadAttribute(gl, 2, sectionView(metadata, binary, 'joint_indices_0'), 4, gl.UNSIGNED_BYTE, true));
    this.buffers.push(uploadAttribute(gl, 3, sectionView(metadata, binary, 'joint_indices_1'), 1, gl.UNSIGNED_BYTE, true));
    this.buffers.push(uploadAttribute(gl, 4, sectionView(metadata, binary, 'weights_0'), 4, gl.FLOAT));
    this.buffers.push(uploadAttribute(gl, 5, sectionView(metadata, binary, 'weights_1'), 1, gl.FLOAT));
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, sectionView(metadata, binary, 'indices'), gl.STATIC_DRAW);
    this.buffers.push(indexBuffer);
    gl.bindVertexArray(null);

    gl.useProgram(this.program);
    this.uniforms = {
      bones: gl.getUniformLocation(this.program, 'uBones[0]'),
      viewport: gl.getUniformLocation(this.program, 'uViewport'),
      cameraXZ: gl.getUniformLocation(this.program, 'uCameraXZ'),
      scale: gl.getUniformLocation(this.program, 'uScale'),
    };
    for (const [name, location] of Object.entries(this.uniforms)) {
      if (location === null) throw new Error(`CoreSkin WebGL uniform missing: ${name}`);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
  }

  resize(width, height, dpr) {
    const pixelWidth = Math.max(1, Math.round(width * dpr));
    const pixelHeight = Math.max(1, Math.round(height * dpr));
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    this.gl.viewport(0, 0, pixelWidth, pixelHeight);
  }

  clear(width, height, dpr) {
    if (this.lost || this.gl.isContextLost()) return;
    this.resize(width, height, dpr);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
  }

  render({ globalRotations, jointPositions, width, height, dpr, cameraX, cameraZ, scale }) {
    if (this.lost || this.gl.isContextLost()) throw new Error('CoreSkin WebGL2 context 已丢失');
    this.resize(width, height, dpr);
    buildBoneMatrices(globalRotations, jointPositions, this.inverseBind, this.boneMatrices);
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniformMatrix4fv(this.uniforms.bones, false, this.boneMatrices);
    gl.uniform2f(this.uniforms.viewport, width, height);
    gl.uniform2f(this.uniforms.cameraXZ, cameraX, cameraZ);
    gl.uniform1f(this.uniforms.scale, scale);
    gl.drawElements(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
    if (!this.checkedFirstDraw) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) throw new Error(`CoreSkin WebGL 首帧错误: 0x${error.toString(16)}`);
      this.checkedFirstDraw = true;
    }
  }
}

export async function loadCoreSkinRenderer(canvas, metadataUrl) {
  const resolvedMetadataUrl = new URL(metadataUrl, window.location.href);
  const metadataResponse = await fetch(resolvedMetadataUrl, { cache: 'no-cache' });
  if (!metadataResponse.ok) throw new Error(`CoreSkin metadata HTTP ${metadataResponse.status}`);
  const metadata = await metadataResponse.json();
  const binaryUrl = new URL(metadata.binary_url, resolvedMetadataUrl);
  const binaryResponse = await fetch(binaryUrl, { cache: 'no-cache' });
  if (!binaryResponse.ok) throw new Error(`CoreSkin binary HTTP ${binaryResponse.status}`);
  const binary = await binaryResponse.arrayBuffer();
  validateCoreSkinMetadata(metadata, binary.byteLength);
  const digest = await sha256Hex(binary);
  if (digest !== metadata.binary_sha256) throw new Error('CoreSkin binary SHA256 不匹配');
  return new CoreSkinRenderer(canvas, metadata, binary);
}
