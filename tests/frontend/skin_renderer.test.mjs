import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import test from 'node:test';

import {
  buildBoneMatrices,
  sectionView,
  skinVertexCpu,
  validateCoreSkinMetadata,
} from '../../site/infinite_demo/skin_renderer.js';

const metadata = JSON.parse(fs.readFileSync(
  new URL('../../site/infinite_demo/assets/core_skin.json', import.meta.url),
  'utf8',
));
const nodeBuffer = fs.readFileSync(new URL('../../site/infinite_demo/assets/core_skin.bin', import.meta.url));
const binary = nodeBuffer.buffer.slice(
  nodeBuffer.byteOffset,
  nodeBuffer.byteOffset + nodeBuffer.byteLength,
);

test('CoreSkin browser bundle is complete and checksum-verified', () => {
  validateCoreSkinMetadata(metadata, binary.byteLength);
  assert.equal(createHash('sha256').update(nodeBuffer).digest('hex'), metadata.binary_sha256);
  assert.equal(metadata.vertex_count, 9084);
  assert.equal(metadata.triangle_count, 18152);
  assert.equal(Math.max(...sectionView(metadata, binary, 'indices')), metadata.vertex_count - 1);
});

test('WebGL column-major bone matrices reproduce the official bind pose', () => {
  const bind = sectionView(metadata, binary, 'bind_transforms');
  const inverseBind = sectionView(metadata, binary, 'inverse_bind_transforms');
  const rotations = new Float32Array(metadata.joint_count * 9);
  const positions = new Float32Array(metadata.joint_count * 3);
  for (let joint = 0; joint < metadata.joint_count; joint += 1) {
    const bindBase = joint * 16;
    for (let row = 0; row < 3; row += 1) {
      positions[joint * 3 + row] = bind[bindBase + row * 4 + 3];
      for (let column = 0; column < 3; column += 1) {
        rotations[joint * 9 + row * 3 + column] = bind[bindBase + row * 4 + column];
      }
    }
  }
  const bones = buildBoneMatrices(rotations, positions, inverseBind);
  let matrixMaxError = 0;
  for (let joint = 0; joint < metadata.joint_count; joint += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        const expected = row === column ? 1 : 0;
        matrixMaxError = Math.max(
          matrixMaxError,
          Math.abs(bones[joint * 16 + column * 4 + row] - expected),
        );
      }
    }
  }
  assert.ok(matrixMaxError < 1e-5, `bind bone matrix error ${matrixMaxError}`);

  const bindVertices = sectionView(metadata, binary, 'positions');
  const joints0 = sectionView(metadata, binary, 'joint_indices_0');
  const joints1 = sectionView(metadata, binary, 'joint_indices_1');
  const weights0 = sectionView(metadata, binary, 'weights_0');
  const weights1 = sectionView(metadata, binary, 'weights_1');
  const vertex = new Float32Array(3);
  const jointIndices = new Uint8Array(5);
  const weights = new Float32Array(5);
  const skinned = new Float32Array(3);
  let vertexMaxError = 0;
  let weightMaxError = 0;
  for (let index = 0; index < metadata.vertex_count; index += 1) {
    vertex.set(bindVertices.subarray(index * 3, index * 3 + 3));
    jointIndices.set(joints0.subarray(index * 4, index * 4 + 4));
    jointIndices[4] = joints1[index];
    weights.set(weights0.subarray(index * 4, index * 4 + 4));
    weights[4] = weights1[index];
    weightMaxError = Math.max(
      weightMaxError,
      Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1),
    );
    skinVertexCpu(vertex, jointIndices, weights, bones, skinned);
    for (let axis = 0; axis < 3; axis += 1) {
      vertexMaxError = Math.max(vertexMaxError, Math.abs(skinned[axis] - vertex[axis]));
    }
  }
  assert.ok(weightMaxError < 2e-6, `weight sum error ${weightMaxError}`);
  assert.ok(vertexMaxError < 2e-5, `bind vertex error ${vertexMaxError}`);
});

test('non-bind five-weight pose matches an independent row-major LBS reference', () => {
  const inverseBind = sectionView(metadata, binary, 'inverse_bind_transforms');
  const rotations = new Float32Array(metadata.joint_count * 9);
  const positions = new Float32Array(metadata.joint_count * 3);
  for (let joint = 0; joint < metadata.joint_count; joint += 1) {
    const ax = (joint + 1) * 0.037;
    const ay = (joint + 1) * -0.029;
    const az = (joint + 1) * 0.021;
    const [sx, cx] = [Math.sin(ax), Math.cos(ax)];
    const [sy, cy] = [Math.sin(ay), Math.cos(ay)];
    const [sz, cz] = [Math.sin(az), Math.cos(az)];
    rotations.set([
      cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
      -sy, cy * sx, cy * cx,
    ], joint * 9);
    positions.set([
      Math.sin(joint * 0.31) * 0.7,
      0.8 + joint * 0.017,
      Math.cos(joint * 0.23) * 0.6,
    ], joint * 3);
  }
  const bones = buildBoneMatrices(rotations, positions, inverseBind);
  const bindVertices = sectionView(metadata, binary, 'positions');
  const joints0 = sectionView(metadata, binary, 'joint_indices_0');
  const joints1 = sectionView(metadata, binary, 'joint_indices_1');
  const weights0 = sectionView(metadata, binary, 'weights_0');
  const weights1 = sectionView(metadata, binary, 'weights_1');
  const jointIndices = new Uint8Array(5);
  const weights = new Float32Array(5);
  const actual = new Float32Array(3);
  let maxError = 0;
  let meanError = 0;
  let values = 0;

  for (let vertex = 0; vertex < metadata.vertex_count; vertex += 1) {
    const source = bindVertices.subarray(vertex * 3, vertex * 3 + 3);
    jointIndices.set(joints0.subarray(vertex * 4, vertex * 4 + 4));
    jointIndices[4] = joints1[vertex];
    weights.set(weights0.subarray(vertex * 4, vertex * 4 + 4));
    weights[4] = weights1[vertex];
    skinVertexCpu(source, jointIndices, weights, bones, actual);
    const expected = [0, 0, 0];
    for (let influence = 0; influence < 5; influence += 1) {
      const joint = jointIndices[influence];
      const rotationBase = joint * 9;
      const positionBase = joint * 3;
      const bindBase = joint * 16;
      for (let row = 0; row < 3; row += 1) {
        let transformed = 0;
        for (let column = 0; column < 4; column += 1) {
          const affine = (
            rotations[rotationBase + row * 3] * inverseBind[bindBase + column]
            + rotations[rotationBase + row * 3 + 1] * inverseBind[bindBase + 4 + column]
            + rotations[rotationBase + row * 3 + 2] * inverseBind[bindBase + 8 + column]
            + positions[positionBase + row] * inverseBind[bindBase + 12 + column]
          );
          transformed += affine * (column < 3 ? source[column] : 1);
        }
        expected[row] += weights[influence] * transformed;
      }
    }
    for (let axis = 0; axis < 3; axis += 1) {
      const error = Math.abs(actual[axis] - expected[axis]);
      maxError = Math.max(maxError, error);
      meanError += error;
      values += 1;
    }
  }
  meanError /= values;
  assert.ok(maxError < 2e-6, `dynamic vertex error ${maxError}`);
  assert.ok(meanError < 1e-7, `dynamic vertex mean error ${meanError}`);
});
