import test from 'node:test';
import assert from 'node:assert/strict';
import { createMaterialsApi } from './materialsApi.js';

test('material API preserves filters, response objects and request payloads', async () => {
  const calls = [];
  const response = { data: { items: [] } };
  const client = Object.fromEntries(
    ['get', 'post'].map((method) => [
      method,
      async (...args) => {
        calls.push([method, ...args]);
        return response;
      },
    ])
  );
  const api = createMaterialsApi(client);
  assert.equal(await api.list({ assignmentId: 42, sourceJobId: 'job-1' }), response);
  await api.list();
  const body = { assignment_id: 42, source_path: '/data', auto_grade: false };
  await api.create(body, 'retry-safe-key');
  await api.get('batch-1');
  await api.retry('batch-1');
  await api.grade('batch-1', false);
  await api.grade('batch-1', true);
  await api.cancel('batch-1');
  assert.deepEqual(calls, [
    ['get', '/api/materials', { params: { assignment_id: 42, source_job_id: 'job-1' } }],
    ['get', '/api/materials', { params: { assignment_id: undefined, source_job_id: undefined } }],
    ['post', '/api/materials', body, { headers: { 'Idempotency-Key': 'retry-safe-key' } }],
    ['get', '/api/materials/batch-1'],
    ['post', '/api/materials/batch-1/retry'],
    ['post', '/api/materials/batch-1/grade', { allow_partial: false }],
    ['post', '/api/materials/batch-1/grade', { allow_partial: true }],
    ['post', '/api/materials/batch-1/cancel'],
  ]);
});

test('material API lets the component handle the original HTTP error', async () => {
  const failure = { response: { status: 409, data: { error: 'already running' } } };
  const api = createMaterialsApi({
    post: async () => {
      throw failure;
    },
  });
  await assert.rejects(api.grade('batch-1', false), (error) => error === failure);
});
