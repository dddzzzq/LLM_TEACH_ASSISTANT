import test from 'node:test';
import assert from 'node:assert/strict';
import { createPortalApi, operationActive } from './portalApi.js';

test('credentials travel in settings bodies and browser controls use the existing RPA service', async () => {
  const calls = [];
  const client = Object.fromEntries(
    ['get', 'post', 'put', 'delete'].map((method) => [
      method,
      (...args) => {
        calls.push([method, ...args]);
        return Promise.resolve({ data: {} });
      },
    ])
  );
  const api = createPortalApi(client);
  await api.save({ username: 'teacher', password: 'test-only-password', trust_device: true });
  await api.createWorkflow(
    { source: 'portal', snapshot_id: 's1', items: [{ homework_id: 'h1' }] },
    'request-key'
  );
  assert.equal(calls[0][1], '/api/portal/connection');
  assert.equal(calls[0][2].password, 'test-only-password');
  assert.equal(calls[1][3].headers['Idempotency-Key'], 'request-key');
  assert.ok(calls.every((call) => !/test-only-password/.test(call[1])));
  assert.equal(api.auth, undefined);
  assert.equal(api.browser, undefined);
});

test('authentication waits remain active while completed operations stop polling controls', () => {
  for (const status of ['WAITING_AUTH', 'WAITING_OTP', 'WAITING_IDENTITY']) {
    assert.equal(operationActive({ status }), true);
  }
  for (const status of ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED', 'EXPIRED']) {
    assert.equal(operationActive({ status }), false);
  }
});

test('assignment grading distinguishes concrete classes from server-expanded all classes', async () => {
  const calls = [];
  const api = createPortalApi({
    post: (...args) => {
      calls.push(args);
      return Promise.resolve({});
    },
  });
  await api.gradeClasses(11, ['class-b'], 'single');
  await api.gradeClasses(11, [], 'all', true);
  assert.deepEqual(calls[0][1], { class_ids: ['class-b'] });
  assert.deepEqual(calls[1][1], { class_ids: [], all_classes: true });
  assert.equal(calls[1][2].headers['Idempotency-Key'], 'all');
});
