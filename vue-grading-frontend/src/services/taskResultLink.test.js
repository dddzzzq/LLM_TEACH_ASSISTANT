import test from 'node:test';
import assert from 'node:assert/strict';
import { taskResultLink } from './taskResultLink.js';

test('task receipts permit portal result pages while rejecting external or extra query destinations', () => {
  for (const value of [
    '/materials',
    '/assignments/7',
    '/settings/portal',
    '/grade-homework?workflow=11111111-1111-4111-8111-111111111111',
  ])
    assert.equal(taskResultLink(value), value);
  for (const value of [
    '/exams/8',
    null,
    {},
    'javascript:alert(1)',
    '//evil.example',
    'https://evil.example',
    '/settings/portal?password=secret',
    '/grade-homework?workflow=invalid',
    '/grade-homework?workflow=11111111-1111-4111-8111-111111111111&next=https://evil.example',
  ])
    assert.equal(taskResultLink(value), '');
});
