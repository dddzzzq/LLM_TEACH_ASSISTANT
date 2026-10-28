import test from 'node:test';
import assert from 'node:assert/strict';
import { taskResultLink } from './taskResultLink.js';

test('task receipts permit portal result pages while rejecting external or extra query destinations', () => {
  for (const value of [
    '/materials',
    '/assignments/7',
    '/assignments/7?tab=materials&class_id=class-a',
    '/assignments/7?class_id=11111111-1111-4111-8111-111111111111&tab=results',
    '/assignments/7?class_id=__unassigned__',
    '/assignments/7?tab=results&class_id=local%3AQeePrQ',
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
    '/assignments/7?class_id=',
    '/assignments/7?class_id=%20',
    '/assignments/7?class_id=class-a&class_id=class-b',
    '/assignments/7?class_id=https://evil.example',
    '/assignments/7?class_id=class-a%0A',
    '/assignments/7?class_id=local:',
    '/assignments/7?class_id=local:../secret',
    '/assignments/7?class_id=class-a&next=https://evil.example',
    '/assignments/7?class_id=class-a#https://evil.example',
    '/assignments/7?tab=https://evil.example&class_id=class-a',
    'https://evil.example/assignments/7?class_id=class-a',
    '/grade-homework?workflow=invalid',
    '/grade-homework?workflow=11111111-1111-4111-8111-111111111111&next=https://evil.example',
  ])
    assert.equal(taskResultLink(value), '');
});
