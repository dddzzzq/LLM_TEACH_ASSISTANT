import test from 'node:test';
import assert from 'node:assert/strict';
import { mainNavigation, isTeacherAccount, isHomeworkPage } from './teacherWorkspace.js';

test('teacher workspace has two entry points and preserves homework subpages', () => {
  assert.deepEqual(
    mainNavigation.map((item) => item.path),
    ['/assignments', '/ai-assistant']
  );
  for (const path of ['/assignments/12', '/assignments/new', '/materials', '/settings/portal'])
    assert.equal(isHomeworkPage(path), true);
  for (const path of ['/exams/2', '/ai-assistant']) assert.equal(isHomeworkPage(path), false);
  assert.equal(isTeacherAccount('teacher'), true);
  assert.equal(isTeacherAccount('admin'), true);
  assert.equal(isTeacherAccount('student'), false);
});
