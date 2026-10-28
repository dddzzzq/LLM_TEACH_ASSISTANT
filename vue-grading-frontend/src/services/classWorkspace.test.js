import test from 'node:test';
import assert from 'node:assert/strict';
import {
  chooseClass,
  finalScore,
  filterClassResults,
  classCanImport,
  canImportClass,
} from './classWorkspace.js';
test('class selection validates a deep link and discards stale remembered classes', () => {
  const classes = [{ id: 'a' }, { id: 'b' }, { id: '__unassigned__' }];
  assert.equal(chooseClass(classes, 'b', 'a'), 'b');
  assert.equal(chooseClass(classes, 'foreign-class', 'a'), 'a');
  assert.equal(chooseClass(classes, '', 'deleted-class'), 'a');
  assert.equal(chooseClass([], 'a', 'a'), '');
  assert.equal(classCanImport({ source: 'history' }), false);
});
test('imports cannot silently write a named class while viewing historical unassigned data', () => {
  assert.equal(canImportClass({ class_name: '一班' }, { source: 'unassigned' }), false);
  assert.equal(canImportClass({ class_name: '一班' }, { source: 'local', name: '二班' }), false);
  assert.equal(canImportClass({ class_name: '一班' }, { source: 'local', name: '一班' }), true);
  assert.equal(
    canImportClass({ class_name: '全部班级', portal_homework_id: 'h' }, { source: 'unassigned' }),
    true
  );
  assert.equal(
    canImportClass(
      { class_name: '全部班级', portal_homework_id: 'h' },
      { source: 'history', name: '旧班' }
    ),
    false
  );
});
test('manual zero is retained and unreviewed placeholder human scores are ignored', () => {
  assert.equal(finalScore({ score: 90, human_score: 0, is_human_reviewed: true }), 0);
  assert.equal(finalScore({ score: 90, human_score: 0, is_human_reviewed: false }), 90);
  const rows = [
    { student_id: '1', student_name: '陈同学', score: 90, human_score: 0, is_human_reviewed: true },
    { student_id: '2', score: 70 },
  ];
  assert.equal(filterClassResults(rows, '陈', 'reviewed')[0].student_id, '1');
  assert.deepEqual(
    filterClassResults(rows).map((row) => row.student_id),
    ['2', '1']
  );
});
