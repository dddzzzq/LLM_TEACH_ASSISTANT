import test from 'node:test';
import assert from 'node:assert/strict';
import {
  filterAssignments,
  assignmentState,
  rubricRows,
  formatActivity,
} from './assignmentWorkspace.js';
test('assignment filters combine course, status and content without changing source', () => {
  const source = [
    {
      id: 1,
      course_name: '软件工程',
      task_name: '实验一',
      question: 'API设计',
      summary: { status: 'READY' },
    },
    { id: 2, course_name: '软件工程', task_name: '实验二', summary: { status: 'GRADED' } },
    { id: 3, course_name: '其他课程', question: 'api' },
  ];
  assert.deepEqual(
    filterAssignments(source, { course: '软件工程', status: 'READY', search: ' api ' }).map(
      (item) => item.id
    ),
    [1]
  );
  assert.equal(source.length, 3);
  assert.equal(filterAssignments(source, { search: '不存在' }).length, 0);
  assert.equal(assignmentState({}).label, '状态待同步');
  assert.equal(formatActivity(null), '尚无处理记录');
});
test('readable rubric supports numeric, structured and legacy descriptions', () => {
  assert.deepEqual(rubricRows({ 功能: 60, 文档: { score: 40, description: '说明设计' } }), [
    { name: '功能', score: 60, description: '' },
    { name: '文档', score: 40, description: '说明设计' },
  ]);
  assert.equal(rubricRows('{"代码":20}')[0].score, 20);
  assert.equal(rubricRows('按要求完成')[0].description, '按要求完成');
  assert.deepEqual(rubricRows(null), []);
});
