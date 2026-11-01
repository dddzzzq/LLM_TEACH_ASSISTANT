import test from 'node:test';
import assert from 'node:assert/strict';
import {
  rubricTemplates,
  createRubricFromTemplate,
  matchingRubricTemplate,
} from './rubricTemplates.js';
import { rubricRows } from './assignmentWorkspace.js';

test('the four templates produce complete 100-point rubrics in the existing save and display format', () => {
  assert.deepEqual(
    rubricTemplates.map((item) => item.name),
    ['答题解题类', '工程代码类', '论文研究类', '展示汇报类']
  );
  for (const template of rubricTemplates) {
    const draft = createRubricFromTemplate(template.id);
    const rows = rubricRows(draft);
    assert.equal(
      rows.reduce((total, item) => total + item.score, 0),
      100
    );
    assert.ok(rows.every((item) => item.name && item.description && item.score > 0));
    // Server serialization can change key order without changing the template.
    const saved = Object.fromEntries(Object.entries(JSON.parse(draft)).reverse());
    assert.equal(matchingRubricTemplate(JSON.stringify(saved))?.id, template.id);
  }
  assert.equal(matchingRubricTemplate(createRubricFromTemplate())?.id, 'problem_solving');
});

test('editing one assignment keeps other assignment drafts and built-in templates unchanged', () => {
  const first = JSON.parse(createRubricFromTemplate('engineering_code'));
  const second = createRubricFromTemplate('engineering_code');
  first['功能与需求完成度'].score = 40;
  first['教师补充项'] = { score: 5, description: '本次作业的补充要求' };
  assert.equal(matchingRubricTemplate(first), null);
  assert.equal(createRubricFromTemplate('engineering_code'), second);
  assert.equal(JSON.parse(second)['功能与需求完成度'].score, 45);
  assert.throws(() => createRubricFromTemplate('missing'));
});

test('legacy, malformed and extended rubrics remain custom without losing their content', () => {
  const legacy = { 功能: 60, 文档: { score: 40, description: '教师原标准' } };
  const original = JSON.stringify(legacy);
  for (const value of [legacy, '{}', '', '{', null, [], { 说明: '按题目评分' }]) {
    assert.equal(matchingRubricTemplate(value), null);
  }
  assert.equal(JSON.stringify(legacy), original);
  const extended = JSON.parse(createRubricFromTemplate('research_paper'));
  extended['分析与论证'].扣分规则 = '保留教师补充规则';
  assert.equal(matchingRubricTemplate(extended), null);
});
