export const assignmentStates = {
  EMPTY: { label: '待导入', tone: 'neutral' },
  PREPARING: { label: '整理中', tone: 'blue' },
  GRADING: { label: '批改中', tone: 'blue' },
  ATTENTION: { label: '待处理', tone: 'amber' },
  READY: { label: '待批改', tone: 'purple' },
  GRADED: { label: '已有结果', tone: 'green' },
};
export function assignmentState(assignment) {
  return assignmentStates[assignment?.summary?.status] || { label: '状态待同步', tone: 'neutral' };
}
export function filterAssignments(assignments, { search = '', course = '', status = '' } = {}) {
  const query = search.trim().toLocaleLowerCase();
  return assignments.filter(
    (item) =>
      (!course || item.course_name === course) &&
      (!status || item.summary?.status === status) &&
      (!query ||
        [item.task_name, item.course_name, item.question].some((value) =>
          String(value || '')
            .toLocaleLowerCase()
            .includes(query)
        ))
  );
}
export function rubricRows(rubric) {
  let value = rubric;
  if (typeof value === 'string') {
    try {
      value = JSON.parse(value);
    } catch {
      return [{ name: '评分说明', description: value, score: null }];
    }
  }
  if (!value || typeof value !== 'object') return [];
  return Object.entries(value).map(([key, item]) => ({
    name: Array.isArray(value) ? item?.name || item?.criterion || `评分项 ${Number(key) + 1}` : key,
    score: typeof item === 'number' ? item : typeof item?.score === 'number' ? item.score : null,
    description: typeof item === 'string' ? item : item?.description || item?.requirement || '',
  }));
}
export function formatActivity(stamp) {
  if (!stamp) return '尚无处理记录';
  const date = new Date(stamp);
  return Number.isNaN(date.getTime())
    ? '时间未知'
    : new Intl.DateTimeFormat('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }).format(date);
}
