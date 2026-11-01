export const UNASSIGNED_CLASS = '__unassigned__';
export const classCanImport = (item) => item?.source === 'portal';
export function canImportClass(assignment, item) {
  if (!assignment || !item) return false;
  if (item.can_import) return true;
  if (item.source === 'unassigned')
    return ['', '全部班级', '所有班级', '未分班', '未分班材料', '历史未分班成绩'].includes(
      (assignment.class_name || '').trim()
    );
  return (
    !assignment.portal_homework_id && item.source === 'local' && item.name === assignment.class_name
  );
}
export function chooseClass(classes, requested, remembered) {
  return (
    classes.find((item) => item.id === requested)?.id ||
    classes.find((item) => item.id === remembered)?.id ||
    classes[0]?.id ||
    ''
  );
}
export function classUnreviewed(item) {
  return Math.max(0, (item?.graded_count || 0) - (item?.reviewed_count || 0));
}
export function classSummary(item) {
  const unreviewed = classUnreviewed(item);
  const detail = item?.graded_count
    ? `${item.reviewed_count || 0} / ${item.graded_count} 已复核`
    : '';
  if (item?.issue_count > 0 || item?.material_status === 'ATTENTION')
    return {
      label: item.issue_count > 0 ? `${item.issue_count} 项材料异常` : '材料需要处理',
      tone: 'warning',
      detail,
    };
  if (item?.grading_status === 'ATTENTION')
    return { label: '批改需要处理', tone: 'warning', detail };
  if (item?.grading_status === 'GRADING') return { label: '批改中', tone: 'active', detail };
  if (item?.material_status === 'PROCESSING')
    return { label: '材料整理中', tone: 'active', detail };
  if (unreviewed) return { label: `待复核 ${unreviewed}`, tone: 'warning', detail };
  if (item?.graded_count > 0) return { label: '成绩已复核', tone: 'success', detail };
  if (item?.material_status === 'READY')
    return {
      label: item?.grading_status === 'READY' ? '待批改' : '材料就绪',
      tone: 'active',
      detail,
    };
  return { label: '待导入', tone: 'muted', detail };
}
export function finalScore(result) {
  return result.is_human_reviewed && Number.isFinite(result.human_score)
    ? result.human_score
    : result.score;
}
export function filterClassResults(results, search = '', review = '', sort = 'review') {
  const query = search.trim().toLocaleLowerCase();
  return results
    .filter(
      (result) =>
        (!query ||
          [result.student_id, result.student_name].some((value) =>
            String(value || '')
              .toLocaleLowerCase()
              .includes(query)
          )) &&
        (!review || (review === 'reviewed' ? result.is_human_reviewed : !result.is_human_reviewed))
    )
    .sort((a, b) => {
      if (sort === 'score_asc') return finalScore(a) - finalScore(b);
      if (sort === 'score_desc') return finalScore(b) - finalScore(a);
      return (
        Number(Boolean(a.is_human_reviewed)) - Number(Boolean(b.is_human_reviewed)) ||
        finalScore(b) - finalScore(a)
      );
    });
}
